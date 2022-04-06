import os
import tempfile
import tvm
from tvm import tir, relay
import tvm.testing
import numpy as np
from tvm.meta_schedule.tune import extract_task_from_relay, Parse
from tvm.meta_schedule.integration import ApplyHistoryBest
import tvm.topi.testing
from tvm.meta_schedule.database import TuningRecord, JSONDatabase


def schedule_matmul_common(sch, block, batched, M, decompose_reduction=True):
    a_y, a_x, _ = sch.get_loops(block)[-3:]

    a_yo, a_yi = sch.split(a_y, factors=[None, min(M, 32)])

    a_xo, a_xi = sch.split(a_x, factors=[None, 16])
    sch.reorder(a_yo, a_xo, a_yi, a_xi)

    a_xi, a_k = sch.get_loops(block)[-2:]
    a_ko, a_ki = sch.split(a_k, factors=[None, 4])
    sch.reorder(a_ko, a_xi, a_ki)

    if batched:
        a_b = sch.get_loops(block)[0]
        fused = sch.fuse(a_b, a_yo, a_xo)
    else:
        fused = sch.fuse(a_yo, a_xo)

    sch.parallel(fused)

    if decompose_reduction:
        dec = sch.decompose_reduction(block, a_ko)
        init_loop = sch.get_loops(dec)[-1]
        sch.vectorize(init_loop)


def schedule_dense(dense_block, M, sch: tir.Schedule, decompose_reduction=True):
    schedule_matmul_common(sch, dense_block, False, M, decompose_reduction=decompose_reduction)


def schedule_batch_matmul(bmm_block, M, sch, decompose_reduction=True):
    schedule_matmul_common(sch, bmm_block, True, M, decompose_reduction=decompose_reduction)


def vnni_relay(decompose_reduction=True):
    M, N, K = 1024, 1024, 1024
    data_shape = (M, K)
    weight_shape = (N, K)

    data_dtype = "uint8"
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    dense = relay.nn.dense(data, weight, out_dtype="int32")
    out = relay.nn.batch_matmul(
        relay.cast(relay.expand_dims(dense, 0), "uint8"),
        relay.cast(relay.expand_dims(dense, 0), "int8"),
        out_dtype="int32",
    )
    relay_mod = tvm.IRModule.from_expr(out)

    target = "llvm -mcpu=cascadelake"

    data = np.random.uniform(1, 10, size=(M, K)).astype("uint8")
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")

    params = {"weight": weight_np}

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    with tempfile.TemporaryDirectory() as work_dir:
        database = JSONDatabase(
            path_workload=os.path.join(work_dir, "database_workload.json"),
            path_tuning_record=os.path.join(work_dir, "database_tuning_record.json"),
        )

        for task in filter(
            lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
            extracted_tasks,
        ):
            # print("task name", task.task_name)
            # print(task.mod)
            mod = Parse._mod(task.dispatched[0])
            workload = database.commit_workload(mod)

            sch = tvm.tir.Schedule(mod)
            # print(sch.mod.script())

            block = sch.get_block("compute")
            schedule_rule = sch.get(block).annotations["schedule_rule"]

            if "dense_vnni" in schedule_rule:
                schedule_dense(block, M, sch, decompose_reduction)

            if "batch_matmul_vnni" in schedule_rule:
                schedule_batch_matmul(block, M, sch, decompose_reduction)

            # print(sch.mod.script())

            tune_rec = TuningRecord(
                sch.trace, [0.0], workload, tvm.target.Target(target), []
            )

            database.commit_tuning_record(tune_rec)

        with ApplyHistoryBest(database):
            with tvm.transform.PassContext(
                opt_level=3,
                config={"relay.backend.use_meta_schedule": True},
            ):
                relay.build(relay_mod, target=target, params=params)


if __name__ == "__main__":
    print("\nWith decompose_reduction=True")
    vnni_relay(decompose_reduction=True)

    # print("\nWith decompose_reduction=False")
    # vnni_relay(decompose_reduction=False)

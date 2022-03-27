import tvm
from tvm import te, tir, relay
from tvm._ffi import register_func
import tvm.testing
import numpy as np
from tvm.meta_schedule.tune import extract_task_from_relay, Parse, tune_extracted_tasks
from tvm.meta_schedule.integration import ApplyHistoryBest
from tvm import meta_schedule as ms
import tempfile
from tvm.topi.transform import layout_transform
import tvm.topi.testing
from tvm.meta_schedule.database import TuningRecord, JSONDatabase
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base

import vnni_common


def matmul(n: int, m: int, k: int):
    X = te.placeholder((m, k), name="X", dtype="uint8")
    packedW = te.placeholder((n // 16, k // 4, 16, 4), name="packedW", dtype="int8")

    ak = te.reduce_axis((0, k), name="k")
    out = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packedW[
                tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4), j % 16, ak % 4
            ].astype("int32"),
            axis=ak,
        ),
        name="compute",
    )
    return [X, packedW, out]


def batch_matmul(batch, n: int, m: int, k: int):
    x = te.placeholder((batch, m, k), name="X", dtype="uint8")
    y = te.placeholder((batch, n, k), name="Y", dtype="int8")
    packed_y_layout = "BNK16n4k"
    packed_y = layout_transform(y, "BNK", packed_y_layout)

    _, n_o, _, n_i, _ = packed_y.shape
    ak = te.reduce_axis((0, k), name="k")

    z = te.compute(
        (batch, m, n_o * n_i),
        lambda b, i, j: te.sum(
            x[b, i, ak].astype("int32")
            * packed_y[
                b, tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4), j % 16, ak % 4
            ].astype("int32"),
            axis=ak,
        ),
        name="compute",
        tag="batch_matmul_vnni",
    )

    return [x, y, z]


def schedule_matmul_common(sch, block, do_tune, batched, M):
    post_blocks = sch.get_consumers(block)

    if len(post_blocks) > 0:
        while True:
            next_post_blocks = []
            for post_block in post_blocks:
                next_consumers = sch.get_consumers(post_block)

                if len(next_consumers) > 0:
                    sch.compute_inline(post_block)

                next_post_blocks += next_consumers

            if len(next_post_blocks) == 0:
                assert len(post_blocks) == 1
                outer_block = post_blocks[0]
                a_y, a_x = sch.get_loops(outer_block)[-2:]
                break

            post_blocks = next_post_blocks
    else:
        a_y, a_x, _ = sch.get_loops(block)[-3:]
        outer_block = block

    if do_tune:
        y_factors = sch.sample_perfect_tile(a_y, n=2, max_innermost_factor=128)
        a_yo, a_yi = sch.split(a_y, factors=y_factors)
    else:
        a_yo, a_yi = sch.split(a_y, factors=[None, min(M, 32)])

    a_xo, a_xi = sch.split(a_x, factors=[None, 16])
    sch.reorder(a_yo, a_xo, a_yi, a_xi)


    if outer_block != block:
        sch.vectorize(a_xi)
        sch.compute_at(block, a_yi)

    a_xi, a_k = sch.get_loops(block)[-2:]
    a_ko, a_ki = sch.split(a_k, factors=[None, 4])
    sch.reorder(a_ko, a_xi, a_ki)

    if batched:
        a_b = sch.get_loops(outer_block)[0]
        fused = sch.fuse(a_b, a_yo, a_xo)
    else:
        fused = sch.fuse(a_yo, a_xo)

    sch.parallel(fused)

    dec = sch.decompose_reduction(block, a_ko)

    init_loop = sch.get_loops(dec)[-1]
    sch.vectorize(init_loop)

    sch.tensorize(a_xi, "dot_16x1x16_uint8_int8_int32_cascadelake")

    return fused


def schedule_dense(dense_block, M, do_tune, sch: tir.Schedule):
    schedule_matmul_common(sch, dense_block, do_tune, False, M)


def schedule_dense_for_tune(sch: tir.Schedule):
    block = sch.get_block("compute")
    return schedule_dense(block, None, True, sch)


def schedule_rule_dense_vnni(sch: tir.Schedule, block):
    schedule_dense(block, None, True, sch)
    return [sch]


def schedule_batch_matmul(bmm_block, M, do_tune, sch, layout_trans_compute_root=False):
    outer_loop = schedule_matmul_common(sch, bmm_block, do_tune, True, M)

    layout_trans_block = sch.get_block("T_layout_trans")

    if layout_trans_compute_root:
        i0, i1, i2, i3, i4 = sch.get_loops(layout_trans_block)
        sch.parallel(sch.fuse(i0, i1, i2))
        sch.vectorize(i4)
    else:
        sch.compute_at(layout_trans_block, outer_loop)
        # fused, ax0, ax1, ax2
        _, _, ax1, ax2 = sch.get_loops(layout_trans_block)
        sch.unroll(ax1)
        sch.vectorize(ax2)


def schedule_batch_matmul_for_tune(sch: tir.Schedule):
    bmm_block = sch.get_block("compute")
    return schedule_batch_matmul(bmm_block, None, True, sch)


def schedule_rule_batch_matmul_vnni(sch: tir.Schedule, bmm_block):
    sch_copy = sch.copy()
    schedule_batch_matmul(bmm_block, None, True, sch, layout_trans_compute_root=False)
    schedule_batch_matmul(bmm_block, None, True, sch_copy, layout_trans_compute_root=True)
    return [sch, sch_copy]


register_func("meta_schedule.dense_vnni", schedule_rule_dense_vnni)
register_func("meta_schedule.batch_matmul_vnni", schedule_rule_batch_matmul_vnni)


fbgemm_workloads = [
    (64, 800, 320),
    (64, 768, 512),
    (16, 256, 512),
    (128, 128, 128),
    (256, 512, 256),
    (1024, 1024, 1024),
]

bert_workloads = [(128, 768, 3072), (128, 768, 768), (128, 3072, 768)]


def verify_dense(sch, target, M, N, K):
    f = tvm.build(sch.mod["main"], target=target, name="dense")
    dev = tvm.device(target, 0)
    a_np = np.random.uniform(1, 10, size=(M, K)).astype("uint8")
    b_np = np.random.uniform(1, 10, size=(N, K)).astype("int8")
    c_np = np.dot(a_np.astype("int32"), b_np.transpose().astype("int32"))

    packW = np.random.uniform(1, 10, size=(N // 16, (K // 4), 16, 4)).astype("int8")

    for r_idx in range(N // 16):
        for ko in range(K // 4):
            for s_idx in range(16):
                for t_idx in range(4):
                    packW[r_idx][ko][s_idx][t_idx] = b_np[r_idx * 16 + s_idx][
                        ko * 4 + t_idx
                    ]

    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(packW, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype="int32"), dev)

    # print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=10)
    gflops = (N * M * K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print(
        "matmul with VNNI TIR tensorization: %f ms, %f GFLOPS"
        % (time_ms, gflops / (time_ms / 1e3))
    )


def test_vnni_dense():
    do_tune = False
    target = "llvm -mcpu=cascadelake --num-cores=4"

    for M, N, K in fbgemm_workloads + bert_workloads:
        workload = te.create_prim_func(matmul(n=N, m=M, k=K))

        if not do_tune:
            ir_module = tvm.IRModule({"main": workload})
            sch = tvm.tir.Schedule(ir_module)
            block = sch.get_block("compute")
            schedule_dense(block, M, do_tune, sch)
        else:
            with tempfile.TemporaryDirectory() as work_dir:
                sch = ms.tune_tir(
                    mod=workload,
                    target=target,
                    # use replay or evolutionary search
                    config=ms.ReplayTraceConfig(
                        num_trials_per_iter=32,
                        num_trials_total=32,
                    ),
                    work_dir=work_dir,
                    space=ms.space_generator.ScheduleFn(schedule_dense_for_tune),
                )
                if sch is None:
                    print("No valid schedule found!")
                else:
                    print(sch.mod.script())
                    print(sch.trace)

        # print(sch.mod.script())

        verify_dense(sch, target, M, N, K)
        break


def test_vnni_batch_matmul():
    do_tune = False

    workloads = []
    for m, n, k in fbgemm_workloads + bert_workloads:
        batch = 8
        workloads.append((batch, m, n, k))

    seq_len = 384
    bert_bmm_workloads = [
        (16, 32, seq_len, 96),
        (16, seq_len, seq_len, 64),
        (16, seq_len, 64, seq_len),
    ]

    target = "llvm -mcpu=cascadelake --num-cores=4"

    for batch, M, N, K in workloads + bert_bmm_workloads:
        workload = te.create_prim_func(batch_matmul(batch, n=N, m=M, k=K))

        if not do_tune:
            ir_module = tvm.IRModule({"main": workload})
            sch = tvm.tir.Schedule(ir_module)
            block = sch.get_block("compute")
            schedule_batch_matmul(block, M, False, sch, False)
        else:
            with tempfile.TemporaryDirectory() as work_dir:
                sch = ms.tune_tir(
                    mod=workload,
                    target=target,
                    # use replay or evolutionary search
                    config=ms.ReplayTraceConfig(
                        num_trials_per_iter=32,
                        num_trials_total=32,
                    ),
                    work_dir=work_dir,
                    space=ms.space_generator.ScheduleFn(schedule_batch_matmul_for_tune),
                )
                if sch is None:
                    print("No valid schedule found!")
                else:
                    print(sch.mod.script())
                    print(sch.trace)

        # print(sch.mod.script())

        f = tvm.build(sch.mod["main"], target=target, name="dense")
        dev = tvm.device(target, 0)
        a_np = np.random.uniform(1, 10, size=(batch, M, K)).astype("uint8")
        b_np = np.random.uniform(1, 10, size=(batch, N, K)).astype("int8")

        c_np = tvm.topi.testing.batch_matmul(a_np, b_np, out_dtype="int32")

        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros((batch, M, N), dtype="int32"), dev)

        # print(f.imported_modules[0].get_source())
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

        evaluator = f.time_evaluator(f.entry_name, dev, number=10)
        gflops = (N * M * K) * 2 * batch / 1e9
        time_ms = evaluator(a, b, c).mean * 1e3
        print(
            "matmul with VNNI TIR tensorization: %f ms, %f GFLOPS"
            % (time_ms, gflops / (time_ms / 1e3))
        )
        break


def vnni_relay():
    M = 1024
    N = 1024
    K = 1024
    data_shape = (M, K)
    weight_shape = (N, K)

    data_dtype = "uint8"
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    dense = relay.nn.dense(data, weight, out_dtype="int32")
    bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
    bias_add = relay.nn.bias_add(dense, bias) + relay.const(1, dtype="int32")
    out = dense
    bmm = relay.nn.batch_matmul(
        relay.cast(relay.expand_dims(bias_add, 0), "uint8"),
        relay.cast(relay.expand_dims(bias_add, 0), "int8"),
        out_dtype="int32",
    )
    out = bmm + relay.const(1, dtype="int32")
    # out = bias_add + relay.const(1, dtype="int32")
    relay_mod = tvm.IRModule.from_expr(out)

    print(relay.transform.InferType()(relay_mod))

    target = "llvm -mcpu=cascadelake"
    dev = tvm.device(target, 0)

    data = np.random.uniform(1, 10, size=(M, K)).astype("uint8")
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    ref = (
        relay.create_executor("vm", mod=relay_mod, device=dev, target=target)
        .evaluate()(*[data, weight_np, bias_np])
        .numpy()
    )

    params = {"weight": weight_np, "bias": bias_np}

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    database = JSONDatabase(
        path_workload="database_workload.json",
        path_tuning_record="database_tuning_record.json",
    )

    for task in filter(
        lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
        extracted_tasks,
    ):
        print("task name", task.task_name)
        print(task.mod)
        mod = Parse._mod(task.dispatched[0])
        workload = database.commit_workload(mod)

        sch = tvm.tir.Schedule(mod)
        print(sch.mod.script())

        block = sch.get_block("compute")
        schedule_rule = sch.get(block).annotations["schedule_rule"]

        if "dense_vnni" in schedule_rule:
            schedule_dense(block, M, False, sch)

        if "batch_matmul_vnni" in schedule_rule:
            schedule_batch_matmul(block, M, False, sch)

        print(sch.mod.script())

        tune_rec = TuningRecord(
            sch.trace, [0.0], workload, tvm.target.Target(target), []
        )

        database.commit_tuning_record(tune_rec)

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

        asm = lib.lib.get_source("asm")
        assert "vpdpbusd" in asm

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data)
    runtime.run()

    out = runtime.get_output(0).numpy()

    np.testing.assert_equal(out, ref)


def test_bert():
    relay_mod, params, input_info = load_quantized_bert_base()

    target = "llvm -mcpu=cascadelake"

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    database = JSONDatabase(
        path_workload="database_workload_bert.json",
        path_tuning_record="database_tuning_record_bert.json",
    )

    for task in filter(
        lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
        extracted_tasks,
    ):
        mod = Parse._mod(task.dispatched[0])

        relay_func = list(task.mod.functions.values())[0]
        out_type = relay_func.body.checked_type

        if database.has_workload(mod) or out_type.dtype == "float32":
            continue

        print(task.task_name)

        sch = tvm.tir.Schedule(mod)
        block = sch.get_block("compute")
        schedule_rule = sch.get(block).annotations["schedule_rule"]

        if "dense_vnni" in schedule_rule:
            M, _ = out_type.shape
            schedule_dense(block, M, False, sch)

        if "batch_matmul_vnni" in schedule_rule:
            _, M, _ = out_type.shape
            schedule_batch_matmul(block, M, False, sch)

        print(sch.mod.script())

        workload = database.commit_workload(mod)

        tune_rec = TuningRecord(
            sch.trace, [0.0], workload, tvm.target.Target(target), []
        )

        database.commit_tuning_record(tune_rec)

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            print("building")
            lib = relay.build(relay_mod, target=target, params=params)
            print("building done")

        # asm = lib.lib.get_source("asm")
        # assert "vpdpbusd" in asm

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    inputs = []

    for name, shape in input_info:
        arr = np.random.uniform(1, 10, size=shape).astype("int64")
        runtime.set_input(name, arr)
        inputs.append(arr)

    print("running")
    runtime.run()
    print("done")

    # ref_outs = relay.create_executor("graph", mod=relay_mod, device=dev, target=target).evaluate()(*inputs)

    # for i in range(2):
    #     out = runtime.get_output(i).numpy()
    #     ref = ref_outs[i].numpy()
    #     np.testing.assert_allclose(out, ref, atol=1e-3, rtol=1e-3)

    print(runtime.benchmark(dev, number=1, repeat=50).mean)


def vnni_relay_tune():
    M = 1024
    N = 1024
    K = 1024
    data_shape = (M, K)
    weight_shape = (N, K)

    data_dtype = "uint8"
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    dense = relay.nn.dense(data, weight, out_dtype="int32")
    bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
    bias_add = relay.nn.bias_add(dense, bias) + relay.const(1, dtype="int32")
    bmm = relay.nn.batch_matmul(
        relay.cast(relay.expand_dims(bias_add, 0), "uint8"),
        relay.cast(relay.expand_dims(bias_add, 0), "int8"),
        out_dtype="int32",
    )
    out = bmm + relay.const(1, dtype="int32")

    relay_mod = tvm.IRModule.from_expr(out)

    target = "llvm -mcpu=cascadelake -num-cores 4"
    dev = tvm.device(target, 0)

    data = np.random.uniform(1, 10, size=(M, K)).astype("uint8")
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    ref = (
        relay.create_executor("vm", mod=relay_mod, device=dev, target=target)
        .evaluate()(*[data, weight_np, bias_np])
        .numpy()
    )

    params = {"weight": weight_np, "bias": bias_np}

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    tune_tasks = list(
        filter(
            lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
            extracted_tasks,
        )
    )

    with tempfile.TemporaryDirectory() as work_dir:
        config = ms.ReplayTraceConfig(
            num_trials_per_iter=64,
            num_trials_total=64,
        )
        database = tune_extracted_tasks(tune_tasks, target, config, work_dir=work_dir)

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

        asm = lib.lib.get_source("asm")
        assert "vpdpbusd" in asm

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data)
    runtime.run()

    out = runtime.get_output(0).numpy()

    np.testing.assert_equal(out, ref)


def test_bert_tune():
    relay_mod, params, input_info = load_quantized_bert_base()

    target = "llvm -mcpu=cascadelake -num-cores 4"

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    tune_tasks = []

    for task in filter(
        lambda task: "dense" in task.task_name or "batch_matmul" in task.task_name,
        extracted_tasks,
    ):
        relay_func = list(task.mod.functions.values())[0]
        out_type = relay_func.body.checked_type

        if out_type.dtype != "float32":
            tune_tasks.append(task)

    with tempfile.TemporaryDirectory() as work_dir:
        config = ms.ReplayTraceConfig(
            num_trials_per_iter=64,
            num_trials_total=64,
        )
        database = tune_extracted_tasks(tune_tasks, target, config, work_dir=work_dir)

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    inputs = []

    for name, shape in input_info.items():
        arr = np.random.uniform(1, 10, size=shape).astype("int64")
        runtime.set_input(name, arr)
        inputs.append(arr)

    print(runtime.benchmark(dev, number=1, repeat=50).mean)


if __name__ == "__main__":
    # test_vnni_batch_matmul()
    # test_vnni_dense()
    # vnni_relay()
    # test_bert()
    vnni_relay_tune()
    # test_bert_tune()

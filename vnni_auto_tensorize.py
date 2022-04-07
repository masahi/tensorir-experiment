import tvm
from tvm import relay
import tvm.testing
import numpy as np
from tvm.meta_schedule.tune import extract_task_from_relay, tune_extracted_tasks
from tvm.meta_schedule.integration import ApplyHistoryBest
from tvm.meta_schedule import schedule_rule
from tvm import meta_schedule as ms
from tvm.tir.tensor_intrin import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN
import tempfile
import tvm.topi.testing


def vnni_relay_tune():
    M, N, K = 1024, 1024, 1024
    data_shape = (M, K)
    weight_shape = (N, K)

    data_dtype = "uint8"
    weight_dtype = "int8"
    out_dtype = "int32"
    # data_dtype = "float32"
    # weight_dtype = "float32"
    # out_dtype = "float32"

    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    dense = relay.nn.dense(data, weight, out_dtype=out_dtype)
    bias = relay.var("bias", shape=(weight_shape[0],), dtype=out_dtype)
    bias_add = relay.nn.bias_add(dense, bias) + relay.const(1, dtype=out_dtype)
    # bmm = relay.nn.batch_matmul(
    #     relay.cast(relay.expand_dims(bias_add, 0), "uint8"),
    #     relay.cast(relay.expand_dims(bias_add, 0), "int8"),
    #     out_dtype="int32",
    # )
    # out = bmm + relay.const(1, dtype="int32")
    out = dense

    relay_mod = tvm.IRModule.from_expr(out)

    target = "llvm -mcpu=cascadelake -num-cores 4"
    dev = tvm.device(target, 0)

    data = np.random.uniform(1, 10, size=(M, K)).astype(data_dtype)
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype(out_dtype)

    ref = (
        relay.create_executor("vm", mod=relay_mod, device=dev, target=target)
        # .evaluate()(*[data, weight_np, bias_np])
        .evaluate()(*[data, weight_np]).numpy()
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
            # num_trials_per_iter=320,
            # max_trials_per_task=320,
            # max_trials_global=20000,
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=32,
        )
        # config = ms.EvolutionarySearchConfig(
        #     num_trials_per_iter=64,
        #     max_trials_per_task=64,
        #     max_trials_global=20000,
        #     population_size=2048,
        #     init_measured_ratio=0.2,
        #     init_min_unmeasured=50,
        #     genetic_num_iters=3,
        #     genetic_mutate_prob=0.85,
        #     genetic_max_fail_count=10,
        #     eps_greedy=0.05,
        # )

        sch_rules = [
            schedule_rule.AutoInline(
                into_producer=False,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=True,
                require_injective=True,
                require_ordered=True,
                disallow_op=["tir.exp"],
            ),
            schedule_rule.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
            schedule_rule.MultiLevelTilingWithIntrin(
                VNNI_INTRIN,
                structure="SSRSRS",
                tile_binds=None,
                max_innermost_factor=64,
                vector_load_lens=None,
                reuse_read=None,
                reuse_write=schedule_rule.ReuseType(
                    req="may",
                    levels=[1, 2],
                    scope="global",
                ),
            ),
            schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=16,
                max_vectorize_extent=64,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
            schedule_rule.RandomComputeLocation(),
        ]

        database = tune_extracted_tasks(
            tune_tasks, target, config, sch_rules=lambda : sch_rules, work_dir=work_dir
        )

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


vnni_relay_tune()

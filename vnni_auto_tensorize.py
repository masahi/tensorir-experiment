import tvm
from tvm import relay
import tvm.testing
import numpy as np
from tvm.meta_schedule.tune import extract_task_from_relay, tune_extracted_tasks
from tvm.meta_schedule.integration import ApplyHistoryBest
from tvm import meta_schedule as ms
import tempfile
import tvm.topi.testing


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
    out = dense

    relay_mod = tvm.IRModule.from_expr(out)

    target = "llvm -mcpu=cascadelake -num-cores 4"
    dev = tvm.device(target, 0)

    data = np.random.uniform(1, 10, size=(M, K)).astype("uint8")
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    ref = (
        relay.create_executor("vm", mod=relay_mod, device=dev, target=target)
        # .evaluate()(*[data, weight_np, bias_np])
        .evaluate()(*[data, weight_np])
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
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=64,
        )
        database = tune_extracted_tasks(tune_tasks, target, config, work_dir=work_dir, postprocs=lambda: [])

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

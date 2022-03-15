import tvm
from tvm import te, tir, relay
from tvm._ffi import register_func
import tvm.testing
import numpy as np
from tvm.meta_schedule.tune import extract_task_from_relay, Parse, tune_extracted_tasks
from tvm.meta_schedule.integration import ApplyHistoryBest
from tvm import meta_schedule as ms
import tempfile
import tvm.topi.testing
from tvm.meta_schedule.database import TuningRecord, JSONDatabase

import vnni_common


def get_conv2d_nchw(
    d_shape,
    w_shape,
    padding,
    strides=(1, 1),
):
    data_dtype = "uint8"
    weight_dtype = "int8"
    out_dtype = "int32"

    data = relay.var("data", shape=d_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=w_shape, dtype=weight_dtype)
    out_channel = w_shape[0]
    return relay.nn.conv2d(
        data=data,
        weight=weight,
        kernel_size=w_shape[2:],
        channels=out_channel,
        padding=padding,
        strides=strides,
        out_dtype=out_dtype,
    )


def vnni_relay():
    data_shape = (1, 32, 64, 64)
    weight_shape = (32, 32, 3, 3)

    padding = (1, 1)

    conv2d = get_conv2d_nchw(data_shape, weight_shape, padding)

    out = conv2d

    relay_mod = tvm.IRModule.from_expr(out)

    print(relay.transform.InferType()(relay_mod))

    target = "llvm -mcpu=cascadelake"
    dev = tvm.device(target, 0)

    data = np.random.uniform(1, 10, data_shape).astype("uint8")
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    # bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    ref_exec = relay.create_executor("vm", mod=relay_mod, device=dev, target=target)
    ref = ref_exec.evaluate()(*[data, weight_np]).numpy()

    params = {"weight": weight_np}

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    tune_tasks = list(
        filter(
            lambda task: "conv2d" in task.task_name,
            extracted_tasks,
        )
    )

    for task in tune_tasks:
        mod = Parse._mod(task.dispatched[0])
        print(mod)

    with tvm.transform.PassContext(
        opt_level=3,
        config={"relay.backend.use_meta_schedule": True},
    ):
        # opt_mod, _ = relay.optimize(relay_mod, target=target, params=params)
        # print(opt_mod)
        lib = relay.build(relay_mod, target=target, params=params)

    asm = lib.lib.get_source("asm")
    assert "vpdpbusd" in asm

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data)
    runtime.run()

    out = runtime.get_output(0).numpy()

    np.testing.assert_equal(out, ref)


if __name__ == "__main__":
    vnni_relay()

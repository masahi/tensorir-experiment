import os
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


def schedule_conv2d(sch, block):
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
                # a_y, a_x = sch.get_loops(outer_block)[-2:]
                break

            post_blocks = next_post_blocks
    else:
        outer_block = block

    (
        batch,
        oc_chunk,
        oh,
        _,
        oc_block,
    ) = sch.get_loops(outer_block)[:5]

    parallel_axis = sch.fuse(batch, oc_chunk, oh)
    sch.parallel(parallel_axis)

    if outer_block != block:
        sch.vectorize(oc_block)
        sch.compute_at(block, parallel_axis)

    (
        ow,
        oc_block,
        kh,
        kw,
        ic_outer,
        ic_f_inner,
        ic_s_inner,
    ) = sch.get_loops(block)[-7:]

    parallel_axis = sch.get_loops(block)[0]
    print(sch.get(parallel_axis))

    vector_width = 16

    ow_chunk, ow_block = sch.split(ow, factors=[None, 16])
    oc_f_inner, oc_s_inner = sch.split(oc_block, factors=[None, vector_width])

    CC = sch.cache_write(block, 0, "global")

    if outer_block == block:
        sch.reverse_compute_at(CC, parallel_axis)

    oc_block_cache_write = sch.get_loops(CC)[-1]
    sch.vectorize(oc_block_cache_write)

    sch.reorder(
        ow_chunk,
        ic_outer,
        kh,
        kw,
        ic_f_inner,
        ow_block,
        oc_f_inner,
        oc_s_inner,
        ic_s_inner,
    )
    sch.unroll(ow_block)
    sch.unroll(oc_f_inner)

    dec = sch.decompose_reduction(block, ic_outer)
    init_loop = sch.get_loops(dec)[-1]
    sch.vectorize(init_loop)

    sch.tensorize(oc_s_inner, "dot_16x1x16_uint8_int8_int32_cascadelake")

    print(sch.mod.script())


def vnni_relay():
    # os.remove("database_tuning_record_conv2d.json")
    # os.remove("database_workload_conv2d.json")

    data_shape = (1, 32, 128, 128)
    weight_shape = (32, 32, 3, 3)
    bias_shape = (weight_shape[0],)
    padding = (1, 1)

    bias = relay.var("bias", shape=bias_shape, dtype="int32")

    conv2d = get_conv2d_nchw(data_shape, weight_shape, padding)
    bias_add = relay.nn.bias_add(conv2d, bias)

    out = bias_add + relay.const(1, dtype="int32")
    # out = conv2d

    relay_mod = tvm.IRModule.from_expr(out)

    print(relay.transform.InferType()(relay_mod))

    target = "llvm -mcpu=cascadelake"
    dev = tvm.device(target, 0)

    data = np.random.uniform(1, 10, data_shape).astype("uint8")
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    bias_np = np.random.uniform(1, 10, size=bias_shape).astype("int32")

    ref_exec = relay.create_executor("vm", mod=relay_mod, device=dev, target=target)
    ref = ref_exec.evaluate()(*[data, weight_np, bias_np]).numpy()
    # ref = ref_exec.evaluate()(*[data, weight_np]).numpy()

    params = {"weight": weight_np, "bias": bias_np}

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    tune_tasks = list(
        filter(
            lambda task: "conv2d" in task.task_name,
            extracted_tasks,
        )
    )

    database = JSONDatabase(
        path_workload="database_workload_conv2d.json",
        path_tuning_record="database_tuning_record_conv2d.json",
    )

    for task in tune_tasks:
        mod = Parse._mod(task.dispatched[0])
        workload = database.commit_workload(mod)

        sch = tvm.tir.Schedule(mod)
        block = sch.get_block("conv2d_NCHWc_int8")

        schedule_rule = sch.get(block).annotations["schedule_rule"]

        if "conv2d_NCHWc_int8" in schedule_rule:
            schedule_conv2d(sch, block)

        tune_rec = TuningRecord(
            sch.trace, [0.0], workload, tvm.target.Target(target), []
        )

        database.commit_tuning_record(tune_rec)

    with ApplyHistoryBest(database):
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

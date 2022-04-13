import re
import time

import tvm
from tvm import relay
import tvm.testing
import numpy as np
from tvm.meta_schedule.tune import tune_extracted_tasks
from tvm.meta_schedule.relay_integration import extract_task_from_relay
from tvm.meta_schedule import ApplyHistoryBest

from tvm.meta_schedule import schedule_rule, postproc
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base
from tvm import meta_schedule as ms
from tvm.tir.tensor_intrin import DP4A_INTRIN
import tempfile
import tvm.topi.testing

config = ms.ReplayTraceConfig(
    num_trials_per_iter=32,
    max_trials_per_task=32,
    max_trials_global=20000,
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
    schedule_rule.MultiLevelTilingWithIntrin(
        DP4A_INTRIN,
        structure="SSSRRSRS",
        tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
        max_innermost_factor=64,
        vector_load_lens=[1, 2, 3, 4],
        reuse_read=schedule_rule.ReuseType(
            req="must",
            levels=[4],
            scope="shared",
        ),
        reuse_write=schedule_rule.ReuseType(
            req="must",
            levels=[3],
            scope="local",
        ),
    ),
    schedule_rule.AutoInline(
        into_producer=True,
        into_consumer=True,
        inline_const_tensor=True,
        disallow_if_then_else=False,
        require_injective=False,
        require_ordered=False,
        disallow_op=None,
    ),
    schedule_rule.CrossThreadReduction(
        thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]
    ),
    schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=-1,  # disable parallelize
        max_vectorize_extent=-1,  # disable vectorize
        unroll_max_steps=[0, 16, 64, 512, 1024],
        unroll_explicit=True,
    ),
]

postprocs = [
    postproc.DisallowDynamicLoop(),
    postproc.RewriteCooperativeFetch(),
    postproc.RewriteUnboundBlock(),
    postproc.RewriteParallelVectorizeUnroll(),
    postproc.RewriteReductionBlock(),
    postproc.RewriteTensorize(),
    postproc.VerifyGPUCode(),
]


target = "vulkan -from_device=0"


def vnni_relay_tune():
    M, N, K = 1024, 1024, 1024
    data_shape = (M, K)
    weight_shape = (N, K)

    data_dtype = "int8"
    weight_dtype = "int8"
    out_dtype = "int32"

    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    dense = relay.nn.dense(data, weight, out_dtype=out_dtype)
    # bias = relay.var("bias", shape=(weight_shape[0],), dtype=out_dtype)
    # bias_add = relay.nn.bias_add(dense, bias) + relay.const(1, dtype=out_dtype)
    bmm = relay.nn.batch_matmul(
        relay.cast(relay.expand_dims(dense, 0), "uint8"),
        relay.cast(relay.expand_dims(dense, 0), "int8"),
        out_dtype="int32",
    )
    out = dense

    relay_mod = tvm.IRModule.from_expr(out)

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
            lambda task: "dense" in task.task_name,
            extracted_tasks,
        )
    )

    with tempfile.TemporaryDirectory() as work_dir:
        database = tune_extracted_tasks(
            tune_tasks,
            target,
            config,
            sch_rules=lambda: sch_rules,
            postprocs=lambda: postprocs,
            work_dir=work_dir,
        )

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input("data", data)
    runtime.run()

    out = runtime.get_output(0).numpy()

    np.testing.assert_equal(out, ref)

    loaded_lib = tvm.runtime.load_module(path_lib)
    runtime = tvm.contrib.graph_executor.GraphModule(loaded_lib["default"](dev))
    runtime.set_input("data", data)
    runtime.run()

    out = runtime.get_output(0).numpy()

    np.testing.assert_equal(out, ref)


def test_bert():
    relay_mod, params, input_info = load_quantized_bert_base()

    relay_mod = relay.transform.FastMath()(relay_mod)
    print("loaded bert")

    target = "llvm -mcpu=cascadelake -num-cores 4"

    t1 = time.time()
    extracted_tasks = extract_task_from_relay(relay_mod, target, params)
    t2 = time.time()

    print("task extracted in ", t2 - t1)

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
        database = tune_extracted_tasks(
            tune_tasks,
            target,
            config,
            sch_rules=lambda: sch_rules,
            postprocs=lambda: postprocs,
            work_dir=work_dir,
        )

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    inputs = []

    for name, shape in input_info:
        arr = np.random.uniform(1, 10, size=shape).astype("int64")
        runtime.set_input(name, arr)
        inputs.append(arr)

    print(runtime.benchmark(dev, number=1, repeat=50).mean)

    path_lib = "deploy_lib_qbert.tar"
    lib.export_library(path_lib)


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


def get_workloads():
    workloads = []
    pat = "--n=(\d+) --h=(\d+) --w=(\d+) --c=(\d+) --k=(\d+) --r=(\d+) --s=(\d+) --pad_h=(\d+) --pad_w=(\d+) --stride_h=(\d+) --stride_w=(\d+)"
    with open("resnet50.txt") as f:
        for line in f.readlines():
            m = re.match(pat, line)
            groups = [int(v) for v in m.groups()]
            workloads.append(
                {
                    "n": groups[0],
                    "h": groups[1],
                    "w": groups[2],
                    "c": groups[3],
                    "k": groups[4],
                    "r": groups[5],
                    "s": groups[6],
                    "pad_h": groups[7],
                    "pad_w": groups[8],
                    "stride_h": groups[9],
                    "stride_w": groups[10],
                }
            )

    return workloads


def vnni_conv2d():

    for workload in get_workloads():
        print(workload, end=",")
        data_shape = (workload["n"], workload["c"], workload["h"], workload["w"])
        weight_shape = (workload["k"], workload["c"], workload["r"], workload["s"])

        bias_shape = (weight_shape[0],)
        padding = (workload["pad_h"], workload["pad_w"])
        stride = (workload["stride_h"], workload["stride_w"])

        bias = relay.var("bias", shape=bias_shape, dtype="int32")

        conv2d = get_conv2d_nchw(data_shape, weight_shape, padding, stride)
        bias_add = relay.nn.bias_add(conv2d, bias)
        out = bias_add

        relay_mod = tvm.IRModule.from_expr(out)

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

        with tempfile.TemporaryDirectory() as work_dir:
            database = tune_extracted_tasks(
                tune_tasks,
                target,
                config,
                sch_rules=lambda: sch_rules,
                postprocs=lambda: postprocs,
                work_dir=work_dir,
            )

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

        break


def run_tvm_model(mod, params, input_name, inp, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(
        lib["default"](tvm.device(target, 0))
    )

    runtime.set_input(input_name, inp)
    runtime.run()
    return runtime.get_output(0).numpy(), runtime


def test_qresnet():
    input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
    inp = np.random.randn(1, 3, 224, 224).astype("float32")

    with open("models/qresnet50.json", "r") as fi:
        relay_mod = tvm.ir.load_json(fi.read())

    with open("models/qresnet50.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())

    extracted_tasks = extract_task_from_relay(relay_mod, target, params)

    tune_tasks = list(
        filter(
            lambda task: "conv2d" in task.task_name,
            extracted_tasks,
        )
    )

    with tempfile.TemporaryDirectory() as work_dir:
        database = tune_extracted_tasks(
            tune_tasks,
            target,
            config,
            sch_rules=lambda: sch_rules,
            postprocs=lambda: postprocs,
            work_dir=work_dir,
        )

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            lib = relay.build(relay_mod, target=target, params=params)

    dev = tvm.cpu(0)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input(input_name, inp)
    runtime.run()

    n_repeat = 100

    print(runtime.benchmark(dev, number=1, repeat=n_repeat))


vnni_relay_tune()
# vnni_conv2d()
# test_bert()
# test_qresnet()

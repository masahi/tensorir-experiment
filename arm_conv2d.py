# import torch
# from torch import nn
# from torchvision.models.quantization import resnet18 as qresnet18
# from torchvision.models.quantization import resnet50 as qresnet50
# from torch.quantization import fuse_modules, QuantWrapper

import os
import tvm
from tvm import te, tir, relay, autotvm
from tvm._ffi import register_func
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing
import numpy as np
from tvm.meta_schedule.tune import tune_extracted_tasks
from tvm.meta_schedule.relay_integration import extract_task_from_relay
from tvm.meta_schedule import ApplyHistoryBest
from tvm.meta_schedule.testing.utils import apply_fixed_schedules
from tvm import meta_schedule as ms
import tempfile
import tvm.topi.testing
from tvm.meta_schedule.database import TuningRecord, JSONDatabase

from PIL import Image
from tvm.contrib.download import download_testdata

from tvm import rpc
from tvm.contrib import utils, ndk

from tvm.tir.tensor_intrin.arm_cpu import ARM_DOT_4x4_i8_NEON_INTRIN, ARM_DOT_4x4_i8_SDOT_INTRIN


# tracker_host = os.environ["TVM_TRACKER_HOST"]
# tracker_port = int(os.environ["TVM_TRACKER_PORT"])
key = "android"

arch = "aarch64"
# target = "llvm -device arm_cpu -mtriple=%s-linux-android -mattr=+neon" % arch
target = "llvm --device arm_cpu --mtriple aarch64-apple-darwin -mattr=+v8.2a,+dotprod --link-params=1"


def get_conv2d_nchw(
    d_shape,
    w_shape,
    padding,
    strides=(1, 1),
):
    data_dtype = "int8"
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


def schedule_conv2d_nchwc(sch, block):
    # print(sch.get_loops(block))
    producers = sch.get_producers(block)

    if len(producers) > 1:
        assert len(producers) == 1
        pad = producers[0]
        batch, ic_chunk, ih = sch.get_loops(pad)[:3]
        sch.parallel(sch.fuse(batch, ic_chunk, ih))

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

    (batch, oc_chunk, oh, ow, oc_block,) = sch.get_loops(
        outer_block
    )[:5]

    parallel_axis = sch.fuse(batch, oc_chunk, oh)
    sch.parallel(parallel_axis)

    if outer_block != block:
        sch.vectorize(oc_block)
        sch.compute_at(block, ow, preserve_unit_loops=True)

    vector_width = 4

    (oc_block, kh, kw, ic_outer, ic_f_inner, ic_s_inner,) = sch.get_loops(
        block
    )[-6:]

    oc_f_inner, oc_s_inner = sch.split(oc_block, factors=[None, vector_width])

    sch.reorder(
        ic_outer,
        kh,
        kw,
        ic_f_inner,
        oc_f_inner,
        oc_s_inner,
        ic_s_inner,
    )

    sch.unroll(oc_f_inner)

    dec = sch.decompose_reduction(block, ic_outer)
    init_loop = sch.get_loops(dec)[-1]
    sch.vectorize(init_loop)

    # ARM_DOT_4x4_i8_NEON_INTRIN,
    sch.tensorize(oc_s_inner, ARM_DOT_4x4_i8_SDOT_INTRIN)


def get_transform():
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_real_image(im_height, im_width):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    return Image.open(img_path).resize((im_height, im_width))


def get_imagenet_input():
    im = get_real_image(224, 224)
    preprocess = get_transform()
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)


def run_tvm_model(mod, params, input_name, inp, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(
        lib["default"](tvm.device(target, 0))
    )

    runtime.set_input(input_name, inp)
    runtime.run()
    return runtime.get_output(0).numpy(), runtime


def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    torch.quantization.convert(model, inplace=True)


def test_torchvision():
    inp = get_imagenet_input()

    qmodel = qresnet50(pretrained=True).eval()

    pt_inp = torch.from_numpy(inp)
    quantize_model(qmodel, pt_inp)
    script_module = torch.jit.trace(qmodel, pt_inp).eval()

    input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
    input_shapes = [(input_name, (1, 3, 224, 224))]
    relay_mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    target = "llvm -mcpu=cascadelake"

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

        # print(task.mod)
        # print(mod)

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
            lib = relay.build(relay_mod, target=target, params=params)

    dev = tvm.cpu(0)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input(input_name, inp)
    runtime.run()

    n_repeat = 100

    print(runtime.benchmark(dev, number=1, repeat=n_repeat))

    if True:
        tvm_result, rt_mod = run_tvm_model(
            relay_mod, params, input_name, inp, target=target
        )
        print(rt_mod.benchmark(dev, number=1, repeat=n_repeat))


# class ConvBn(nn.Module):
#     def __init__(self, with_relu=False):
#         super().__init__()
#         layers = [nn.Conv2d(64, 64, 3, bias=True), nn.BatchNorm2d(64)]
#         if with_relu:
#             layers.append(nn.ReLU())
#         self.conv = nn.Sequential(*layers)
#         self.quant_wrap = QuantWrapper(self.conv)
#         self.with_relu = with_relu

#     def forward(self, x):
#         return self.quant_wrap(x)

#     def fuse_model(self):
#         indices = ["0", "1"]
#         if self.with_relu:
#             indices.append("2")
#         fuse_modules(self.conv, indices, inplace=True)


def test_torch_qconv2d():
    # pt_inp = torch.rand((1, 64, 64, 64))
    # inp = pt_inp.numpy()
    inp = get_imagenet_input()
    pt_inp = torch.from_numpy(inp)

    # qmodel = ConvBn().eval()
    qmodel = qresnet50(pretrained=True).eval()

    quantize_model(qmodel, pt_inp)

    script_module = torch.jit.trace(qmodel, pt_inp).eval()

    input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
    input_shapes = [(input_name, inp.shape)]
    relay_mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    relay_mod = convert_conv2d_layout(relay_mod, {"qnn.conv2d": ["NHWC", "HWIO"]})

    target = "llvm --device arm_cpu --mtriple aarch64-apple-darwin -mattr=+v8.2a,+dotprod"

    tasks = autotvm.task.extract_from_program(
        relay_mod["main"],
        target=target,
        params=params,
    )

    for task in tasks:
        print(task)

    return
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

        print(task.mod)
        # print(mod)

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
            lib = relay.build(relay_mod, target=target, params=params)

    dev = tvm.cpu(0)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    runtime.set_input(input_name, inp)
    runtime.run()

    n_repeat = 100

    print(runtime.benchmark(dev, number=1, repeat=n_repeat))

    if True:
        tvm_result, rt_mod = run_tvm_model(
            relay_mod, params, input_name, inp, target=target
        )
        print(rt_mod.benchmark(dev, number=1, repeat=n_repeat))


def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)


def arm_conv2d_relay():
    data_shape = (1, 64, 56, 56)
    weight_shape = (64, 64, 3, 3)
    bias_shape = (weight_shape[0],)
    padding = (1, 1)

    bias = relay.var("bias", shape=bias_shape, dtype="int32")

    conv2d = get_conv2d_nchw(data_shape, weight_shape, padding)
    bias_add = relay.nn.bias_add(conv2d, bias)

    out = bias_add + relay.const(1, dtype="int32")
    # out = conv2d

    relay_mod = tvm.IRModule.from_expr(out)

    dev = tvm.device(target, 0)

    data = np.random.randint(low=-127, high=128, size=data_shape).astype("int8")
    weight_np = np.random.randint(low=-127, high=128, size=weight_shape).astype("int8")
    bias_np = np.random.randint(low=-127, high=128, size=bias_shape).astype("int32")
    params = {"weight": weight_np, "bias": bias_np}

    use_nhwc = False

    if use_nhwc:
        relay_mod = convert_conv2d_layout(relay_mod, {"nn.conv2d": ["NHWC", "HWIO"]})

    # print(relay.transform.InferType()(relay_mod))

    def schedule_fn(task, sch):
        if "conv2d" not in task.task_name:
            return False

        block = sch.get_block("conv2d_NCHWc_int8")

        # Looks up schedule_rule annotation.
        # See the comment in test_tune_relay_manual_tir_vnni().
        schedule_rule = sch.get(block).annotations["schedule_rule"]

        if "conv2d_NCHWc_int8" in schedule_rule:
            print(sch.mod.script())
            schedule_conv2d_nchwc(sch, block)
            return True

        return False

    with tvm.transform.PassContext(config={"relay.FuseOps.link_params": 1}):
        database = apply_fixed_schedules(relay_mod, target, params, schedule_fn)

    print("building")

    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            # opt_mod, _ = relay.optimize(relay_mod, target=target, params=params)
            # print(opt_mod)
            lib = relay.build(relay_mod, target=target, params=params)
            # print(lib.lib.get_source("asm"))

    # print(type(lib))
    # return
    mod = lib["default"](dev)
    runtime = tvm.contrib.graph_executor.GraphModule(mod)

    runtime.set_input("data", tvm.nd.array(data, dev))
    runtime.run()

    out = runtime.get_output(0).numpy()

    ref_exec = relay.create_executor(
        "vm", mod=relay_mod, device=tvm.cpu(0), target="llvm"
    )
    # ref = ref_exec.evaluate()(*[data, weight_np, bias_np]).numpy()
    ref = ref_exec.evaluate()(*[data]).numpy()

    np.testing.assert_equal(out, ref)

    print("ok")
    print(runtime.benchmark(dev, number=1, repeat=100))


if __name__ == "__main__":
    arm_conv2d_relay()
    # test_torchvision()
    # test_torch_qconv2d()

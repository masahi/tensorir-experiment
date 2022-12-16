import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor

import onnxruntime
import onnx

from tvm.contrib.download import download_testdata
from tvm.tir.tensor_intrin.x86 import VNNI_DOT_16x4_INTRIN as VNNI_INTRIN
from tvm import meta_schedule as ms
from PIL import Image


SCH_RULES_FOR_VNNI = [
    ms.schedule_rule.ApplyCustomRule(),
    ms.schedule_rule.AutoInline(
        into_producer=False,
        into_consumer=True,
        inline_const_tensor=True,
        disallow_if_then_else=True,
        require_injective=True,
        require_ordered=True,
        disallow_op=["tir.exp"],
    ),
    ms.schedule_rule.AddRFactor(max_jobs_per_core=16, max_innermost_factor=64),
    ms.schedule_rule.MultiLevelTilingWithIntrin(
        VNNI_INTRIN,
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
        reuse_write=ms.schedule_rule.ReuseType(
            req="may",
            levels=[1, 2],
            scope="global",
        ),
    ),
    ms.schedule_rule.MultiLevelTiling(
        structure="SSRSRS",
        tile_binds=None,
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
        reuse_write=ms.schedule_rule.ReuseType(
            req="may",
            levels=[1, 2],
            scope="global",
        ),
    ),
    ms.schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=16,
        max_vectorize_extent=64,
        unroll_max_steps=[0, 16, 64, 512],
        unroll_explicit=True,
    ),
    ms.schedule_rule.RandomComputeLocation(),
]

POSTPROCS_FOR_VNNI = [
    ms.postproc.DisallowDynamicLoop(),
    ms.postproc.RewriteParallelVectorizeUnroll(),
    ms.postproc.RewriteReductionBlock(),
    ms.postproc.RewriteTensorize(vectorize_init_loop=True),
]

def get_input():
    img_url = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    img_path = download_testdata(img_url, "imagenet_cat.png", module="data")

    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")

    img_data = np.transpose(img_data, (2, 0, 1))

    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

    return np.expand_dims(norm_img_data, axis=0).astype("float32")


def get_labels():
    labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    labels_path = download_testdata(labels_url, "synset.txt", module="data")

    with open(labels_path, "r") as f:
        labels = [l.rstrip() for l in f]

    return labels


def run_tvm(mod, params, target):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    module.set_input("input", inp)
    module.run()
    return module.get_output(0).numpy()[0]


def tune(mod, params, target):
    work_dir = "work"
    with ms.Profiler() as profiler:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=target,
            work_dir=work_dir,
            max_trials_global=20000,
            num_trials_per_iter=32,
            max_trials_per_task=128,
            params=params,
            space=ms.space_generator.PostOrderApply(
                sch_rules=SCH_RULES_FOR_VNNI,
                postprocs=POSTPROCS_FOR_VNNI,
            ),
            module_equality="anchor-block",
        )
        lib = ms.relay_integration.compile_relay(
            database=database,
            mod=converted_mod,
            target=target,
            params=params,
        )

    print(profiler.table())

    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    module.set_input("input", inp)
    module.run()

    return module.get_output(0).numpy().numpy()


inp = get_input()

onnx_path = "regnet_x_400mf_qint8.onnx"

ort_sess = onnxruntime.InferenceSession(onnx_path)

onnx_input_dict = {"input": inp}
ort_output = ort_sess.run(None, onnx_input_dict)[0][0]

model_onnx = onnx.load(onnx_path)
# mod, params = relay.frontend.from_onnx(
#     model_onnx,
#     shape={"input": (1, 3, 224, 224)},
#     freeze_params=True
# )

# with open("mod.json", "w") as fo:
#     fo.write(tvm.ir.save_json(mod))

# with open("mod.params", "wb") as fo:
#     fo.write(relay.save_param_dict(params))

with open("mod.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())

with open("mod.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

passes = tvm.transform.Sequential([
    relay.transform.InferType(),
    relay.transform.FakeQuantizationToInteger(),
])

qmod = passes(mod)

target = "llvm -mcpu=cascadelake -num-cores 6"

# tvm_output = run_tvm(mod, params, target)
# tvm_output_quant = run_tvm(qmod, params, target)
tvm_output = tune(qmod, params, target)


print(np.max(np.abs(tvm_output - ort_output)), np.mean(np.abs(tvm_output - ort_output)))
# print(np.max(np.abs(tvm_output - tvm_output_quant)), np.mean(np.abs(tvm_output - tvm_output_quant)))

labels = get_labels()

ranks = np.argsort(tvm_output)[::-1]
for rank in ranks[0:5]:
    print("class='%s'" % labels[rank])

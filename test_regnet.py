import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor

import onnxruntime
import onnx

from tvm.contrib.download import download_testdata
from PIL import Image


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


inp = get_input()

onnx_path = "regnet_x_400mf_qint8.onnx"

ort_sess = onnxruntime.InferenceSession(onnx_path)

onnx_input_dict = {"input": inp}
ort_output = ort_sess.run(None, onnx_input_dict)[0][0]


# model_onnx = onnx.load(onnx_path)
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

target = "llvm"

tvm_output = run_tvm(mod, params, target)
tvm_output_quant = run_tvm(passes(mod), params, target)

print(np.max(np.abs(tvm_output - ort_output)), np.mean(np.abs(tvm_output - ort_output)))
print(np.max(np.abs(tvm_output - tvm_output_quant)), np.mean(np.abs(tvm_output - tvm_output_quant)))

labels = get_labels()

ranks = np.argsort(tvm_output_quant)[::-1]
for rank in ranks[0:5]:
    print("class='%s'" % labels[rank])

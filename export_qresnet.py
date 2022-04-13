import torch
from torchvision.models.quantization import resnet18 as qresnet18
from torchvision.models.quantization import resnet50 as qresnet50

import tvm
from tvm import relay


def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    torch.quantization.convert(model, inplace=True)


qmodel = qresnet50(pretrained=True).eval()

input_name = "input"
pt_inp = torch.randn([1, 3, 224, 224])
quantize_model(qmodel, pt_inp)
script_module = torch.jit.trace(qmodel, pt_inp).eval()


input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

with open("models/qresnet50.json", "w") as fo:
    fo.write(tvm.ir.save_json(mod))

with open("models/qresnet50.params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

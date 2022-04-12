import numpy as np

import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, RandomTuner

import warnings
warnings.simplefilter('ignore')

input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
inp = np.random.randn(1, 3, 224, 224).astype("float32")

with open("models/qresnet50.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())

with open("models/qresnet50.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

target = "llvm -mcpu=cascadelake"

measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(
        number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
    ),
)

def tune(log_file, task, use_random=False):
    if use_random:
        tuner_obj = RandomTuner(task)
    else:
        tuner_obj = XGBTuner(task, loss_type="rank")

    n_trial = len(task.config_space)
    tuner_obj.tune(
        n_trial=n_trial,
        early_stopping=None,
        measure_option=measure_option,
        callbacks=[
            autotvm.callback.progress_bar(n_trial),
            autotvm.callback.log_to_file(log_file),
        ],
    )

with tvm.transform.PassContext(opt_level=3):
    opt_mod, opt_params = relay.optimize(mod, params=params, target=target)

tasks = autotvm.task.extract_from_program(
    opt_mod["main"],
    target=target,
    params=opt_params,
)

log_file = "qresnet50_vnni.log"

for task in tasks[:-1]:
    tune(log_file, task)

with autotvm.apply_history_best(log_file):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, params=params, target=target)

dev = tvm.device(target, 0)
runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

print(runtime.benchmark(dev, number=1, repeat=50).mean)

import torch
import tempfile
from os import path as osp
import numpy as np

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.meta_schedule import ReplayTraceConfig
from tvm.meta_schedule.database import JSONDatabase
from tvm.meta_schedule.tune import tune_relay
from tvm.target.target import Target


target = "nvidia/geforce-rtx-3070"
input_shape = (1, 512, 7, 7)
dev = tvm.cuda()
data = tvm.nd.array(np.random.randn(*input_shape).astype("float32"), dev)

input_name = "inp"
torch_mod = torch.nn.Softmax(dim=1)

scripted_model = torch.jit.trace(torch_mod, torch.rand(input_shape).float())
mod, params = relay.frontend.from_pytorch(scripted_model, [(input_name, input_shape)])

target = Target(target)
with tempfile.TemporaryDirectory() as work_dir:
    rt_mod1: tvm.runtime.Module = tune_relay(
        mod=mod,
        params=params,
        target=target,
        config=ReplayTraceConfig(
            num_trials_per_iter=32,
            max_trials_per_task=32,
            max_trials_global=32,
        ),
        work_dir=work_dir,
        database=JSONDatabase(
            osp.join(work_dir, "workload.json"), osp.join(work_dir, "records.json")
        ),
    )

    path_lib = "softmax.tar"
    rt_mod1.export_library(path_lib)

    # Compile without meta-scheduler for correctness check
    with tvm.transform.PassContext(opt_level=0):
        rt_mod2 = relay.build(mod, target=target, params=params)

    def get_output(data, lib):
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_name, data)
        module.run()
        return module.get_output(0).numpy()

    # Check correctness
    actual_output = get_output(data, rt_mod1)
    expected_output = get_output(data, rt_mod2)
    assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4)

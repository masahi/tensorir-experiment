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
from tvm.topi.transform import layout_transform
import tvm.topi.testing
from tvm.meta_schedule.database import TuningRecord, JSONDatabase
from tvm.meta_schedule.testing.tlcbench import load_quantized_bert_base

from tvm import rpc
from tvm.contrib import utils, ndk

import arm_common


tracker_host = os.environ["TVM_TRACKER_HOST"]
tracker_port = int(os.environ["TVM_TRACKER_PORT"])
key = "android"

arch = "aarch64"
# target = "llvm -device arm_cpu -mtriple=%s-linux-android -mattr=+neon" % arch
target = "llvm --device arm_cpu --mtriple aarch64-linux-gnu -mattr=+v8.2a,+dotprod"


fbgemm_workloads = [
    (64, 800, 320),
    (64, 768, 512),
    (16, 256, 512),
    (128, 128, 128),
    (256, 512, 256),
    (1024, 1024, 1024),
]

bert_workloads = [(128, 768, 3072), (128, 768, 768), (128, 3072, 768)]


def test_dense():
    # # target = "llvm -mcpu=cascadelake --num-cores=4"
    # tracker = rpc.connect_tracker(tracker_host, tracker_port)
    # remote = tracker.request(key, priority=0, session_timeout=0)
    # dev = remote.cpu(0)

    for M, N, K in fbgemm_workloads + bert_workloads:
        data_shape = (M, K)
        weight_shape = (N, K)

        data_dtype = "int8"

        data = relay.var("data", shape=data_shape, dtype=data_dtype)
        weight = relay.var("weight", shape=weight_shape, dtype="int8")
        bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
        dense = relay.nn.dense(data, weight, out_dtype="int32")
        out = relay.nn.bias_add(dense, bias)
        mod = tvm.IRModule.from_expr(out)

        a = np.random.uniform(1, 10, size=data_shape).astype(data_dtype)
        b = np.random.uniform(1, 10, size=weight_shape).astype("int8")
        c = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

        params = {"weight": b, "bias":c}

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        asm = lib.lib.get_source("asm")
        assert "dot" in asm
        return

        fname = "lib_%d_%d_%d.so" % (M, N, K)
        temp = utils.tempdir()
        path_dso_cpu = temp.relpath(fname)
        lib.export_library(path_dso_cpu, ndk.create_shared)

        # Establish remote connection with target hardware
        # target = "llvm -mcpu=cascadelake --num-cores=4"
        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        remote = tracker.request(key, priority=0, session_timeout=0)
        dev = remote.cpu(0)

        remote.upload(path_dso_cpu)
        lib = remote.load_module(fname)

        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        runtime.set_input("data", a)
        runtime.run()

        out = runtime.get_output(0).numpy()
        ref = np.dot(a.astype("int32"), b.transpose().astype("int32")) + c

        np.testing.assert_equal(out, ref)

        gflops = (N * M * K) * 2 / 1e9

        time_ms = runtime.benchmark(dev, number=1, repeat=100).mean * 1e3
        print(
            "matmul with VNNI TIR tensorization: %f ms, %f GFLOPS"
            % (time_ms, gflops / (time_ms / 1e3))
        )


test_dense()

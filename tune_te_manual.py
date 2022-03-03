import pytest
import tempfile
import tvm
from tvm.script import tir as T
from tvm import te, tir
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing import te_workload
import tvm.testing
import numpy as np
import os
from tvm.contrib import nvcc
import sys

def test_integration_matmul():
    N = 512
    M = 512
    K = 512
    workload = te_workload.matmul(n=N, m=M, k=K)
    workload = te.create_prim_func(workload)

    def schedule(sch: tir.Schedule):
        block = sch.get_block("C")
        # Step 1. Rule-Multi-Level-Tiling
        i, j, k = sch.get_loops(block)
        i_factors = sch.sample_perfect_tile(i, n=5)
        j_factors = sch.sample_perfect_tile(j, n=5)
        k_factors = sch.sample_perfect_tile(k, n=3)
        i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
        k0, k1, k2 = sch.split(k, k_factors)
        # pylint: enable=invalid-name
        sch.reorder(
            # fmt: off
            i0, j0,   # S => blockIdx.x
            i1, j1,   # S => blockIdx.y
            j2, i2,   # S => threadIdx.x
            # cache_write here
            k0,       # R
            # vectorized cooperative fetching here
            k1,       # R
            i3, j3,   # S
            k2,       # R
            i4, j4,
            # S
            # fmt: on
        )
        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idx = sch.fuse(j2, i2)
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idx, "threadIdx.x")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, k0, True)
            sch.annotate(block_read, "meta_schedule.cooperative_fetch", 1)
            return block_read

        A_sh = fetch_to_shared(block, 1, 2)
        B_sh = fetch_to_shared(block, 2, 2)


    # with tempfile.TemporaryDirectory() as work_dir:
    #     sch = ms.tune_tir(
    #         mod=workload,
    #         target=tvm.target.Target("vulkan -from_device=0"),
    #         # use replay or evolutionary search
    #         config=ms.ReplayTraceConfig(
    #             num_trials_per_iter=256,
    #             num_trials_total=256,
    #         ),
    #         # config=ms.EvolutionarySearchConfig(),
    #         work_dir=work_dir,
    #         space=ms.space_generator.ScheduleFn(schedule)
    #         )
    #     if sch is None:
    #         print("No valid schedule found!")
    #     else:
    #         print(sch.mod.script())
    #         print(sch.trace)

    ir_module = tvm.IRModule({"main": workload})
    sch = tvm.tir.Schedule(ir_module)
    schedule(sch)

    dev = tvm.device("vulkan -from_device=0", 0)
    a_np = np.random.uniform(size=(N, K)).astype("float32")
    b_np = np.random.uniform(size=(K, M)).astype("float32")
    c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
    f = tvm.build(sch.mod['main'], target="vulkan -from_device=0", name="dense")
    print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    gflops = (N*M*K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


if __name__ == "__main__":
    test_integration_matmul()

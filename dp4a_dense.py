import tvm
from tvm import te, tir, relay
from tvm import meta_schedule as ms
import tempfile
import tvm.testing
import numpy as np
import os
from tvm.contrib import nvcc
import sys
from tvm.tir.tensor_intrin import DP4A_INTRIN, AMDGPU_SDOT4_INTRIN


def matmul(n: int, m: int, k: int):
    a = te.placeholder((n, k), name="A", dtype="int8")
    b = te.placeholder((m, k), name="B", dtype="int8")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(
            a[i, k].astype("int32") * b[j, k].astype("int32"), axis=[k]
        ),
        name="C",
    )
    return (a, b, c)


def test_integration_matmul():
    N = 512
    M = 512
    K = 512
    workload = matmul(n=N, m=M, k=K)
    workload = te.create_prim_func(workload)

    def schedule(sch: tir.Schedule):
        block = sch.get_block("C")
        # Step 1. Rule-Multi-Level-Tiling
        i, j, k = sch.get_loops(block)

        i_factors = sch.sample_perfect_tile(i, n=3)
        j_factors = sch.sample_perfect_tile(j, n=3)
        by, ty, yi = sch.split(i, factors=i_factors)
        bx, tx, xi = sch.split(j, factors=j_factors)
        ko, ki = sch.split(k, [None, 4])
        ko_factors = sch.sample_perfect_tile(ko, n=2)
        ko, kt = sch.split(ko, factors=ko_factors)

        # pylint: enable=invalid-name
        sch.reorder(by, bx, ty, tx, yi, xi)

        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        CC = sch.cache_write(block, 0, "local")
        sch.reverse_compute_at(CC, tx)

        # print(len(sch.get_loops(CC)))

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, ko, True)
            sch.annotate(block_read, "meta_schedule.cooperative_fetch", 1)
            return block_read

        A_sh = fetch_to_shared(block, 0, 2)
        B_sh = fetch_to_shared(block, 1, 2)

        dec = sch.decompose_reduction(block, ko)
        # print(sch.mod)

        # print(sch.mod)
        sch.tensorize(ki, AMDGPU_SDOT4_INTRIN)
        print(sch.mod)

    with tempfile.TemporaryDirectory() as work_dir:
        sch = ms.tune_tir(
            mod=workload,
            target=tvm.target.Target("rocm"),
            # use replay or evolutionary search
            config=ms.ReplayTraceConfig(
                num_trials_per_iter=32, max_trials_per_task=32, max_trials_global=32
            ),
            # config=ms.EvolutionarySearchConfig(),
            work_dir=work_dir,
            space=ms.space_generator.ScheduleFn(schedule),
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)

    # ir_module = tvm.IRModule({"main": workload})
    # sch = tvm.tir.Schedule(ir_module)
    # schedule(sch)

    dev = tvm.device("rocm", 0)
    a_np = np.random.uniform(1, 10, size=(N, K)).astype("int8")
    b_np = np.random.uniform(1, 10, size=(M, K)).astype("int8")
    c_np = np.dot(a_np.astype("int32"), b_np.astype("int32").transpose())
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype="int32"), dev)
    f = tvm.build(sch.mod["main"], target="rocm", name="dense")
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)
    print("ok")

    evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    gflops = (N*M*K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))

    print(f.imported_modules[0].get_source("asm"))


def sdot4_bmm_relay():
    M = 1024
    N = 1024
    K = 1024
    data_shape = (8, M, K)
    weight_shape = (8, N, K)

    data_dtype = "int8"
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    dense = relay.nn.batch_matmul(data, weight, out_dtype="int32")
    out = dense

    relay_mod = tvm.IRModule.from_expr(out)

    target = "rocm"
    dev = tvm.device(target, 0)

    data = np.random.uniform(1, 10, size=data_shape).astype("int8")
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")
    bias_np = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

    ref = (
        relay.create_executor("vm", mod=relay_mod, device=dev, target=target)
        .evaluate()(*[data, weight_np])
        .numpy()
    )


if __name__ == "__main__":
    # test_integration_matmul()
    sdot4_bmm_relay()

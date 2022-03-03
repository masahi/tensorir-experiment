import tvm
from tvm.script import tir as T
from tvm import te, tir
from tvm import meta_schedule as ms
import tvm.testing
import numpy as np


def matmul(n: int, m: int, k: int):
    X = te.placeholder((m, k), name="X", dtype="uint8")
    packedW = te.placeholder((n // 16, k // 4, 16, 4), name="packedW", dtype="int8")

    ak = te.reduce_axis((0, k), name="k")
    out = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packedW[tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4),  j % 16, ak % 4].astype("int32"),
            axis=ak,
        ),
        name="C",
    )
    return [X, packedW, out]


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
        # i_factors = sch.sample_perfect_tile(i, n=5)
        # j_factors = sch.sample_perfect_tile(j, n=5)
        # k_factors = sch.sample_perfect_tile(k, n=3)
        # i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
        # j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
        # k0, k1, k2 = sch.split(k, k_factors)
        # # pylint: enable=invalid-name
        # sch.reorder(
        #     # fmt: off
        #     i0, j0,   # S => blockIdx.x
        #     i1, j1,   # S => blockIdx.y
        #     j2, i2,   # S => threadIdx.x
        #     # cache_write here
        #     k0,       # R
        #     # vectorized cooperative fetching here
        #     k1,       # R
        #     i3, j3,   # S
        #     k2,       # R
        #     i4, j4,
        #     # S
        #     # fmt: on
        # )

    ir_module = tvm.IRModule({"main": workload})
    sch = tvm.tir.Schedule(ir_module)
    schedule(sch)

    target = "llvm -mcpu=cascadelake"
    dev = tvm.device(target, 0)
    a_np = np.random.uniform(1, 10, size=(N, K)).astype("uint8")
    b_np = np.random.uniform(1, 10, size=(M, K)).astype("int8")
    c_np = np.dot(a_np.astype("int32"), b_np.transpose().astype("int32"))

    packW = np.random.uniform(1, 10, size=(N // 16, (K // 4), 16, 4)).astype("int8")

    for r_idx in range(N // 16):
        for ko in range(K // 4):
            for s_idx in range(16):
                for t_idx in range(4):
                    packW[r_idx][ko][s_idx][t_idx] = b_np[r_idx * 16 + s_idx][ko * 4 + t_idx]


    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(packW, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype="int32"), dev)
    f = tvm.build(sch.mod['main'], target=target, name="dense")

    # print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    # evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    # gflops = (N*M*K) * 2 / 1e9
    # time_ms = evaluator(a, b, c).mean * 1e3
    # print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


if __name__ == "__main__":
    test_integration_matmul()

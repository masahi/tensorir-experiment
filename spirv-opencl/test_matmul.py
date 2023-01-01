import numpy as np

import tvm
from tvm import te, tir


def get_matmul_packed(m, n, k, factor):
    X = te.placeholder((m, k), name="X", dtype="float32")
    W = te.placeholder((k // factor, n, factor), name="W", dtype="float32")
    ak = te.reduce_axis((0, k), name="k")

    matmul = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("float32") * W[ak // factor, j, ak % factor].astype("float32"),
            axis=ak,
        ),
        name="compute",
    )

    return te.create_prim_func([X, W, matmul])


M, N, K = 8, 8, 32
func = get_matmul_packed(M, N, K, 2)
sch = tir.Schedule(func)
block = sch.get_block("compute")

i, j, k = sch.get_loops(block)
i_outer, i_inner = sch.split(i, factors=[None, 8])
sch.bind(i_outer, "blockIdx.x")

k_outer, k_inner = sch.split(k, factors=[None, 16])
sch.reorder(i_outer, k_outer, i_inner, j, k_inner)

def fetch_to_shared(block, idx, ndim):
    block_read = sch.cache_read(block, idx, "shared")
    sch.compute_at(block_read, k_outer)
    warp_size = 8

    fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

    _, f_2, f_3 = sch.split(fused, factors=[None, warp_size, 4])
    sch.bind(f_2, 'threadIdx.x')


fetch_to_shared(block, 0, 2)
fetch_to_shared(block, 1, 3)

init = sch.decompose_reduction(block, sch.get_loops(block)[1])

print(sch.mod.script())

target = "opencl -device=spirv"
# target = "vulkan"

f = tvm.build(sch.mod, target=target)
dev = tvm.device(target, 0)

A = tvm.nd.array(np.random.randn(M, K).astype("float32"), dev)
B = tvm.nd.array(np.random.randn(K // 2, N, 2).astype("float32"), dev)
C = tvm.nd.array(np.random.randn(M, N).astype("float32"), dev)

f(A, B, C)

A_np = A.numpy()
B_np = B.numpy()
B_unpacked = np.zeros((K, N)).astype("float32")

for k in range(B_unpacked.shape[0]):
    for j in range(B_unpacked.shape[1]):
        B_unpacked[k, j] = B_np[k // 2, j, k % 2]

out = C.numpy()
ref = np.dot(A_np, B_unpacked)

print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))

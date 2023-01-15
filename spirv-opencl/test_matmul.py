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


M, N, K = 16, 8, 16
func = get_matmul_packed(M, N, K, 2)
sch = tir.Schedule(func)
block = sch.get_block("compute")

i, j, k = sch.get_loops(block)
# sch.bind(i_outer, "blockIdx.x")

i_outer, i_inner = sch.split(i, factors=[None, 2])
j_outer, j_inner = sch.split(j, factors=[None, 8])
sch.reorder(i_outer, j_outer, i_inner, j_inner)
sch.bind(sch.fuse(i_outer, j_outer), "blockIdx.x")

sch.bind(i_inner, "threadIdx.y")
sch.bind(j_inner, "threadIdx.x")

def fetch_to_shared(block, idx, ndim):
    block_read = sch.cache_read(block, idx, "shared")
    sch.compute_at(block_read, j_inner)
    warp_size = 8

    fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

    f_1, f_2, f_3 = sch.split(fused, factors=[2, warp_size, None])
    sch.bind(f_1, 'threadIdx.y')
    sch.bind(f_2, 'threadIdx.x')
    sch.vectorize(f_3)

fetch_to_shared(block, 0, 2)
fetch_to_shared(block, 1, 3)

init = sch.decompose_reduction(block, sch.get_loops(block)[3])

print(sch.mod.script())

target = "opencl -device=spirv"
# target = "vulkan"
# target = "opencl"

f = tvm.build(sch.mod, target=target)

# print(f.imported_modules[0].get_source())

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

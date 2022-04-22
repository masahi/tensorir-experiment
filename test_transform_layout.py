import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.te import create_prim_func
from tvm.tir import Schedule


def dense(n: int, m: int, k: int):
    a = te.placeholder((n, k), name="A")
    b = te.placeholder((m, k), name="B")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(a[i, k] * b[j, k], axis=[k]),
        name="C",
    )
    return (a, b, c)

matmul = create_prim_func(dense(n=16, m=16, k=16))

sch = Schedule(matmul)
block = sch.get_block("C")

A_shared = sch.cache_read(block, 0, "shared")
B_shared = sch.cache_read(block, 1, "shared")

def shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    matrix_id = 2 * (j // 8) + (i // 8)

    return thread_id, 2 * matrix_id + j % 2

block = sch.get_block("C")
A_warp = sch.cache_read(block, 0, "warp")
B_warp = sch.cache_read(block, 1, "warp")

sch.transform_layout(
    A_warp,
    0,
    "write",
    index_map=shared_16x16_to_ldmatrix_32x8_layout
)

sch.transform_layout(
    B_warp,
    0,
    "write",
    index_map=shared_16x16_to_ldmatrix_32x8_layout
)

print(sch.mod.script())

# print(tvm.lower(sch.mod["main"]))
f = tvm.build(sch.mod["main"], target="llvm", name="dense")
dev = tvm.cpu(0)

a_np = np.random.uniform(size=(16, 16)).astype("float32")
b_np = np.random.uniform(size=(16, 16)).astype("float32")
c_np = np.dot(a_np, b_np.transpose())

a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(np.zeros((16, 16), dtype="float32"), dev)

# print(f.imported_modules[0].get_source())
f(a, b, c)
tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

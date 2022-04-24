import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.te import create_prim_func
from tvm.tir import Schedule
from tvm.script import tir as T
from tvm import tir


@T.prim_func
def ldmatrix_a_desc(a: T.handle, c: T.handle) -> None:
    A_shared = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="shared"
    )
    A_warp = T.match_buffer(
        c, (32, 8), "float16", align=128, offset_factor=16, scope="warp"
    )

    with T.block("root"):
        T.reads(A_shared[0:16, 0:16])
        T.writes(A_warp[0:32, 0:8])

        for ax0, ax1 in T.grid(16, 16):
            with T.block("A_shared_warp"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A_shared[v0, v1])
                T.writes(
                    A_warp[v0 % 8 * 4 + v1 % 8 // 2, v1 // 8 * 4 + v0 // 8 * 2 + v1 % 2]
                )
                A_warp[
                    v0 % 8 * 4 + v1 % 8 // 2, v1 // 8 * 4 + v0 // 8 * 2 + v1 % 2
                ] = A_shared[v0, v1]


@T.prim_func
def ldmatrix_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A_shared = T.match_buffer(
        a,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="shared",
        strides=[s1, s0],
    )
    A_warp = T.match_buffer(
        c, (32, 8), "float16", align=128, offset_factor=16, scope="warp"
    )
    with T.block("root"):
        T.reads(A_shared[0:16, 0:16])
        T.writes(A_warp[0:32, 0:8])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        T.evaluate(
            T.ptx_ldmatrix(
                0,
                4,
                ".b16",
                A_warp.data,
                8 * tx,
                A_shared.data,
                16 * (tx % 16) + 8 * (tx // 16),
                dtype="float16",
            )
        )


@T.prim_func
def ldmatrix_b_desc(a: T.handle, c: T.handle) -> None:
    B_shared = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="shared"
    )
    B_warp = T.match_buffer(
        c, (32, 8), "float16", align=128, offset_factor=16, scope="warp"
    )

    with T.block("root"):
        T.reads(B_shared[0:16, 0:16])
        T.writes(B_warp[0:32, 0:8])

        for ax0, ax1 in T.grid(16, 16):
            with T.block("B_shared_warp"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(B_shared[v0, v1])
                T.writes(
                    B_warp[v0 % 8 * 4 + v1 % 8 // 2, v1 // 8 * 4 + v0 // 8 * 2 + v1 % 2]
                )
                B_warp[
                    v0 % 8 * 4 + v1 % 8 // 2, v1 // 8 * 4 + v0 // 8 * 2 + v1 % 2
                ] = B_shared[v0, v1]


@T.prim_func
def ldmatrix_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    B_shared = T.match_buffer(
        a,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="shared",
        strides=[s1, s0],
    )
    B_warp = T.match_buffer(
        c, (32, 8), "float16", align=128, offset_factor=16, scope="warp"
    )
    with T.block("root"):
        T.reads(B_shared[0:16, 0:16])
        T.writes(B_warp[0:32, 0:8])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        T.evaluate(
            T.ptx_ldmatrix(
                0,
                4,
                ".b16",
                B_warp.data,
                8 * tx,
                B_shared.data,
                16 * (tx % 16) + 8 * (tx // 16),
                dtype="float16",
            )
        )


@T.prim_func
def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32, 8), "float16", align=128, offset_factor=16, scope="warp")
    B = T.match_buffer(b, (32, 8), "float16", align=128, offset_factor=16, scope="warp")
    C = T.match_buffer(c, (32, 8), "float32", align=128, offset_factor=16, scope="warp")

    with T.block("root"):
        T.reads(C[0:32, 0:8], A[0:32, 0:8], B[0:32, 0:8])
        T.writes(C[0:32, 0:8])
        for i, j, k in T.grid(16, 16, 16):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [i, j, k])
                T.reads(
                    C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2],
                    A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2],
                    B[j % 8 * 4 + k % 8 // 2, k // 8 * 4 + j // 8 * 2 + k % 2],
                )
                T.writes(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2])
                C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] = C[
                    i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2
                ] + T.cast(
                    A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2],
                    "float32",
                ) * T.cast(
                    B[j % 8 * 4 + k % 8 // 2, k // 8 * 4 + j // 8 * 2 + k % 2],
                    "float32",
                )


@T.prim_func
def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32, 8), "float16", align=128, offset_factor=16, scope="warp")
    B = T.match_buffer(b, (32, 8), "float16", align=128, offset_factor=16, scope="warp")
    C = T.match_buffer(c, (32, 8), "float32", align=128, offset_factor=16, scope="warp")

    with T.block("root"):
        T.reads(C[0:32, 0:8], A[0:32, 0:8], B[0:32, 0:8])
        T.writes(C[0:32, 0:8])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        T.evaluate(
            T.ptx_mma(
                "m16n8k16",
                "row",
                "col",
                "fp16",
                "fp16",
                "fp32",
                A.data,
                A.elem_offset + tx * 8,
                B.data,
                B.elem_offset + tx * 8,
                C.data,
                C.elem_offset + tx * 8,
                False,
                dtype="float32",
            )
        )

        T.evaluate(
            T.ptx_mma(
                "m16n8k16",
                "row",
                "col",
                "fp16",
                "fp16",
                "fp32",
                A.data,
                A.elem_offset + tx * 8 + 4,
                B.data,
                B.elem_offset + tx * 8 + 4,
                C.data,
                C.elem_offset + tx * 8 + 4,
                False,
                dtype="float32",
            )
        )


tir.TensorIntrin.register("mma.ldmatrix_a", ldmatrix_a_desc, ldmatrix_a_impl)
tir.TensorIntrin.register("mma.ldmatrix_b", ldmatrix_b_desc, ldmatrix_b_impl)
tir.TensorIntrin.register("mma.mma_sync", mma_sync_desc, mma_sync_impl)


def dense(n: int, m: int, k: int):
    a = te.placeholder((n, k), name="A", dtype="float16")
    b = te.placeholder((m, k), name="B", dtype="float16")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(
            tvm.tir.Cast("float32", a[i, k]) * tvm.tir.Cast("float32", b[j, k]),
            axis=[k],
        ),
        name="C",
    )
    return (a, b, c)


matmul = create_prim_func(dense(n=16, m=16, k=16))

sch = Schedule(matmul)
block = sch.get_block("C")

i, j, k = sch.get_loops(block)

i1, i2 = sch.split(i, factors=[None, 16])
sch.bind(i1, "blockIdx.x")
# sch.bind(i2, "threadIdx.x")

def fetch_to_shared(block, idx):
    block_read = sch.cache_read(block, idx, "shared")
    sch.compute_at(block_read, i1, True)
    warp_size = 32
    loops = sch.get_loops(block_read)
    fused = sch.fuse(*loops[-2:])
    f_0, f_1 = sch.split(fused, factors=[None, warp_size])
    sch.bind(f_1, "threadIdx.x")

    return block_read


A_shared = fetch_to_shared(block, 0)
B_shared = fetch_to_shared(block, 1)


def shared_16x16_to_ldmatrix_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    matrix_id = 2 * (j // 8) + (i // 8)

    return thread_id, 2 * matrix_id + j % 2

block = sch.get_block("C")

A_warp = sch.cache_read(block, 0, "warp")

sch.transform_layout(A_warp, 0, "write", index_map=shared_16x16_to_ldmatrix_32x8_layout)
sch.tensorize(sch.get_loops(A_warp)[1], "mma.ldmatrix_a")

B_warp = sch.cache_read(block, 1, "warp")
sch.transform_layout(B_warp, 0, "write", index_map=shared_16x16_to_ldmatrix_32x8_layout)
sch.tensorize(sch.get_loops(B_warp)[1], "mma.ldmatrix_b")

C_warp = sch.cache_write(block, 0, "warp")
sch.reverse_compute_at(C_warp, sch.get_loops(block)[0])
sch.transform_layout(C_warp, 0, "read", index_map=shared_16x16_to_ldmatrix_32x8_layout)

# bind cache write
warp_loop1, warp_loop2 = sch.get_loops(C_warp)[-2:]
f_0, f_1 = sch.split(warp_loop1, factors=[None, 8])
f_2, f_3 = sch.split(warp_loop2, factors=[None, 4])
sch.reorder(f_1, f_2, f_0, f_3)
fused_1 = sch.fuse(f_1, f_2)
fused_2 = sch.fuse(f_0, f_3)
sch.bind(fused_1, "threadIdx.x")

block_init_c = sch.decompose_reduction(block, sch.get_loops(block)[1])
init_loop1, init_loop2 = sch.get_loops(block_init_c)[-2:]
f_0, f_1 = sch.split(init_loop1, factors=[None, 8])
f_2, f_3 = sch.split(init_loop2, factors=[None, 4])
sch.reorder(f_1, f_2, f_0, f_3)
fused_1 = sch.fuse(f_1, f_2)
fused_2 = sch.fuse(f_0, f_3)
sch.bind(fused_1, "threadIdx.x")

sch.tensorize(sch.get_loops(block)[1], "mma.mma_sync")

print(sch.mod.script())

# lowered = tvm.lower(sch.mod["main"])

target = "vulkan -from_device=0"
f = tvm.build(sch.mod["main"], target=target, name="dense")
# dev = tvm.device(target, 0)

# a_np = np.random.uniform(size=(16, 16)).astype("float16")
# b_np = np.random.uniform(size=(16, 16)).astype("float16")
# c_np = np.dot(a_np.astype("float32"), b_np.transpose().astype("float32"))

# a = tvm.nd.array(a_np, dev)
# b = tvm.nd.array(b_np, dev)
# c = tvm.nd.array(np.zeros((16, 16), dtype="float32"), dev)

# # print(f.imported_modules[0].get_source())
# f(a, b, c)
# tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

import tvm
from tvm.script import tir as T
from tvm import tir

# -----------------
# Tensor intrinsic registrations

@T.prim_func
def ldmatrix_a_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16, "float16", align=128, offset_factor=16,
                         scope="shared")
    C = T.match_buffer(c, (32, 8), "float16", align=128, offset_factor=16,
                         scope="warp")

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 32, 0 : 8])

        for ax0, ax1 in T.grid(16, 16):
            with T.block("A_shared_warp"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v0, v1])
                T.writes(C[v0 % 8 * 4 + v1 % 8 // 2, v1 // 8 * 4 + v0 // 8 * 2 + v1 % 2])
                C[v0 % 8 * 4 + v1 % 8 // 2, v1 // 8 * 4 + v0 // 8 * 2 + v1 % 2] = A[v0, v1]


@T.prim_func
def ldmatrix_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = T.match_buffer(c, (32, 8), "float16", align=128, offset_factor=16, scope="warp")
    tx = T.env_thread("threadIdx.x")

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 32, 0 : 8])

        T.evaluate(
            T.ptx_ldmatrix(
                0,
                4,
                ".b16",
                C.data,
                8 * tx,
                A.data,
                16 * (tx % 16) + 8 * (tx // 16),
                dtype="float16",
            )
        )


@T.prim_func
def ldmatrix_b_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32, 8), "float16", align=128, offset_factor=16,
                         scope="shared")
    C = T.match_buffer(c, (32, 8), "float16", align=128, offset_factor=16,
                         scope="warp")

    with T.block("root"):
        T.reads(A[0 : 32, 0 : 8])
        T.writes(C[0 : 32, 0 : 8])
        for i, j in T.grid(32, 8):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def ldmatrix_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (32, 8), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = T.match_buffer(c, (32, 8), "float16", align=128, offset_factor=16, scope="warp")
    tx = T.env_thread("threadIdx.x")

    with T.block("root"):
        T.reads(A[0 : 32, 0 : 8])
        T.writes(C[0 : 32, 0 : 8])

        T.evaluate(
            T.ptx_ldmatrix(
                0,
                4,
                ".b16",
                A.data,
                8 * tx,
                C.data,
                8 * tx,
                dtype="float16",
            )
        )


# @T.prim_func
# def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
#     A = T.match_buffer(a, (32, 8), "float16", align=128, offset_factor=16,
#                          scope="warp")
#     B = T.match_buffer(b, (32, 8), "float16", align=128, offset_factor=16,
#                          scope="warp")
#     C = T.match_buffer(c, (32, 8), "float32", align=128, offset_factor=16,
#                          scope="warp")

#     with T.block("root"):
#         T.reads(C[0 : 32, 0 : 8], A[0 : 32, 0 : 8], B[0: 32, 0 : 8])
#         T.writes(C[0 : 32, 0 : 8])
#         for i, j, k in T.grid(16, 16, 16):
#             with T.block("C"):
#                 i, j, k = T.axis.remap("SSR", [i0, i1, i2])
#                 T.reads(A_shared_warp[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2], B_shared_warp[j % 8 * 4 + k % 8 // 2, k // 8 * 4 + j // 8 * 2 + k % 2])
#                 T.writes(C[i, j])
#                 C[i, j] = C[i, j] + A_shared_warp[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2] * B_shared_warp[j % 8 * 4 + k % 8 // 2, k // 8 * 4 + j // 8 * 2 + k % 2]



# @T.prim_func
# def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
#     A = T.match_buffer(a, (32, 8), "float16", align=128, offset_factor=16,
#                          scope="warp")
#     B = T.match_buffer(b, (32, 8), "float16", align=128, offset_factor=16,
#                          scope="warp")
#     C = T.match_buffer(c, (32, 8), "float32", align=128, offset_factor=16,
#                          scope="warp")

#     with T.block("root"):
#         T.reads(C[0 : 32, 0 : 8], A[0 : 32, 0 : 8], B[0: 32, 0 : 8])
#         T.writes(C[0 : 32, 0 : 8])
#         tx = T.env_thread("threadIdx.x")
#         T.evaluate(
#             T.ptx_mma(
#                 "m16n8k16",
#                 "row",
#                 "col",
#                 "fp16",
#                 "fp16",
#                 "fp16",
#                 A.data,
#                 A.elem_offset + tx * 8,
#                 B.data,
#                 B.elem_offset + tx * 8,
#                 C.data,
#                 C.elem_offset + tx * 8,
#                 False,
#                 dtype="float16",
#             )
#         )

#         T.evaluate(
#             T.ptx_mma(
#                 "m16n8k16",
#                 "row",
#                 "col",
#                 "fp16",
#                 "fp16",
#                 "fp16",
#                 A.data,
#                 A.elem_offset + tx * 8 + 4,
#                 B.data,
#                 B.elem_offset + tx * 8 + 4,
#                 C.data,
#                 C.elem_offset + tx * 8 + 4,
#                 False,
#                 dtype="float16",
#             )
#         )


tir.TensorIntrin.register("mma.ldmatrix_a", ldmatrix_a_desc, ldmatrix_a_impl)
tir.TensorIntrin.register("mma.ldmatrix_b", ldmatrix_b_desc, ldmatrix_b_impl)
tir.TensorIntrin.register("mma.mma_sync", mma_sync_desc, mma_sync_impl)

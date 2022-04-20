import tvm
from tvm.script import tir as T
from tvm import tir

# -----------------
# Tensor intrinsic registrations

@T.prim_func
def ldmatrix_a_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                         scope="shared")
    C = T.match_buffer(c, (16,16), "float16", align=128, offset_factor=16,
                         scope="wmma.matrix_a")

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def ldmatrix_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = T.match_buffer(c, (8,), "float16", align=128, offset_factor=16, scope="local")
    tx = T.env_thread("threadIdx.x")

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 8])

        T.evaluate(
            T.ptx_ldmatrix(
                0,
                4,
                ".b16",
                C.data,
                0,
                C.data,
                16 * (tx % 16) + 8 * (tx // 16),
                dtype="float16",
            )
        )


@T.prim_func
def ldmatrx_b_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                         scope="shared")
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16,
                         scope="wmma.matrix_b")

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def ldmatrix_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = T.match_buffer(c, (8,), "float16", align=128, offset_factor=16, scope="local")

    tx = T.env_thread("threadIdx.x")


    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 8])

        T.evaluate(
            T.ptx_ldmatrix(
                0,
                4,
                ".b16",
                C.data,
                0,
                C.data,
                16 * (tx % 16) + 8 * (tx // 16),
                dtype="float16",
            )
        )

@T.prim_func
def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                         scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16,
                         scope="wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16,
                         scope="wmma.accumulator")

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0: 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block(""):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + T.cast(A[vii, vkk], 'float32') * T.cast(B[vkk, vjj], 'float32')


@T.prim_func
def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                         scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16,
                         scope="wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16,
                         scope="wmma.accumulator")

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0: 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(T.tvm_mma_sync(C.data, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                                  A.data, A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                                  B.data, B.elem_offset // 256 + T.floordiv(T.floormod(B.elem_offset, 256), 16),
                                  C.data, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16), dtype='handle'))


@T.prim_func
def mma_fill_desc(c: T.handle) -> None:
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")

    with T.block("root"):
        T.reads()
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = T.float32(0)


@T.prim_func
def mma_fill_impl(c: T.handle) -> None:
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    with T.block("root"):
        T.reads()
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16), T.float32(0), dtype="handle"))


@T.prim_func
def mma_store_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def mma_store_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global", strides=[s1, s0])
    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(T.tvm_store_matrix_sync(
            A.data, 16, 16, 16, A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16), C.access_ptr("w"), s1, "row_major",
            dtype="handle"))



tir.TensorIntrin.register("mma.ldmatrix_a", ldmatrix_a_desc, ldmatrix_a_impl)
tir.TensorIntrin.register("mma.ldmatrix_b", ldmatrix_b_desc, ldmatrix_b_impl)

tir.TensorIntrin.register("mma.mma_sync", mma_sync_desc, mma_sync_impl)
tir.TensorIntrin.register("mma.fill", mma_fill_desc, mma_fill_impl)
tir.TensorIntrin.register("mma.store", mma_store_desc, mma_store_impl)


# ---------------------

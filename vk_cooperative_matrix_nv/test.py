import numpy as np

import tvm
from tvm.script import tir as T
from tvm.tir import TensorIntrin
from tvm import te, tir


@T.prim_func
def cooperative_matrix_load_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=64, offset_factor=8, scope="shared")
    C = T.match_buffer(
        c, (16, 16), "float16", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


def get_load_impl(column_major):
    @T.prim_func
    def cooperative_matrix_load_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.var("int32")
        s0 = T.var("int32")
        A = T.match_buffer(
            a,
            (16, 16),
            "float16",
            align=64,
            offset_factor=8,
            scope="shared",
            strides=[s1, s0],
        )
        C = T.match_buffer(
            c, (16, 16), "float16", align=64, offset_factor=8, scope="cooperative_matrix_nv"
        )

        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 32)
            T.evaluate(
                T.cooperative_matrix_load_NV(
                    C.data,
                    C.elem_offset,
                    A.access_ptr("r"),
                    s1,
                    column_major,
                    dtype="handle",
                )
            )

    return cooperative_matrix_load_impl

@T.prim_func
def cooperative_matrix_store_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float32", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )
    C = T.match_buffer(c, (16, 16), "float32", align=64, offset_factor=8, scope="global")
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def cooperative_matrix_store_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float32", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )
    C = T.match_buffer(
        c,
        (16, 16),
        "float32",
        align=64,
        offset_factor=8,
        scope="global",
        strides=[s1, s0],
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)
        T.evaluate(
            T.cooperative_matrix_store_NV(
                C.access_ptr("w"), A.data, A.elem_offset, s1, False, dtype="handle"
            )
        )


@T.prim_func
def cooperative_matrix_fill_desc(c: T.handle) -> None:
    C = T.match_buffer(
        c, (16, 16), "float32", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )

    with T.block("root"):
        T.reads()
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = T.float32(0)


@T.prim_func
def cooperative_matrix_fill_impl(c: T.handle) -> None:
    C = T.match_buffer(
        c, (16, 16), "float32", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )
    with T.block("root"):
        T.reads()
        T.writes(C[0:16, 0:16])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)
        T.evaluate(
            T.cooperative_matrix_fill_NV(
                C.data, C.elem_offset, T.float32(0), dtype="handle"
            )
        )


@T.prim_func
def cooperative_matrix_mad_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )
    B = T.match_buffer(
        b, (16, 16), "float16", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )
    C = T.match_buffer(
        c, (16, 16), "float32", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + T.cast(A[vii, vkk], "float32") * T.cast(
                    B[vkk, vjj], "float32"
                )


@T.prim_func
def cooperative_matrix_mad_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )
    B = T.match_buffer(
        b, (16, 16), "float16", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )
    C = T.match_buffer(
        c, (16, 16), "float32", align=64, offset_factor=8, scope="cooperative_matrix_nv"
    )

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)
        T.evaluate(
            T.cooperative_matrix_mad_NV(
                A.data,
                A.elem_offset,
                B.data,
                B.elem_offset,
                C.data,
                C.elem_offset,
                dtype="handle",
            )
        )


TensorIntrin.register(
    "cooperative_matrix_load_a", cooperative_matrix_load_desc, get_load_impl(False)
)
TensorIntrin.register(
    "cooperative_matrix_load_b", cooperative_matrix_load_desc, get_load_impl(True)
)

TensorIntrin.register(
    "cooperative_matrix_store", cooperative_matrix_store_desc, cooperative_matrix_store_impl
)
TensorIntrin.register(
    "cooperative_matrix_fill", cooperative_matrix_fill_desc, cooperative_matrix_fill_impl
)
TensorIntrin.register("cooperative_matrix_mad", cooperative_matrix_mad_desc, cooperative_matrix_mad_impl)


def get_matmul(m, n, k):
    X = te.placeholder((m, k), name="X", dtype="float16")
    W = te.placeholder((k, n), name="W", dtype="float16")
    ak = te.reduce_axis((0, k), name="k")

    matmul = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("float32")
            * W[ak, j].astype("float32"),
            axis=ak,
        ),
        name="compute",
    )

    return te.create_prim_func([X, W, matmul])


M, N, K = 16, 16, 16

func = get_matmul(M, N, K)
sch = tir.Schedule(func)
block = sch.get_block("compute")

i, j, k = sch.get_loops(block)
i_outer, i_inner = sch.split(i, factors=[None, 16])
j_outer, j_inner = sch.split(j, factors=[None, 16])
k_outer, k_inner = sch.split(k, factors=[None, 16])
sch.reorder(i_outer, j_outer, k_outer, i_inner, j_inner, k_inner)
fused_outer = sch.fuse(i_outer, j_outer)
sch.bind(fused_outer, "blockIdx.x")


def fetch_to_shared(block, idx, ndim):
    block_read = sch.cache_read(block, idx, "shared")
    sch.compute_at(block_read, k_outer)
    warp_size = 32

    fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

    vector_size = 4
    _, f_2, f_3 = sch.split(fused, factors=[None, warp_size, vector_size])
    sch.bind(f_2, "threadIdx.x")
    sch.vectorize(f_3)


def tensorize_load(block, dim, j_split, intrin):
    loops = sch.get_loops(block)
    i, j = loops[-dim : (len(loops) - dim + 2)]

    i0, i1 = sch.split(i, factors=[None, 16])
    j0, j1 = sch.split(j, factors=[None, j_split])
    sch.reorder(i0, j0, i1, j1)
    sch.unroll(i0)
    sch.unroll(j0)
    sch.tensorize(i1, intrin)


fetch_to_shared(block, 0, 2)
fetch_to_shared(block, 1, 2)

A_joint = sch.cache_read(block, 0, "cooperative_matrix_nv")
B_joint = sch.cache_read(block, 1, "cooperative_matrix_nv")

tensorize_load(A_joint, 2, 16, "cooperative_matrix_load_a")
tensorize_load(B_joint, 2, 16, "cooperative_matrix_load_b")

store = sch.cache_write(block, 0, "cooperative_matrix_nv")
sch.reverse_compute_at(store, fused_outer)
init = sch.decompose_reduction(block, sch.get_loops(block)[1])

sch.tensorize(sch.get_loops(init)[1], "cooperative_matrix_fill")
sch.tensorize(sch.get_loops(store)[1], "cooperative_matrix_store")
sch.tensorize(sch.get_loops(block)[2], "cooperative_matrix_mad")

print(sch.mod.script())

target = "vulkan -from_device=0"
f = tvm.build(sch.mod, target=target)
# dev = tvm.device(target, 0)

# A = tvm.nd.array(np.random.randn(M, K).astype("float16"), dev)
# B = tvm.nd.array(np.random.randn(K, N).astype("float16"), dev)
# C = tvm.nd.array(np.random.randn(M, N).astype("float32"), dev)

# f(A, B, C)
# out = C.numpy()

# A_np = A.numpy()
# B_np = B.numpy()
# ref = np.dot(A_np.astype("float32"), B_np.astype("float32"))

# print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))

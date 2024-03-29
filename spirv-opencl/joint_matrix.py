import numpy as np

import tvm
from tvm.script import tir as T
from tvm.tir import TensorIntrin
from tvm import te, tir


@T.prim_func
def joint_matrix_load_a_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 16), "float16", align=64, offset_factor=8, scope="shared")
    C = T.match_buffer(
        c, (8, 16), "float16", align=64, offset_factor=8, scope="joint_matrix"
    )

    with T.block("root"):
        T.reads(A[0:8, 0:16])
        T.writes(C[0:8, 0:16])
        for i, j in T.grid(8, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def joint_matrix_load_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a,
        (8, 16),
        "float16",
        align=64,
        offset_factor=8,
        scope="shared",
        strides=[s1, s0],
    )
    C = T.match_buffer(
        c, (8, 16), "float16", align=64, offset_factor=8, scope="joint_matrix"
    )

    with T.block("root"):
        T.reads(A[0:8, 0:16])
        T.writes(C[0:8, 0:16])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(
            T.joint_matrix_load_intel(
                C.data,
                C.elem_offset,
                A.access_ptr("r"),
                s1,
                "row_major",
                dtype="handle",
            )
        )


@T.prim_func
def joint_matrix_load_b_desc(b: T.handle, c: T.handle) -> None:
    B = T.match_buffer(
        b, (8, 8, 2), "float16", align=64, offset_factor=8, scope="shared"
    )
    C = T.match_buffer(
        c, (8, 8, 2), "float16", align=64, offset_factor=8, scope="joint_matrix"
    )

    with T.block("root"):
        T.reads(B[0:8, 0:8, 0:2])
        T.writes(C[0:8, 0:8, 0:2])
        for i, j, k in T.grid(8, 8, 2):
            with T.block("load"):
                vii, vjj, vkk = T.axis.remap("SSS", [i, j, k])
                C[vii, vjj, vkk] = B[vii, vjj, vkk]


@T.prim_func
def joint_matrix_load_b_impl(b: T.handle, c: T.handle) -> None:
    s2 = T.var("int32")
    s1 = T.var("int32")
    s0 = T.var("int32")

    B = T.match_buffer(
        b,
        (8, 8, 2),
        "float16",
        align=64,
        offset_factor=8,
        scope="shared",
        strides=[s2, s1, s0],
    )
    C = T.match_buffer(
        c, (8, 8, 2), "float16", align=64, offset_factor=8, scope="joint_matrix"
    )

    with T.block("root"):
        T.reads(B[0:8, 0:8, 0:2])
        T.writes(C[0:8, 0:8, 0:2])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(
            T.joint_matrix_load_intel(
                C.data, C.elem_offset, B.access_ptr("r"), s2, "packed_b", dtype="handle"
            )
        )


@T.prim_func
def joint_matrix_store_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (8, 8), "float32", align=64, offset_factor=8, scope="joint_matrix"
    )
    C = T.match_buffer(c, (8, 8), "float32", align=64, offset_factor=8, scope="global")
    with T.block("root"):
        T.reads(A[0:8, 0:8])
        T.writes(C[0:8, 0:8])
        for i, j in T.grid(8, 8):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def joint_matrix_store_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a, (8, 8), "float32", align=64, offset_factor=8, scope="joint_matrix"
    )
    C = T.match_buffer(
        c,
        (8, 8),
        "float32",
        align=64,
        offset_factor=8,
        scope="global",
        strides=[s1, s0],
    )

    with T.block("root"):
        T.reads(A[0:8, 0:8])
        T.writes(C[0:8, 0:8])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(
            T.joint_matrix_store_intel(
                C.access_ptr("w"), A.data, A.elem_offset, s1, dtype="handle"
            )
        )


@T.prim_func
def joint_matrix_fill_desc(c: T.handle) -> None:
    C = T.match_buffer(
        c, (8, 8), "float32", align=64, offset_factor=8, scope="joint_matrix"
    )

    with T.block("root"):
        T.reads()
        T.writes(C[0:8, 0:8])
        for i, j in T.grid(8, 8):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = T.float32(0)


@T.prim_func
def joint_matrix_fill_impl(c: T.handle) -> None:
    C = T.match_buffer(
        c, (8, 8), "float32", align=64, offset_factor=8, scope="joint_matrix"
    )
    with T.block("root"):
        T.reads()
        T.writes(C[0:8, 0:8])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(
            T.joint_matrix_fill_intel(
                C.data, C.elem_offset, T.float32(0), dtype="handle"
            )
        )


@T.prim_func
def joint_matrix_mad_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (8, 16), "float16", align=64, offset_factor=8, scope="joint_matrix"
    )
    B = T.match_buffer(
        b, (8, 8, 2), "float16", align=64, offset_factor=8, scope="joint_matrix"
    )
    C = T.match_buffer(
        c, (8, 8), "float32", align=64, offset_factor=8, scope="joint_matrix"
    )

    with T.block("root"):
        T.reads(C[0:8, 0:8], A[0:8, 0:16], B[0:8, 0:8, 0:2])
        T.writes(C[0:8, 0:8])
        for i, j, k in T.grid(8, 8, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + T.cast(A[vii, vkk], "float32") * T.cast(
                    B[vkk // 2, vjj, vkk % 2], "float32"
                )


@T.prim_func
def joint_matrix_mad_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (8, 16), "float16", align=64, offset_factor=8, scope="joint_matrix"
    )
    B = T.match_buffer(
        b, (8, 8, 2), "float16", align=64, offset_factor=8, scope="joint_matrix"
    )
    C = T.match_buffer(
        c, (8, 8), "float32", align=64, offset_factor=8, scope="joint_matrix"
    )

    with T.block("root"):
        T.reads(C[0:8, 0:8], A[0:8, 0:16], B[0:8, 0:8, 0:2])
        T.writes(C[0:8, 0:8])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(
            T.joint_matrix_mad_intel(
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
    "joint_matrix_load_a", joint_matrix_load_a_desc, joint_matrix_load_a_impl
)
TensorIntrin.register(
    "joint_matrix_load_b", joint_matrix_load_b_desc, joint_matrix_load_b_impl
)
TensorIntrin.register(
    "joint_matrix_store", joint_matrix_store_desc, joint_matrix_store_impl
)
TensorIntrin.register(
    "joint_matrix_fill", joint_matrix_fill_desc, joint_matrix_fill_impl
)
TensorIntrin.register("joint_matrix_mad", joint_matrix_mad_desc, joint_matrix_mad_impl)


def get_matmul_packed(m, n, k, factor):
    X = te.placeholder((m, k), name="X", dtype="float16")
    W = te.placeholder((k // factor, n, factor), name="W", dtype="float16")
    ak = te.reduce_axis((0, k), name="k")

    matmul = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("float32")
            * W[ak // factor, j, ak % factor].astype("float32"),
            axis=ak,
        ),
        name="compute",
    )

    return te.create_prim_func([X, W, matmul])


M, N, K = 8, 8, 16
# M, N, K = 16, 16, 32
# M, N, K = 4096, 4096, 4096
func = get_matmul_packed(M, N, K, 2)
sch = tir.Schedule(func)
block = sch.get_block("compute")

i, j, k = sch.get_loops(block)
i_outer, i_inner = sch.split(i, factors=[None, 8])
j_outer, j_inner = sch.split(j, factors=[None, 8])
k_outer, k_inner = sch.split(k, factors=[None, 16])
sch.reorder(i_outer, j_outer, k_outer, i_inner, j_inner, k_inner)
fused_outer = sch.fuse(i_outer, j_outer)
sch.bind(fused_outer, "blockIdx.x")


def fetch_to_shared(block, idx, ndim):
    block_read = sch.cache_read(block, idx, "shared")
    sch.compute_at(block_read, k_outer)
    warp_size = 8

    fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

    vector_size = 8
    _, f_2, f_3 = sch.split(fused, factors=[None, warp_size, vector_size])
    sch.bind(f_2, "threadIdx.x")
    sch.vectorize(f_3)


def tensorize_load(block, dim, j_split, intrin):
    loops = sch.get_loops(block)
    i, j = loops[-dim : (len(loops) - dim + 2)]

    i0, i1 = sch.split(i, factors=[None, 8])
    j0, j1 = sch.split(j, factors=[None, j_split])
    sch.reorder(i0, j0, i1, j1)
    sch.unroll(i0)
    sch.unroll(j0)
    sch.tensorize(i1, intrin)


fetch_to_shared(block, 0, 2)
fetch_to_shared(block, 1, 3)

A_joint = sch.cache_read(block, 0, "joint_matrix")
B_joint = sch.cache_read(block, 1, "joint_matrix")

tensorize_load(A_joint, 2, 16, "joint_matrix_load_a")
tensorize_load(B_joint, 3, 8, "joint_matrix_load_b")

store = sch.cache_write(block, 0, "joint_matrix")
sch.reverse_compute_at(store, fused_outer)
init = sch.decompose_reduction(block, sch.get_loops(block)[1])

sch.tensorize(sch.get_loops(init)[1], "joint_matrix_fill")
sch.tensorize(sch.get_loops(store)[1], "joint_matrix_store")
sch.tensorize(sch.get_loops(block)[2], "joint_matrix_mad")

print(sch.mod.script())

target = "opencl -device=spirv -supports_float16=1"

f = tvm.build(sch.mod, target=target)
dev = tvm.device(target, 0)

A = tvm.nd.array(np.random.randn(M, K).astype("float16"), dev)
B = tvm.nd.array(np.random.randn(K // 2, N, 2).astype("float16"), dev)
C = tvm.nd.array(np.random.randn(M, N).astype("float32"), dev)

f(A, B, C)

A_np = A.numpy()
B_np = B.numpy()
B_unpacked = np.zeros((K, N)).astype("float16")

for k in range(B_unpacked.shape[0]):
    for j in range(B_unpacked.shape[1]):
        B_unpacked[k, j] = B_np[k // 2, j, k % 2]

out = C.numpy()
ref = np.dot(A_np.astype("float32"), B_unpacked.astype("float32"))

print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))

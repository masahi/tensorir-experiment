import numpy as np

import tvm
from tvm.script import tir as T
from tvm.tir import TensorIntrin
from tvm import te, tir


@T.prim_func
def joint_matrix_load_a_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 16), "float16", align=64, offset_factor=16,
                         scope="global")
    C = T.match_buffer(c, (8, 16), "float16", align=64, offset_factor=16,
                         scope="joint_matrix")

    with T.block("root"):
        T.reads(A[0 : 8, 0 : 16])
        T.writes(C[0 : 8, 0 : 16])
        for i, j in T.grid(8, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def joint_matrix_load_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (8, 16), "float16", align=64, offset_factor=16, scope="global", strides=[s1, s0])
    C = T.match_buffer(c, (8, 16), "float16", align=64, offset_factor=16, scope="joint_matrix")

    with T.block("root"):
        T.reads(A[0 : 8, 0 : 16])
        T.writes(C[0 : 8, 0 : 16])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(T.joint_matrix_load_intel(C.data, A.data, s1, "row_major", dtype="handle"))


@T.prim_func
def joint_matrix_load_b_desc(b: T.handle, c: T.handle) -> None:
    B = T.match_buffer(b, (8, 8, 2), "float16", align=64, offset_factor=16,
                         scope="global")
    C = T.match_buffer(c, (8, 8, 2), "float16", align=64, offset_factor=16,
                         scope="joint_matrix")

    with T.block("root"):
        T.reads(B[0 : 8, 0 : 8, 0 : 2])
        T.writes(C[0 : 8, 0 : 8, 0 : 2])
        for i, j, k in T.grid(8, 8, 2):
            with T.block("load"):
                vii, vjj, vkk = T.axis.remap("SSS", [i, j, k])
                C[vii, vjj, vkk] = B[vii, vjj, vkk]


@T.prim_func
def joint_matrix_load_b_impl(b: T.handle, c: T.handle) -> None:
    s2 = T.var("int32")
    s1 = T.var("int32")
    s0 = T.var("int32")

    B = T.match_buffer(b, (8, 8, 2), "float16", align=64, offset_factor=16, scope="global", strides=[s2, s1, s0])
    C = T.match_buffer(c, (8, 8, 2), "float16", align=64, offset_factor=16, scope="joint_matrix")

    with T.block("root"):
        T.reads(B[0 : 8, 0 : 8, 0 : 2])
        T.writes(C[0 : 8, 0 : 8, 0 : 2])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(T.joint_matrix_load_intel(C.data, B.data, s2, "packed_b", dtype="handle"))


@T.prim_func
def joint_matrix_store_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 8), "float32", align=64, offset_factor=16, scope="joint_matrix")
    C = T.match_buffer(c, (8, 8), "float32", align=64, offset_factor=16, scope="global")
    with T.block("root"):
        T.reads(A[0 : 8, 0 : 8])
        T.writes(C[0 : 8, 0 : 8])
        for i, j in T.grid(8, 8):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def joint_matrix_store_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (8, 8), "float32", align=64, offset_factor=16, scope="joint_matrix")
    C = T.match_buffer(c, (8, 8), "float32", align=64, offset_factor=16, scope="global", strides=[s1, s0])

    with T.block("root"):
        T.reads(A[0 : 8, 0 : 8])
        T.writes(C[0 : 8, 0 : 8])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(T.joint_matrix_store_intel(C.data, A.data, s1, dtype="handle"))


@T.prim_func
def joint_matrix_fill_desc(c: T.handle) -> None:
    C = T.match_buffer(c, (8, 8), "float32", align=64, offset_factor=16, scope="joint_matrix")

    with T.block("root"):
        T.reads()
        T.writes(C[0 : 8, 0 : 8])
        for i, j in T.grid(8, 8):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = T.float32(0)


@T.prim_func
def joint_matrix_fill_impl(c: T.handle) -> None:
    C = T.match_buffer(c, (8, 8), "float32", align=64, offset_factor=16, scope="joint_matrix")
    with T.block("root"):
        T.reads()
        T.writes(C[0 : 8, 0 : 8])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(T.joint_matrix_fill_intel(C.data, T.float32(0), dtype="handle"))


@T.prim_func
def joint_matrix_mad_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 16), "float16", align=64, offset_factor=16,
                         scope="joint_matrix")
    B = T.match_buffer(b, (8, 8, 2), "float16", align=64, offset_factor=16,
                         scope="joint_matrix")
    C = T.match_buffer(c, (8, 8), "float32", align=64, offset_factor=16,
                         scope="joint_matrix")

    with T.block("root"):
        T.reads(C[0 : 8, 0 : 8], A[0 : 8, 0 : 16], B[0: 8, 0 : 8, 0 : 2])
        T.writes(C[0 : 8, 0 : 8])
        for i, j, k in T.grid(8, 8, 16):
            with T.block("update"):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + T.cast(A[vii, vkk], 'float32') * T.cast(B[vkk // 2, vjj, vkk % 2], 'float32')


@T.prim_func
def joint_matrix_mad_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 16), "float16", align=64, offset_factor=16,
                         scope="joint_matrix")
    B = T.match_buffer(b, (8, 8, 2), "float16", align=64, offset_factor=16,
                         scope="joint_matrix")
    C = T.match_buffer(c, (8, 8), "float32", align=64, offset_factor=16,
                         scope="joint_matrix")

    with T.block("root"):
        T.reads(C[0 : 8, 0 : 8], A[0 : 8, 0 : 16], B[0: 8, 0 : 8, 0 : 2])
        T.writes(C[0 : 8, 0 : 8])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 8)
        T.evaluate(T.joint_matrix_mad_intel(A.data, B.data, C.data, dtype='handle'))


TensorIntrin.register("joint_matrix_load_a", joint_matrix_load_a_desc, joint_matrix_load_a_impl)
TensorIntrin.register("joint_matrix_load_b", joint_matrix_load_b_desc, joint_matrix_load_b_impl)
TensorIntrin.register("joint_matrix_store",  joint_matrix_store_desc, joint_matrix_store_impl)
TensorIntrin.register("joint_matrix_fill",  joint_matrix_fill_desc, joint_matrix_fill_impl)
TensorIntrin.register("joint_matrix_mad",  joint_matrix_mad_desc, joint_matrix_mad_impl)


def get_matmul_packed(m, n, k, factor):
    X = te.placeholder((m, k), name="X", dtype="float16")
    W = te.placeholder((k // factor, n, factor), name="W", dtype="float16")

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


m, n, k = 8, 8, 16
func = get_matmul_packed(m, n, k, 2)
sch = tir.Schedule(func)
block = sch.get_block("compute")

i = sch.get_loops(block)[0]
i1, _ = sch.split(i, factors=[None, 8])
sch.bind(i1, "blockIdx.x")

A = sch.cache_read(block, 0, "joint_matrix")
sch.compute_at(A, i1)
B = sch.cache_read(block, 1, "joint_matrix")
sch.compute_at(B, i1)
C = sch.cache_write(block, 0, "joint_matrix")
sch.reverse_compute_at(C, i1)

sch.decompose_reduction(block, sch.get_loops(block)[1])

sch.tensorize(sch.get_loops(sch.get_block("X_joint_matrix"))[1], "joint_matrix_load_a")
sch.tensorize(sch.get_loops(sch.get_block("W_joint_matrix"))[1], "joint_matrix_load_b")
sch.tensorize(sch.get_loops(sch.get_block("compute_init"))[1], "joint_matrix_fill")
sch.tensorize(sch.get_loops(sch.get_block("compute_joint_matrix"))[1], "joint_matrix_store")
sch.tensorize(sch.get_loops(sch.get_block("compute_update"))[1], "joint_matrix_mad")

print(sch.mod.script())

target = "opencl -device=spirv -supports_int8=1 -supports_float16=1 -supports_int64=1 -supports_float64=1"

f = tvm.build(sch.mod, target=target)
dev = tvm.device(target, 0)

A = tvm.nd.array(np.random.randn(8, 16).astype("float16"), dev)
B = tvm.nd.array(np.random.randn(8, 8, 2).astype("float16"), dev)
C = tvm.nd.array(np.random.randn(8, 8).astype("float32"), dev)

f(A, B, C)

A_np = A.numpy()
B_np = B.numpy()
B_unpacked = np.zeros((16, 8)).astype("float16")

for k in range(B_unpacked.shape[0]):
    for j in range(B_unpacked.shape[1]):
        B_unpacked[k, j] = B_np[k // 2, j, k % 2]

out = C.numpy()
ref = np.dot(A_np.astype("float32"), B_unpacked.astype("float32"))

print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))

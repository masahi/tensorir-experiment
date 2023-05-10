import tempfile

import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.tir import TensorIntrin, IntImm, Cast
from tvm import te, tir


def get_schedule_fun(tune):
    def schedule(sch):
        block = sch.get_block("compute")

        i, j, k = sch.get_loops(block)

        i, i_inner = sch.split(i, factors=[None, 16])
        j, j_inner = sch.split(j, factors=[None, 16])
        k, k_inner = sch.split(k, factors=[None, 16])

        sch.reorder(
            i,
            j,
            k,
            i_inner,
            j_inner,
            k_inner,
        )

        block_outer = sch.blockize(i_inner)
        block_inner = block

        if tune:
            i_factors = sch.sample_perfect_tile(i, n=5)
            j_factors = sch.sample_perfect_tile(j, n=5)
            k_factors = sch.sample_perfect_tile(k, n=3)
            num_ty = sch.get(i_factors[2]) * sch.get(j_factors[2])
        else:
            i_factors, j_factors, k_factors = (
                [4, 8, 1, 8, 1],
                [1, 32, 4, 1, 2],
                [128, 1, 2],
            )
            num_ty = i_factors[2] * j_factors[2]

        i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
        k0, k1, k2 = sch.split(k, k_factors)

        sch.reorder(
            i0,
            j0,
            i1,
            j1,
            j2,
            i2,
            k0,
            k1,
            i3,
            j3,
            k2,
            i4,
            j4,
        )

        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, k0)
            vector_size = 4
            warp_size = 32
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

            _, f_1, f_2, f_3 = sch.split(
                fused, factors=[None, num_ty, warp_size, vector_size]
            )
            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.vectorize(f_3)

            # TODO: why it doesn't work
            # sch.storage_align(block_read, 0, axis=-2, factor=32, offset=8)
            return block_read

        fetch_to_shared(block_outer, 0, 2)
        fetch_to_shared(block_outer, 1, 2)

        loop = sch.get_loops(block_outer)[-1]
        A_mat = sch.cache_read(block_outer, 0, "cooperative_matrix_nv")
        B_mat = sch.cache_read(block_outer, 1, "cooperative_matrix_nv")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        store = sch.cache_write(block_outer, 0, "cooperative_matrix_nv")
        sch.reverse_compute_at(store, thread_idy)

        ii, jj = sch.get_loops(store)[-2:]
        io, ii = sch.split(ii, factors=[None, 16])
        jo, ji = sch.split(jj, factors=[None, 16])
        sch.reorder(io, jo, ii, ji)

        loop = sch.get_loops(block_outer)[3]
        block_init_c = sch.decompose_reduction(block_outer, loop)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        for l in sch.get_loops(sch.get_block("compute_o_init"))[-4:]:
            sch.unroll(l)

        for l in sch.get_loops(store)[-4:-2]:
            sch.unroll(l)

        for l in sch.get_loops(sch.get_block("compute_o_update"))[-5:]:
            sch.unroll(l)

        i, j = sch.get_loops(A_mat)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)
        sch.unroll(i0)
        sch.unroll(j0)
        sch.tensorize(i1, "cooperative_matrix_load")

        i, j = sch.get_loops(B_mat)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)
        sch.unroll(i0)
        sch.unroll(j0)
        sch.tensorize(i1, "cooperative_matrix_load")

        sch.tensorize(sch.get_loops(block_init_c_inner)[-2], "cooperative_matrix_fill")
        sch.tensorize(sch.get_loops(store)[-2], "cooperative_matrix_store")
        sch.tensorize(sch.get_loops(block_inner)[-3], "cooperative_matrix_mad")

    return schedule


@T.prim_func
def cooperative_matrix_load_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=64, offset_factor=8, scope="shared"
    )
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
            c,
            (16, 16),
            "float16",
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
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
                    16,
                    16,
                    s1,
                    column_major,
                    dtype="handle",
                )
            )

    return cooperative_matrix_load_impl


def get_store_desc(out_dtype="float32", out_scope="global"):
    @T.prim_func
    def cooperative_matrix_store_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (16, 16),
            out_dtype,
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
        )
        C = T.match_buffer(
            c, (16, 16), out_dtype, align=64, offset_factor=8, scope=out_scope
        )
        with T.block("root"):
            T.reads(A[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            for i, j in T.grid(16, 16):
                with T.block("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = A[vii, vjj]

    return cooperative_matrix_store_desc


def get_store_impl(out_dtype="float32", out_scope="global"):
    @T.prim_func
    def cooperative_matrix_store_impl(a: T.handle, c: T.handle) -> None:
        s1 = T.var("int32")
        s0 = T.var("int32")
        A = T.match_buffer(
            a,
            (16, 16),
            out_dtype,
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
        )
        C = T.match_buffer(
            c,
            (16, 16),
            out_dtype,
            align=64,
            offset_factor=8,
            scope=out_scope,
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

    return cooperative_matrix_store_impl


def get_fill_desc(out_dtype="float32"):
    zero = IntImm("int32", 0).astype(out_dtype)

    @T.prim_func
    def cooperative_matrix_fill_desc(c: T.handle) -> None:
        C = T.match_buffer(
            c,
            (16, 16),
            out_dtype,
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
        )

        with T.block("root"):
            T.reads()
            T.writes(C[0:16, 0:16])
            for i, j in T.grid(16, 16):
                with T.block("init"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    C[vii, vjj] = zero

    return cooperative_matrix_fill_desc


def get_fill_impl(out_dtype="float32"):
    zero = IntImm("int32", 0).astype(out_dtype)

    @T.prim_func
    def cooperative_matrix_fill_impl(c: T.handle) -> None:
        C = T.match_buffer(
            c,
            (16, 16),
            out_dtype,
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
        )

        with T.block("root"):
            T.reads()
            T.writes(C[0:16, 0:16])
            tx = T.env_thread("threadIdx.x")
            T.launch_thread(tx, 32)
            T.evaluate(
                T.cooperative_matrix_fill_NV(
                    C.data, C.elem_offset, 16, 16, zero, dtype="handle"
                )
            )

    return cooperative_matrix_fill_impl


def get_mad_desc(out_dtype="float32"):
    def maybe_cast(v):
        if out_dtype in ["float32", "int32"]:
            return Cast(out_dtype, v)
        return v

    @T.prim_func
    def cooperative_matrix_mad_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (16, 16),
            "float16",
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
        )
        B = T.match_buffer(
            b,
            (16, 16),
            "float16",
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
        )
        C = T.match_buffer(
            c,
            (16, 16),
            out_dtype,
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
        )

        with T.block("root"):
            T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
            T.writes(C[0:16, 0:16])
            for i, j, k in T.grid(16, 16, 16):
                with T.block("update"):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    C[vii, vjj] = C[vii, vjj] + maybe_cast(A[vii, vkk]) * maybe_cast(
                        B[vkk, vjj]
                    )

    return cooperative_matrix_mad_desc


def get_mad_impl(out_dtype="float32"):
    @T.prim_func
    def cooperative_matrix_mad_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            (16, 16),
            "float16",
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
        )
        B = T.match_buffer(
            b,
            (16, 16),
            "float16",
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
        )
        C = T.match_buffer(
            c,
            (16, 16),
            out_dtype,
            align=64,
            offset_factor=8,
            scope="cooperative_matrix_nv",
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

    return cooperative_matrix_mad_impl


def get_matmul(m, n, k, out_dtype="float32"):
    X = te.placeholder((m, k), name="X", dtype="float16")
    W = te.placeholder((k, n), name="W", dtype="float16")
    ak = te.reduce_axis((0, k), name="k")

    if out_dtype == "float32":
        matmul = te.compute(
            (m, n),
            lambda i, j: te.sum(
                X[i, ak].astype("float32") * W[ak, j].astype("float32"),
                axis=ak,
            ),
            name="compute",
        )
    else:
        matmul = te.compute(
            (m, n),
            lambda i, j: te.sum(X[i, ak] * W[ak, j], axis=ak),
            name="compute",
        )

    return te.create_prim_func([X, W, matmul])


out_dtype = "float32"

TensorIntrin.register(
    "cooperative_matrix_load", cooperative_matrix_load_desc, get_load_impl(False)
)

TensorIntrin.register(
    "cooperative_matrix_store",
    get_store_desc(out_dtype),
    get_store_impl(out_dtype),
)
TensorIntrin.register(
    "cooperative_matrix_fill", get_fill_desc(out_dtype), get_fill_impl(out_dtype)
)
TensorIntrin.register(
    "cooperative_matrix_mad", get_mad_desc(out_dtype), get_mad_impl(out_dtype)
)


tune = False
M, N, K = 4096, 4096, 4096
target = "vulkan -from_device=0"

workload = get_matmul(M, N, K, out_dtype)

if tune:
    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.tune_tir(
            mod=workload,
            target=tvm.target.Target(target),
            max_trials_global=128,
            work_dir=work_dir,
            space=ms.space_generator.ScheduleFn(get_schedule_fun(tune)),
        )
        sch = ms.tir_integration.compile_tir(db, workload, target)
        print(sch.trace)
else:
    sch = tir.Schedule(workload)
    get_schedule_fun(tune)(sch)

f = tvm.build(sch.mod, target=target)
dev = tvm.device(target, 0)

A = tvm.nd.array(np.random.randn(M, K).astype("float16"), dev)
B = tvm.nd.array(np.random.randn(K, N).astype("float16"), dev)
C = tvm.nd.array(np.random.randn(M, N).astype(out_dtype), dev)

f(A, B, C)

evaluator = f.time_evaluator(f.entry_name, dev, number=10)
gflops = (N * M * K) * 2 / 1e9
time_ms = evaluator(A, B, C).mean * 1e3
print("%f GFLOPS" % (gflops / (time_ms / 1e3)))

out = C.numpy()

A_np = A.numpy()
B_np = B.numpy()
ref = np.dot(A_np.astype("float32"), B_np.astype("float32"))

print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))

import pytest
import tempfile
import tvm
from tvm.script import tir as T
import tvm.meta_schedule.testing.te_workload as te_workload
from tvm import te, tir
from tvm import meta_schedule as ms
import tvm.testing
import numpy as np


@T.prim_func
def ldmatrix_a_desc(a: T.handle, c: T.handle) -> None:
    A_shared = T.match_buffer(
        a, (16, 8), "float16", align=128, offset_factor=1, scope="shared"
    )
    A_warp = T.match_buffer(
        c, (32, 4), "float16", align=128, offset_factor=1, scope="warp"
    )

    with T.block("root"):
        T.reads(A_shared[0:16, 0:8])
        T.writes(A_warp[0:32, 0:4])

        for ax0, ax1 in T.grid(16, 8):
            with T.block("A_shared_warp"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A_shared[v0, v1])
                T.writes(A_warp[v0 % 8 * 4 + v1 // 2, v0 // 8 * 2 + v1 % 2])
                A_warp[v0 % 8 * 4 + v1 // 2, v0 // 8 * 2 + v1 % 2] = A_shared[v0, v1]


@T.prim_func
def ldmatrix_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A_shared = T.match_buffer(
        a,
        (16, 8),
        "float16",
        align=128,
        offset_factor=1,
        scope="shared",
        strides=[s1, s0],
    )
    A_warp = T.match_buffer(
        c, (32, 4), "float16", align=128, offset_factor=1, scope="warp"
    )
    with T.block("root"):
        T.reads(A_shared[0:16, 0:8])
        T.writes(A_warp[0:32, 0:4])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        T.evaluate(
            T.ptx_ldmatrix(
                0,
                2,
                ".b16",
                A_warp.data,
                A_warp.elem_offset + 4 * tx,
                A_shared.access_ptr("r"),
                s1 * (tx % 16),
                dtype="float16",
            )
        )


@T.prim_func
def ldmatrix_b_desc(a: T.handle, c: T.handle) -> None:
    B_shared = T.match_buffer(
        a, (8, 8), "float16", align=128, offset_factor=1, scope="shared"
    )
    B_shared_warp = T.match_buffer(
        c, (32, 2), "float16", align=128, offset_factor=1, scope="warp"
    )

    with T.block("root"):
        T.reads(B_shared[0:8, 0:8])
        T.writes(B_shared_warp[0:32, 0:2])

        for ax0, ax1 in T.grid(8, 8):
            with T.block("A_shared_warp"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(B_shared[v0, v1])
                T.writes(B_shared_warp[v1 * 4 + v0 // 2, v0 % 2])
                B_shared_warp[v1 * 4 + v0 // 2, v0 % 2] = B_shared[v0, v1]


@T.prim_func
def ldmatrix_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    B_shared = T.match_buffer(
        a,
        (8, 8),
        "float16",
        align=128,
        offset_factor=1,
        scope="shared",
        strides=[s1, s0],
    )
    B_warp = T.match_buffer(
        c, (32, 2), "float16", align=128, offset_factor=1, scope="warp"
    )
    with T.block("root"):
        T.reads(B_shared[0:8, 0:8])
        T.writes(B_warp[0:32, 0:2])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        T.evaluate(
            T.ptx_ldmatrix(
                1,
                1,
                ".b16",
                B_warp.data,
                B_warp.elem_offset + 2 * tx,
                B_shared.access_ptr("r"),
                s1 * (tx % 8),
                dtype="float16",
            )
        )


@T.prim_func
def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [32, 4], dtype="float16", scope="warp")
    B = T.match_buffer(b, [32, 2], dtype="float16", scope="warp")
    C = T.match_buffer(c, [32, 4], dtype="float32", scope="warp")
    with T.block("root"):
        T.reads(C[0:32, 0:4], A[0:32, 0:4], B[0:32, 0:2])
        T.writes(C[0:32, 0:4])
        for i0, i1, i2 in T.grid(16, 8, 8):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(
                    C[i % 8 * 4 + j % 8 // 2, i % 16 // 8 * 2 + j % 2],
                    A[i % 8 * 4 + k % 8 // 2, i % 16 // 8 * 2 + k % 2],
                    B[j % 8 * 4 + k % 8 // 2, k % 2],
                )
                T.writes(C[i % 8 * 4 + j % 8 // 2, i % 16 // 8 * 2 + j % 2])
                C[i % 8 * 4 + j % 8 // 2, i % 16 // 8 * 2 + j % 2] = C[
                    i % 8 * 4 + j % 8 // 2, i % 16 // 8 * 2 + j % 2
                ] + T.cast(
                    A[i % 8 * 4 + k % 8 // 2, i % 16 // 8 * 2 + k % 2], "float32"
                ) * T.cast(
                    B[j % 8 * 4 + k % 8 // 2, k % 2], "float32"
                )


@T.prim_func
def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32, 4), "float16", align=128, offset_factor=1, scope="warp")
    B = T.match_buffer(b, (32, 2), "float16", align=128, offset_factor=1, scope="warp")
    C = T.match_buffer(c, (32, 4), "float32", align=128, offset_factor=1, scope="warp")

    with T.block("root"):
        T.reads(C[0:32, 0:4], A[0:32, 0:4], B[0:32, 0:2])
        T.writes(C[0:32, 0:4])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)
        T.evaluate(
            T.ptx_mma(
                "m16n8k8",
                "row",
                "col",
                "fp16",
                "fp16",
                "fp32",
                A.data,
                A.elem_offset + tx * 4,
                B.data,
                B.elem_offset + tx * 2,
                C.data,
                C.elem_offset + tx * 4,
                False,
                dtype="float32",
            )
        )


tir.TensorIntrin.register("mma.ldmatrix_a", ldmatrix_a_desc, ldmatrix_a_impl)
tir.TensorIntrin.register("mma.ldmatrix_b", ldmatrix_b_desc, ldmatrix_b_impl)
tir.TensorIntrin.register("mma_sync", mma_sync_desc, mma_sync_impl)


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


N = 4096
M = 4096
K = 4096

# workload = te.create_prim_func(dense(n=N, m=M, k=K))

workload = te.create_prim_func(te_workload.matmul_fp16(n=N, m=M, k=K))

tune = True


def schedule(sch: tir.Schedule):
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    # Step 1. Rule-Auto-Tensorize
    # pylint: disable=invalid-name
    i, i_tc = sch.split(i, factors=[None, 16])
    j, j_tc = sch.split(j, factors=[None, 8])
    k, k_tc = sch.split(k, factors=[None, 8])
    sch.reorder(
        # fmt: off
        i, j, k,
        # tensor core
        i_tc, j_tc, k_tc,
        # fmt: on
    )
    block_inner = sch.blockize(i_tc)

    block_outer, block_inner = block_inner, block
    del block

    # Step 2. Rule-Multi-Level-Tiling

    if tune:
        i_factors = sch.sample_perfect_tile(i, n=5)
        j_factors = sch.sample_perfect_tile(j, n=5)
        k_factors = sch.sample_perfect_tile(k, n=3)
    else:
        i_factors = [1, 16, 4, 2, 2]
        j_factors = [1, 64, 1, 8, 1]
        k_factors = [128, 4, 1]

    i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
    j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
    k0, k1, k2 = sch.split(k, k_factors)
    # pylint: enable=invalid-name
    sch.reorder(
        # fmt: off
        i0, j0,   # S => blockIdx.x
        i1, j1,   # S => blockIdx.y
        j2, i2,   # S => threadIdx.y
        # cache_write here
        k0,       # R
        # vectorized cooperative fetching here
        k1,       # R
        i3, j3,   # S
        k2,       # R
        i4, j4,
        # S
        # fmt: on
    )

    block_idx = sch.fuse(i0, j0)
    block_idy = sch.fuse(i1, j1)
    thread_idy = sch.fuse(j2, i2)
    sch.bind(block_idx, "blockIdx.x")
    sch.bind(block_idy, "blockIdx.y")
    sch.bind(thread_idy, "threadIdx.y")

    if isinstance(i_factors[2], int):
        num_ty = i_factors[2] * j_factors[2]
    else:
        num_ty = sch.get(i_factors[2]) * sch.get(j_factors[2])

    def fetch_to_shared(block, idx, ndim):
        block_read = sch.cache_read(block, idx, "shared")
        sch.compute_at(block_read, k0)
        vector_size = 8
        warp_size = 32
        fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
        f_0, f_1, f_2, f_3 = sch.split(
            fused, factors=[None, num_ty, warp_size, vector_size]
        )
        sch.bind(f_2, "threadIdx.x")
        sch.bind(f_1, "threadIdx.y")
        sch.vectorize(f_3)
        sch.storage_align(block_read, 0, axis=-2, factor=32, offset=8)

        return block_read

    A_sh = fetch_to_shared(block_outer, 0, 2)
    B_sh = fetch_to_shared(block_outer, 1, 2)

    # Step 3. Postproc-Rewrite-Tensorize
    # Step 3.1. Cache read
    loop = sch.get_loops(block_outer)[-1]

    A_warp = sch.cache_read(block_outer, 0, "warp")
    B_warp = sch.cache_read(block_outer, 1, "warp")

    sch.compute_at(A_warp, k1)
    sch.compute_at(B_warp, k1)

    # Step 3.2. Cache write
    C_warp = sch.cache_write(block_outer, 0, "warp")
    # block_outer, C_warp = C_warp, block_outer
    sch.reverse_compute_at(C_warp, thread_idy)
    # Wuwei: we also need spliting the write back stage.
    ii, jj = sch.get_loops(C_warp)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 8])
    sch.reorder(io, jo, ii, ji)
    # Step 3.3. Decompose
    block_init_c = sch.decompose_reduction(block_outer, sch.get_loops(block_outer)[3])

    def tile_wmma_fragment(block_read, height):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, height])
        j0, j1 = sch.split(j, factors=[None, 8])
        sch.reorder(i0, j0, i1, j1)
        return i1

    def lambda_a(i, j):
        i_0 = i // 16
        j_0 = j // 8

        i = i % 16
        j = j % 8
        return (
            i_0,
            j_0,
            (i % 8) * 4 + (j % 8) // 2,
            4 * (j // 8) + (i // 8) * 2 + (j % 8) % 2,
        )

    def lambda_b(i, j):
        i_0 = i // 8
        j_0 = j // 8
        i = i % 8
        j = j % 8
        return i_0, j_0, i // 2 + j * 4, i % 2

    loop_a = tile_wmma_fragment(A_warp, 16)
    loop_b = tile_wmma_fragment(B_warp, 8)

    sch.transform_layout(A_warp, 0, "write", index_map=lambda_a)
    sch.transform_layout(
        B_warp,
        0,
        "write",
        index_map=lambda_b,
    )
    sch.transform_layout(
        C_warp,
        0,
        "read",
        index_map=lambda_a,
    )

    use_ldmatrix = True

    if use_ldmatrix:
        sch.tensorize(loop_a, "mma.ldmatrix_a")
        sch.tensorize(loop_b, "mma.ldmatrix_b")
    else:
        warp_loop1, warp_loop2 = sch.get_loops(A_warp)[-2:]
        f_0, f_1 = sch.split(warp_loop1, factors=[None, 8])
        f_2, f_3 = sch.split(warp_loop2, factors=[None, 2])
        sch.reorder(f_1, f_2, f_0, f_3)
        fused_1 = sch.fuse(f_1, f_2)
        fused_2 = sch.fuse(f_0, f_3)
        sch.bind(fused_1, "threadIdx.x")

        warp_loop1, warp_loop2 = sch.get_loops(B_warp)[-2:]
        f_0, f_1 = sch.split(warp_loop1, factors=[4, 2])
        sch.reorder(warp_loop2, f_0, f_1)
        fused_1 = sch.fuse(warp_loop2, f_0)
        sch.bind(fused_1, "threadIdx.x")

    loop = sch.get_loops(block_inner)[-3]
    sch.tensorize(loop, "mma_sync")

    block_init_c = sch.get_block("C_init")
    init_loop1, init_loop2 = sch.get_loops(block_init_c)[-2:]
    f_0, f_1 = sch.split(init_loop1, factors=[None, 8])
    f_2, f_3 = sch.split(init_loop2, factors=[None, 2])
    sch.reorder(f_1, f_2, f_0, f_3)
    fused_1 = sch.fuse(f_1, f_2)
    fused_2 = sch.fuse(f_0, f_3)
    sch.bind(fused_1, "threadIdx.x")

    warp_loop1, warp_loop2 = sch.get_loops(C_warp)[-2:]
    f_0, f_1 = sch.split(warp_loop1, factors=[None, 8])
    f_2, f_3 = sch.split(warp_loop2, factors=[None, 2])
    sch.reorder(f_1, f_2, f_0, f_3)
    fused_1 = sch.fuse(f_1, f_2)
    fused_2 = sch.fuse(f_0, f_3)
    sch.bind(fused_1, "threadIdx.x")


ir_module = tvm.IRModule({"main": workload})
sch = tvm.tir.Schedule(ir_module)
schedule(sch)
print(sch.mod.script())

if tune:
    with tempfile.TemporaryDirectory() as work_dir:
        sch = ms.tune_tir(
            mod=workload,
            target=tvm.target.Target("nvidia/geforce-rtx-3070"),
            # use replay or evolutionary search
            config=ms.TuneConfig(
                strategy="evolutionary",
                num_trials_per_iter=32,
                max_trials_per_task=128,
                max_trials_global=128,
            ),
            work_dir=work_dir,
            space=ms.space_generator.ScheduleFn(schedule),
        )
        if sch is None:
            print("No valid schedule found!")
        else:
            print(sch.mod.script())
            print(sch.trace)
else:
    target = "cuda"
    f = tvm.build(sch.mod["main"], target=target, name="dense")
    print(f.imported_modules[0].get_source())

dev = tvm.device("cuda", 0)
a_np = np.random.uniform(size=(N, K)).astype("float16")
b_np = np.random.uniform(size=(M, K)).astype("float16")
c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)
f = tvm.build(sch.mod["main"], target="cuda", name="dense")

f(a, b, c)
tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)
print("ok")

evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
gflops = (N * M * K) * 2 / 1e9
time_ms = evaluator(a, b, c).mean * 1e3
print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))

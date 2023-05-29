import tempfile

import numpy as np

import tvm
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.tir import TensorIntrin, IntImm, Cast
from tvm import te, tir
from tvm.tir.tensor_intrin.cuda import (
    WMMA_LOAD_16x16x16_F16_A_INTRIN,
    WMMA_LOAD_16x16x16_F16_B_INTRIN,
    WMMA_SYNC_16x16x16_f16f16f32_INTRIN,
    WMMA_FILL_16x16x16_F32_INTRIN,
    WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN,
)


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
                [64, 1, 4, 1, 1],
                [1, 64, 1, 2, 2],
                [128, 2, 1],
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

            sch.storage_align(block_read, 0, axis=-2, factor=32, offset=8)
            return block_read

        fetch_to_shared(block_outer, 0, 2)
        fetch_to_shared(block_outer, 1, 2)

        loop = sch.get_loops(block_outer)[-1]
        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        store = sch.cache_write(block_outer, 0, "wmma.accumulator")
        sch.reverse_compute_at(store, thread_idy)

        ii, jj = sch.get_loops(store)[-2:]
        io, ii = sch.split(ii, factors=[None, 16])
        jo, ji = sch.split(jj, factors=[None, 16])
        sch.reorder(io, jo, ii, ji)

        loop = sch.get_loops(block_outer)[3]
        block_init_c = sch.decompose_reduction(block_outer, loop)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        i, j = sch.get_loops(A_mat)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)
        sch.unroll(i0)
        sch.unroll(j0)
        sch.tensorize(i1, WMMA_LOAD_16x16x16_F16_A_INTRIN)

        i, j = sch.get_loops(B_mat)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)
        sch.unroll(i0)
        sch.unroll(j0)
        sch.tensorize(i1, WMMA_LOAD_16x16x16_F16_B_INTRIN)

        sch.tensorize(sch.get_loops(block_init_c_inner)[-2], WMMA_FILL_16x16x16_F32_INTRIN)
        sch.tensorize(sch.get_loops(store)[-2], WMMA_STORE_16x16x16_F32_GLOBAL_INTRIN)
        sch.tensorize(sch.get_loops(block_inner)[-3], WMMA_SYNC_16x16x16_f16f16f32_INTRIN)

    return schedule


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

tune = True
M, N, K = 4096, 4096, 4096

target = tvm.target.Target("vulkan -from_device=0")
# target = tvm.target.Target("nvidia/geforce-rtx-3070")

workload = get_matmul(M, N, K, out_dtype)

if tune:
    with tempfile.TemporaryDirectory() as work_dir:
        db = ms.tune_tir(
            mod=workload,
            target=tvm.target.Target(target),
            max_trials_global=128,
            work_dir=work_dir,
            # space=ms.space_generator.ScheduleFn(get_schedule_fun(tune)),
        )
        sch = ms.tir_integration.compile_tir(db, workload, target)
        print(sch.trace)
else:
    sch = tir.Schedule(workload)
    get_schedule_fun(tune)(sch)

# print(sch.mod)
f = tvm.build(sch.mod, target=target)
dev = tvm.device(target.kind.name, 0)

A = tvm.nd.array(np.random.randn(M, K).astype("float16"), dev)
B = tvm.nd.array(np.random.randn(K, N).astype("float16"), dev)
C = tvm.nd.array(np.random.randn(M, N).astype(out_dtype), dev)

f(A, B, C)

evaluator = f.time_evaluator(f.entry_name, dev, number=100)
gflops = (N * M * K) * 2 / 1e9
time_ms = evaluator(A, B, C).mean * 1e3
print("%f GFLOPS" % (gflops / (time_ms / 1e3)))

out = C.numpy()

A_np = A.numpy()
B_np = B.numpy()
ref = np.dot(A_np.astype("float32"), B_np.astype("float32"))

print(np.max(np.abs(out - ref)), np.mean(np.abs(out - ref)))

import tvm
from tvm import te, tir
import tvm.testing
import numpy as np
import tvm.topi.testing
from tvm.script import tir as T
from tvm.tir import TensorIntrin


pytest_plugins = [
    "tvm.contrib.hexagon.pytest_plugin",
]

@T.prim_func
def dot_product_32x4_u8u8i32_desc(
    A: T.Buffer((4,), "uint8", offset_factor=1),
    B: T.Buffer((32, 4), "uint8", offset_factor=1),
    C: T.Buffer((32,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:32], A[0:4], B[0:32, 0:4])
        T.writes(C[0:32])
        for i in T.serial(0, 32):
            with T.init():
                C[i] = T.int32(0)
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")


@T.prim_func
def dot_product_32x4_u8u8i32_vrmpy(
    A: T.Buffer((4,), "uint8", offset_factor=1),
    B: T.Buffer((32, 4), "uint8", offset_factor=1),
    C: T.Buffer((32,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:32], A[0:4], B[0:32, 0:4])
        T.writes(C[0:32])

        A_u8x4 = A.vload([0], "uint8x4")
        A_i32 = T.reinterpret(A_u8x4, dtype="int32")

        B_i8x128 = B.vload([0, 0], dtype="uint8x128")
        B_i32x32 = T.reinterpret(B_i8x128, dtype="int32x32")

        C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
            T.uint32(3),
            C[T.ramp(T.int32(0), 1, 32)],
            B_i32x32,
            A_i32,
            dtype="int32x32",
        )


VRMPY_INTRIN = "dot_32x4_vrmpy"

TensorIntrin.register(
    VRMPY_INTRIN, dot_product_32x4_u8u8i32_desc, dot_product_32x4_u8u8i32_vrmpy
)

def matmul(n: int, m: int, k: int):
    X = te.placeholder((m, k), name="X", dtype="uint8")
    packedW = te.placeholder((n // 32, k // 4, 32, 4), name="packedW", dtype="uint8")

    ak = te.reduce_axis((0, k), name="k")
    out = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packedW[
                tvm.tir.indexdiv(j, 32), tvm.tir.indexdiv(ak, 4), j % 32, ak % 4
            ].astype("int32"),
            axis=ak,
        ),
        name="compute",
    )
    return [X, packedW, out]


def schedule_matmul_common(sch, block, batched, M):
    a_y, a_x, _ = sch.get_loops(block)[-3:]
    outer_block = block

    a_yo, a_yi = sch.split(a_y, factors=[None, min(M, 32)])

    a_xo, a_xi = sch.split(a_x, factors=[None, 32])
    sch.reorder(a_yo, a_xo, a_yi, a_xi)

    a_xi, a_k = sch.get_loops(block)[-2:]
    a_ko, a_ki = sch.split(a_k, factors=[None, 4])
    sch.reorder(a_ko, a_xi, a_ki)

    if batched:
        a_b = sch.get_loops(outer_block)[0]
        fused = sch.fuse(a_b, a_yo, a_xo)
    else:
        fused = sch.fuse(a_yo, a_xo)

    # sch.parallel(fused)

    dec = sch.decompose_reduction(block, a_ko)

    init_loop = sch.get_loops(dec)[-1]
    sch.vectorize(init_loop)

    sch.tensorize(a_xi, VRMPY_INTRIN)
    # print(sch.mod.script())

    return fused


def schedule_dense(dense_block, M, sch: tir.Schedule):
    schedule_matmul_common(sch, dense_block, False,  M)


bert_workloads = [(128, 768, 768), (128, 3072, 768), (128, 768, 3072)]


@tvm.testing.requires_hexagon
def test_vrmpy_dense(hexagon_session):
    target_hexagon = tvm.target.hexagon("v68", link_params=True)

    for M, N, K in bert_workloads:
        workload = te.create_prim_func(matmul(n=N, m=M, k=K))

        ir_module = tvm.IRModule({"main": workload})
        sch = tvm.tir.Schedule(ir_module)
        block = sch.get_block("compute")
        schedule_dense(block, M, sch)

        f = tvm.build(sch.mod["main"], target=tvm.target.Target(target_hexagon, host=target_hexagon), name="dense")

        module = hexagon_session.load_module(f)

        a_np = np.random.uniform(1, 10, size=(M, K)).astype("uint8")
        b_np = np.random.uniform(1, 10, size=(N, K)).astype("uint8")
        c_np = np.dot(a_np.astype("int32"), b_np.transpose().astype("int32"))

        packW = np.random.uniform(1, 10, size=(N // 32, (K // 4), 32, 4)).astype("uint8")

        for r_idx in range(N // 32):
            for ko in range(K // 4):
                for s_idx in range(32):
                    for t_idx in range(4):
                        packW[r_idx][ko][s_idx][t_idx] = b_np[r_idx * 32 + s_idx][
                            ko * 4 + t_idx
                        ]

        dev = hexagon_session.device
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(packW, dev)
        c = tvm.nd.array(np.zeros((M, N), dtype="int32"), dev)

        # print(f.imported_modules[0].get_source())
        module(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)
        evaluator = module.time_evaluator(module.entry_name, hexagon_session.device, number=20)
        time_ms = evaluator(a, b, c).mean * 1e3
        gflops = (N * M * K) * 2 / 1e9
        print("time elapsed: ", time_ms)
        print("GOPS:", gflops / (time_ms / 1e3))

        break

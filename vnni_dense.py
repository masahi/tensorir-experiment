import tvm
from tvm.script import tir as T
from tvm import te, tir
import tvm.testing
import numpy as np
from tvm.script.registry import register


@register
def int32x16(imm, span):
    return imm.astype("int32x16", span)


def matmul(n: int, m: int, k: int):
    X = te.placeholder((m, k), name="X", dtype="uint8")
    packedW = te.placeholder((n // 16, k // 4, 16, 4), name="packedW", dtype="int8")

    ak = te.reduce_axis((0, k), name="k")
    out = te.compute(
        (m, n),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packedW[tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4),  j % 16, ak % 4].astype("int32"),
            axis=ak,
        ),
        name="C",
    )
    return [X, packedW, out]



@T.prim_func
def dot_product_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "uint8")
    B = T.match_buffer(b, (16, 4), "int8")
    C = T.match_buffer(c, (16,), "int32")

    with T.block("root"):
        T.reads(C[0:16], A[0 : 4], B[0 : 16, 0 : 4])
        T.writes(C[0:16])
        for i in range(0, 16):
            with T.init():
                C[i] = T.int32(0)
            for k in range(0, 4):
                with T.block("update"):
                    vi = T.axis.remap("R", [k])
                    C[i] = C[i] + T.cast(A[vi], "int32") * T.cast(B[i, vi], "int32")


vnni_inst_name = "llvm.x86.avx512.vpdpbusd.512"
llvm_id = tvm.target.codegen.llvm_lookup_intrinsic_id(vnni_inst_name)

@T.prim_func
def dot_product_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "uint8")
    B = T.match_buffer(b, (16, 4), "int8")
    C = T.match_buffer(c, (16,), "int32")

    with T.block("root"):
        T.reads(C[0:16], A[0 : 4], B[0 : 16, 0 : 4])
        T.writes(C[0:16])

        T.evaluate(
            T.call_llvm_pure_intrin(
            T.int32(9785), # cannot use the variable llvm_id
            T.uint32(0),
            T.int32x16(0),
            T.broadcast(T.reinterpret(A.vload([0], dtype="uint8x4"), dtype="int32"), 16),
            T.reinterpret(B.vload([0, 0], dtype="int8x64"), dtype="int32x16"),
            dtype="int32x16")
        )


N = 512
M = 512
K = 512

# workload = matmul(n=N, m=M, k=K)
# workload = te.create_prim_func(workload)
# ir_module = tvm.IRModule({"main": dot_product_intrin})
# print(ir_module)
# ir_module = tvm.IRModule({"main": workload})
# print(ir_module.script())


tir.TensorIntrin.register("dot_16x1x16_uint8_int8_int32_cascadelake", dot_product_desc, dot_product_intrin)


def test_integration_matmul():
    N = 512
    M = 512
    K = 512
    workload = matmul(n=N, m=M, k=K)
    workload = te.create_prim_func(workload)

    def schedule(sch: tir.Schedule):
        block = sch.get_block("C")
        a_y, a_x, a_k = sch.get_loops(block)
        a_yo, a_yi = sch.split(a_y, factors=[None, 32])
        a_xo, a_xi = sch.split(a_x, factors=[None, 16])
        a_ko, a_ki = sch.split(a_k, factors=[None, 4])
        sch.reorder(a_yo, a_xo, a_yi, a_ko, a_xi, a_ki)
        sch.tensorize(a_xi, "dot_16x1x16_uint8_int8_int32_cascadelake")
        fused = sch.fuse(a_yo, a_xo)
        sch.parallel(fused)

    ir_module = tvm.IRModule({"main": workload})
    sch = tvm.tir.Schedule(ir_module)
    schedule(sch)

    print(sch.mod.script())

    target = "llvm -mcpu=cascadelake"
    dev = tvm.device(target, 0)
    a_np = np.random.uniform(1, 10, size=(N, K)).astype("uint8")
    b_np = np.random.uniform(1, 10, size=(M, K)).astype("int8")
    c_np = np.dot(a_np.astype("int32"), b_np.transpose().astype("int32"))

    packW = np.random.uniform(1, 10, size=(N // 16, (K // 4), 16, 4)).astype("int8")

    for r_idx in range(N // 16):
        for ko in range(K // 4):
            for s_idx in range(16):
                for t_idx in range(4):
                    packW[r_idx][ko][s_idx][t_idx] = b_np[r_idx * 16 + s_idx][ko * 4 + t_idx]


    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(packW, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype="int32"), dev)
    f = tvm.build(sch.mod['main'], target=target, name="dense")

    # print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=10)
    gflops = (N*M*K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


if __name__ == "__main__":
    test_integration_matmul()

import tvm
from tvm.script import tir as T
from tvm import te, tir, relay
import tvm.testing
import numpy as np
from tvm.script.registry import register
from tvm.meta_schedule.tune import tune_relay, extract_task_from_relay


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
            * packedW[
                tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4), j % 16, ak % 4
            ].astype("int32"),
            axis=ak,
        ),
        name="C",
    )
    return [X, packedW, out]


@T.prim_func
def dot_product_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "uint8", offset_factor=1)
    B = T.match_buffer(b, (16, 4), "int8", offset_factor=1)
    C = T.match_buffer(c, (16,), "int32", offset_factor=1)

    with T.block("root"):
        T.reads(C[0:16], A[0:4], B[0:16, 0:4])
        T.writes(C[0:16])
        for i in T.serial(0, 16):
            with T.init():
                C[i] = T.int32(0)
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")


vnni_inst_name = "llvm.x86.avx512.vpdpbusd.512"
llvm_id = tvm.target.codegen.llvm_lookup_intrinsic_id(vnni_inst_name)


@T.prim_func
def dot_product_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "uint8", offset_factor=1)
    B = T.match_buffer(b, (16, 4), "int8", offset_factor=1)
    C = T.match_buffer(c, (16,), "int32", offset_factor=1)

    B_vec = T.buffer_decl(
        data=B.data, dtype=B.dtype, strides=B.strides, shape=(64,), offset_factor=1, elem_offset=B.elem_offset, name="B_vec"
    )

    with T.block("root"):
        T.reads(C[0:16], A[0:4], B[0:16, 0:4])
        T.writes(C[0:16])

        C[T.ramp(T.int32(0), 1, 16)] += T.call_llvm_pure_intrin( # Note: this is an update +=
            T.int32(9785),  # cannot use the variable llvm_id
            T.uint32(0),
            T.int32x16(0),
            T.broadcast(T.reinterpret(A[T.ramp(T.int32(0), 1, 4)], dtype="int32"), 16),
            T.reinterpret(B_vec[T.ramp(T.int32(0), 1, 64)], dtype="int32x16"),
            dtype="int32x16",
        )


N = 512
M = 512
K = 512

# workload = matmul(n=N, m=M, k=K)
# workload = te.create_prim_func(workload)
ir_module = tvm.IRModule({"main": dot_product_intrin})
print(ir_module)
# ir_module = tvm.IRModule({"main": workload})
# print(ir_module.script())


tir.TensorIntrin.register(
    "dot_16x1x16_uint8_int8_int32_cascadelake", dot_product_desc, dot_product_intrin
)


def schedule(sch: tir.Schedule, top_block_name="C"):
    block = sch.get_block(top_block_name)
    a_y, a_x, a_k = sch.get_loops(block)
    a_yo, a_yi = sch.split(a_y, factors=[None, 32])
    a_xo, a_xi = sch.split(a_x, factors=[None, 16])
    a_ko, a_ki = sch.split(a_k, factors=[None, 4])
    sch.reorder(a_yo, a_xo, a_yi, a_ko, a_xi, a_ki)
    fused = sch.fuse(a_yo, a_xo)
    sch.parallel(fused)
    dec = sch.decompose_reduction(block, a_ko)

    init_loop = sch.get_loops(dec)[-1]
    sch.vectorize(init_loop)

    # sch.tensorize(a_xi, "dot_16x1x16_uint8_int8_int32_cascadelake")

    print(sch.mod.script())


def test_integration_matmul():
    N = 128
    M = 32
    K = 96
    workload = matmul(n=N, m=M, k=K)
    workload = te.create_prim_func(workload)

    ir_module = tvm.IRModule({"main": workload})
    sch = tvm.tir.Schedule(ir_module)
    schedule(sch)
    # return

    # print(sch.mod.script())

    target = "llvm -mcpu=cascadelake"
    f = tvm.build(sch.mod["main"], target=target, name="dense")
    # return
    dev = tvm.device(target, 0)
    a_np = np.random.uniform(1, 10, size=(M, K)).astype("uint8")
    b_np = np.random.uniform(1, 10, size=(N, K)).astype("int8")
    c_np = np.dot(a_np.astype("int32"), b_np.transpose().astype("int32"))

    packW = np.random.uniform(1, 10, size=(N // 16, (K // 4), 16, 4)).astype("int8")

    for r_idx in range(N // 16):
        for ko in range(K // 4):
            for s_idx in range(16):
                for t_idx in range(4):
                    packW[r_idx][ko][s_idx][t_idx] = b_np[r_idx * 16 + s_idx][
                        ko * 4 + t_idx
                    ]

    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(packW, dev)
    c = tvm.nd.array(np.zeros((M, N), dtype="int32"), dev)

    # print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=10)
    gflops = (N * M * K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print(
        "matmul with tensor core: %f ms, %f GFLOPS"
        % (time_ms, gflops / (time_ms / 1e3))
    )


def tune_dense_vnni():
    data_shape = (32, 96)
    weight_shape = (128, 96)

    data_dtype = "uint8"
    data = relay.var("data", shape=data_shape, dtype=data_dtype)
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
    dense = relay.nn.dense(data, weight, out_dtype="int32")
    # out = relay.nn.bias_add(dense, bias)
    out = dense
    mod = tvm.IRModule.from_expr(out)

    target = "llvm -mcpu=cascadelake"

    weight_np = np.random.uniform(1, 10, size=weight_shape).astype("int8")

    params = {"weight": weight_np}

    extracted_tasks = extract_task_from_relay(mod, target, params)

    for task in extracted_tasks:
        print(task.mod)
        mod = task.dispatched[0]
        mod = mod.with_attr("global_symbol", "main")
        mod = mod.with_attr("tir.noalias", True)
        mod = tvm.IRModule({"main": mod})
        sch = tvm.tir.Schedule(mod)
        schedule(sch, top_block_name="compute")
        print(sch.mod.script())


if __name__ == "__main__":
    test_integration_matmul()
#     # tune_dense_vnni()

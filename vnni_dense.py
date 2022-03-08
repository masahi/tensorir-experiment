import tvm
from tvm.script import tir as T
from tvm import te, tir, relay
import tvm.testing
import numpy as np
from tvm.script.registry import register
from tvm.meta_schedule.tune import extract_task_from_relay
from tvm import meta_schedule as ms
import tempfile
from tvm.topi.transform import layout_transform
import tvm.topi.testing


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


def batch_matmul(batch, n: int, m: int, k: int):
    x = te.placeholder((batch, m, k), name="X", dtype="uint8")
    y = te.placeholder((batch, n, k), name="Y", dtype="int8")
    packed_y_layout = "BNK16n4k"
    packed_y = layout_transform(y, "BNK", packed_y_layout)

    _, n_o, _, n_i, _ = packed_y.shape
    ak = te.reduce_axis((0, k), name="k")

    z = te.compute(
        (batch, m, n_o * n_i),
        lambda b, i, j: te.sum(
            x[b, i, ak].astype("int32")
            * packed_y[b, tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4), j % 16, ak % 4].astype(
                "int32"
            ),
            axis=ak,
        ),
        tag="batch_matmul_vnni",
    )

    return [x, y, z]


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

    with T.block("root"):
        T.reads(C[0:16], A[0:4], B[0:16, 0:4])
        T.writes(C[0:16])

        C[T.ramp(T.int32(0), 1, 16)] += T.call_llvm_pure_intrin( # Note: this is an update +=
            T.int32(9785),  # cannot use the variable llvm_id
            T.uint32(0),
            T.int32x16(0),
            T.broadcast(T.reinterpret(A.vload([0], "uint8x4"), dtype="int32"), 16),
            T.reinterpret(B.vload([0, 0], dtype="int8x64"), dtype="int32x16"),
            dtype="int32x16",
        )


tir.TensorIntrin.register(
    "dot_16x1x16_uint8_int8_int32_cascadelake", dot_product_desc, dot_product_intrin
)

def schedule_matmul_common(sch, block, a_y, a_x, a_k, do_tune, M):
    if do_tune:
        y_factors = sch.sample_perfect_tile(a_y, n=2)
        a_yo, a_yi = sch.split(a_y, factors=y_factors)
    else:
        a_yo, a_yi = sch.split(a_y, factors=[None, min(M, 32)])

    a_xo, a_xi = sch.split(a_x, factors=[None, 16])
    a_ko, a_ki = sch.split(a_k, factors=[None, 4])
    sch.reorder(a_yo, a_xo, a_yi, a_ko, a_xi, a_ki)
    fused = sch.fuse(a_yo, a_xo)
    dec = sch.decompose_reduction(block, a_ko)

    init_loop = sch.get_loops(dec)[-1]
    sch.vectorize(init_loop)

    sch.tensorize(a_xi, "dot_16x1x16_uint8_int8_int32_cascadelake")

    return fused

def schedule(M, do_tune, sch: tir.Schedule):
    block = sch.get_block("C")
    a_y, a_x, a_k = sch.get_loops(block)
    outer_loop = schedule_matmul_common(sch, block, a_y, a_x, a_k, do_tune, M)

    sch.parallel(outer_loop)


def schedule_for_tune(sch: tir.Schedule):
    return schedule(None, False, sch)


def schedule_batch_matmul(M, sch):
    layout_trans_block = sch.get_block("T_layout_trans")
    bmm_block = sch.get_block("compute")
    print(sch.get(bmm_block))

    a_b, a_y, a_x, a_k = sch.get_loops(bmm_block)
    gemm_outer_loop = schedule_matmul_common(sch, bmm_block, a_y, a_x, a_k, False, M)

    fused = sch.fuse(a_b, gemm_outer_loop)
    sch.parallel(fused)


fbgemm_workloads = [
    (64, 800, 320),
    (64, 768, 512),
    (16, 256, 512),
    (128, 128, 128),
    (256, 512, 256),
    (1024, 1024, 1024),
]

bert_workloads = [(128, 768, 3072), (128, 768, 768), (128, 3072, 768)]


def test_vnni_dense():
    do_tune = False
    target = "llvm -mcpu=cascadelake --num-cores=4"

    for M, N, K in fbgemm_workloads + bert_workloads:
        workload = te.create_prim_func(matmul(n=N, m=M, k=K))

        if not do_tune:
            ir_module = tvm.IRModule({"main": workload})
            sch = tvm.tir.Schedule(ir_module)
            schedule(M, do_tune, sch)
        else:
            with tempfile.TemporaryDirectory() as work_dir:
                sch = ms.tune_tir(
                    mod=workload,
                    target=target,
                    # use replay or evolutionary search
                    config=ms.ReplayTraceConfig(
                        num_trials_per_iter=32,
                        num_trials_total=32,
                    ),
                    work_dir=work_dir,
                    space=ms.space_generator.ScheduleFn(schedule_for_tune)
                    )
                if sch is None:
                    print("No valid schedule found!")
                else:
                    # print(sch.mod.script())
                    # print(sch.trace)
                    pass

        f = tvm.build(sch.mod["main"], target=target, name="dense")
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
            "matmul with VNNI TIR tensorization: %f ms, %f GFLOPS"
            % (time_ms, gflops / (time_ms / 1e3))
        )


def test_vnni_batch_matmul():

    workloads = []
    for m, n, k in fbgemm_workloads + bert_workloads:
        batch = 8
        workloads.append((batch, m, n, k))

    seq_len = 128
    bert_bmm_workloads = [(16, seq_len, seq_len, 64), (16, seq_len, 64, seq_len)]

    do_tune = False
    target = "llvm -mcpu=cascadelake --num-cores=4"

    for batch, M, N, K in bert_bmm_workloads:
        workload = te.create_prim_func(batch_matmul(batch, n=N, m=M, k=K))

        ir_module = tvm.IRModule({"main": workload})
        sch = tvm.tir.Schedule(ir_module)
        schedule_batch_matmul(M, sch)

        print(sch.mod.script())

        f = tvm.build(sch.mod["main"], target=target, name="dense")
        dev = tvm.device(target, 0)
        a_np = np.random.uniform(1, 10, size=(batch, M, K)).astype("uint8")
        b_np = np.random.uniform(1, 10, size=(batch, N, K)).astype("int8")

        c_np = tvm.topi.testing.batch_matmul(a_np, b_np, out_dtype="int32")

        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros((batch, M, N), dtype="int32"), dev)

        # print(f.imported_modules[0].get_source())
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)
        print("ok")
        break

        # evaluator = f.time_evaluator(f.entry_name, dev, number=10)
        # gflops = (N * M * K) * 2 / 1e9
        # time_ms = evaluator(a, b, c).mean * 1e3
        # print(
        #     "matmul with VNNI TIR tensorization: %f ms, %f GFLOPS"
        #     % (time_ms, gflops / (time_ms / 1e3))
        # )


def vnni_relay():
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
    test_vnni_batch_matmul()
    # test_vnni_dense()
#     # tune_dense_vnni()

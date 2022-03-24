from tvm import tir
from tvm.script import tir as T
from tvm.script.registry import register


@register
def int32x4(imm, span):
    return imm.astype("int32x4", span)


@register
def int8x4(imm, span):
    return imm.astype("int8x4", span)


@T.prim_func
def dot_product_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "int8", offset_factor=1)
    B = T.match_buffer(b, (4, 4), "int8", offset_factor=1)
    C = T.match_buffer(c, (4,), "int32", offset_factor=1)

    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])
        for i in T.serial(0, 4):
            with T.init():
                C[i] = T.int32(0)
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")


@T.prim_func
def dot_int8_int8_int32_neon(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "int8", offset_factor=1)
    B = T.match_buffer(b, (4, 4), "int8", offset_factor=1)
    C = T.match_buffer(c, (4,), "int32", offset_factor=1)

    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])

        A_int8 = A.vload([0], "int8x4")
        re_int32 = T.reinterpret(A_int8, dtype="int32")
        vec_ai32 = T.broadcast(re_int32, 2)
        vec_a = T.reinterpret(vec_ai32, dtype="int8x8")

        vec_b = B.vload([0, 0], dtype="int8x8")

        multiply = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            T.uint32(2),
            vec_a,
            vec_b,
            dtype="int16x8",
        )

        pair1 = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            T.uint32(1),
            multiply,
            dtype="int32x4",
        )

        vec_b_2 = B.vload([2, 0], dtype="int8x8")

        multiply_2 = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            T.uint32(2),
            vec_a,
            vec_b_2,
            dtype="int16x8",
        )

        pair2 = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            T.uint32(1),
            multiply_2,
            dtype="int32x4",
        )

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.addp.v4i32"),
            T.uint32(2),
            pair1,
            pair2,
            dtype="int32x4",
        )


@T.prim_func
def dot_int8_int8_int32_neon_82(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "int8", offset_factor=1)
    B = T.match_buffer(b, (4, 4), "int8", offset_factor=1)
    C = T.match_buffer(c, (4,), "int32", offset_factor=1)

    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])

        A_i8x4 = A.vload([0], "int8x4")
        A_i32 = T.reinterpret(A_i8x4, dtype="int32")
        vec_ai32 = T.broadcast(A_i32, 4)
        vec_a = T.reinterpret(vec_ai32, dtype="int8x16")

        vec_b = B.vload([0, 0], dtype="int8x16")

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.sdot.v4i32.v16i8"),
            T.uint32(3),
            T.int32x4(0),
            vec_a,
            vec_b,
            dtype="int32x4",
        )


tir.TensorIntrin.register(
    "dot_int8_int8_int32_neon", dot_product_desc, dot_int8_int8_int32_neon
)

tir.TensorIntrin.register(
    "dot_int8_int8_int32_neon_82", dot_product_desc, dot_int8_int8_int32_neon_82
)

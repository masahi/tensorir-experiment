from tvm import tir
from tvm.script import tir as T
from tvm.script.registry import register


@register
def int32x16(imm, span):
    return imm.astype("int32x16", span)


@register
def int8x4(imm, span):
    return imm.astype("int8x4", span)


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


@T.prim_func
def dot_product_intrin(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (4,), "uint8", offset_factor=1)
    B = T.match_buffer(b, (16, 4), "int8", offset_factor=1)
    C = T.match_buffer(c, (16,), "int32", offset_factor=1)

    with T.block("root"):
        T.reads(C[0:16], A[0:4], B[0:16, 0:4])
        T.writes(C[0:16])

        A_u8x4 = A.vload([0], "uint8x4")
        A_i32 = T.reinterpret(A_u8x4, dtype="int32")

        B_i8x64 = B.vload([0, 0], dtype="int8x64")
        B_i32x16 = T.reinterpret(B_i8x64, dtype="int32x16")

        C[
            T.ramp(T.int32(0), 1, 16)
        ] += T.call_llvm_pure_intrin(  # Note: this is an update +=
            T.llvm_lookup_intrinsic_id("llvm.x86.avx512.vpdpbusd.512"),
            T.uint32(0),
            T.int32x16(0),
            T.broadcast(A_i32, 16),
            B_i32x16,
            dtype="int32x16",
        )


tir.TensorIntrin.register(
    "dot_16x1x16_uint8_int8_int32_cascadelake", dot_product_desc, dot_product_intrin
)

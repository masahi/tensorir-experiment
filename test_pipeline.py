import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.te import create_prim_func
from tvm.tir import Schedule
from tvm.script import tir as T
from tvm import tir


@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(4096, 4096), "float16"], B: T.Buffer[(4096, 4096), "float16"], C: T.Buffer[(4096, 4096), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.tensor_core_enabled":"1", "warp_execution":1})
            C_reindex_wmma_accumulator = T.alloc_buffer([4096, 4096], dtype="float32", scope="wmma.accumulator")
            A_reindex_shared = T.alloc_buffer([4096, 4096], dtype="float16", scope="shared.dyn")
            B_reindex_shared = T.alloc_buffer([4096, 4096], dtype="float16", scope="shared.dyn")
            A_reindex_shared_wmma_matrix_a = T.alloc_buffer([4096, 4096], dtype="float16", scope="wmma.matrix_a")
            B_reindex_shared_wmma_matrix_b = T.alloc_buffer([4096, 4096], dtype="float16", scope="wmma.matrix_b")
            for ax_0_0_ax_0_0_fused in T.thread_binding(512, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
                for ax_0_1_ax_0_1_fused in T.thread_binding(8, thread="blockIdx.y"):
                    for ax_0_2_ax_0_2_fused in T.thread_binding(4, thread="threadIdx.y"):
                        for ax_0_4_init, ax_0_4_init_1 in T.grid(2, 2):
                            with T.block("update_o_init"):
                                bv_o = T.axis.spatial(256, ax_0_0_ax_0_0_fused // 8 * 4 + ax_0_2_ax_0_2_fused // 2 * 2 + ax_0_4_init)
                                bv_o_1 = T.axis.spatial(256, ax_0_0_ax_0_0_fused % 8 * 32 + ax_0_1_ax_0_1_fused * 4 + ax_0_2_ax_0_2_fused % 2 * 2 + ax_0_4_init_1)
                                T.reads()
                                T.writes(C_reindex_wmma_accumulator[bv_o * 16 : bv_o * 16 + 16, bv_o_1 * 16 : bv_o_1 * 16 + 16])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32})
                                for ax_1_0, ax_1_0_1 in T.grid(1, 1):
                                    with T.block("update_init_o"):
                                        bv_init_o = T.axis.spatial(1, 0)
                                        bv_init_o_1 = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_wmma_accumulator[bv_o * 16 : bv_o * 16 + 16, bv_o_1 * 16 : bv_o_1 * 16 + 16])
                                        C_1 = T.match_buffer(C_reindex_wmma_accumulator[bv_o * 16 : bv_o * 16 + 16, bv_o_1 * 16 : bv_o_1 * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                                        T.evaluate(T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // 256 + C_1.elem_offset % 256 // 16, T.float32(0), dtype="handle"))
                        for ax_0_0 in T.serial(64, annotations={"software_pipeline_order":[0, 1, 2], "software_pipeline_stage":[0, 0, 3]}):
                            with T.block("A_reindex_shared"):
                                v0, v1 = T.axis.remap("SS", [ax_0_0_ax_0_0_fused, ax_0_0])
                                T.reads(A[v0 // 8 * 64 : v0 // 8 * 64 + 64, v1 * 64 : v1 * 64 + 64])
                                T.writes(A_reindex_shared[v0 // 8 * 64 : v0 // 8 * 64 + 64, v1 * 64 : v1 * 64 + 64])
                                T.block_attr({"auto_copy":1, "double_buffer_scope":0, "local_stage":0, "vector_bytes":16})
                                for ax0, ax1 in T.grid(64, 64):
                                    A_reindex_shared[v0 // 8 * 64 + ax0, v1 * 64 + ax1] = A[v0 // 8 * 64 + ax0, v1 * 64 + ax1]
                            with T.block("B_reindex_shared"):
                                v0, v1, v2 = T.axis.remap("SSS", [ax_0_0_ax_0_0_fused, ax_0_1_ax_0_1_fused, ax_0_0])
                                T.reads(B[v0 % 8 * 512 + v1 * 64 : v0 % 8 * 512 + v1 * 64 + 64, v2 * 64 : v2 * 64 + 64])
                                T.writes(B_reindex_shared[v0 % 8 * 512 + v1 * 64 : v0 % 8 * 512 + v1 * 64 + 64, v2 * 64 : v2 * 64 + 64])
                                T.block_attr({"auto_copy":1, "double_buffer_scope":0, "local_stage":0, "vector_bytes":16})
                                for ax0, ax1 in T.grid(64, 64):
                                    B_reindex_shared[v0 % 8 * 512 + v1 * 64 + ax0, v2 * 64 + ax1] = B[v0 % 8 * 512 + v1 * 64 + ax0, v2 * 64 + ax1]
                            for ax_0_1 in T.serial(2):
                                with T.block("A_reindex_shared_wmma.matrix_a"):
                                    v0, v1, v2, v3 = T.axis.remap("SSSS", [ax_0_0_ax_0_0_fused, ax_0_2_ax_0_2_fused, ax_0_0, ax_0_1])
                                    T.reads(A_reindex_shared[v0 // 8 * 64 + v1 // 2 * 32 : v0 // 8 * 64 + v1 // 2 * 32 + 32, v2 * 64 + v3 * 32 : v2 * 64 + v3 * 32 + 32])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0 // 8 * 64 + v1 // 2 * 32 : v0 // 8 * 64 + v1 // 2 * 32 + 32, v2 * 64 + v3 * 32 : v2 * 64 + v3 * 32 + 32])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(32, 32):
                                        A_reindex_shared_wmma_matrix_a[v0 // 8 * 64 + v1 // 2 * 32 + ax0, v2 * 64 + v3 * 32 + ax1] = A_reindex_shared[v0 // 8 * 64 + v1 // 2 * 32 + ax0, v2 * 64 + v3 * 32 + ax1]
                                with T.block("B_reindex_shared_wmma.matrix_b"):
                                    v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [ax_0_0_ax_0_0_fused, ax_0_1_ax_0_1_fused, ax_0_2_ax_0_2_fused, ax_0_0, ax_0_1])
                                    T.reads(B_reindex_shared[v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 : v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 + 32, v3 * 64 + v4 * 32 : v3 * 64 + v4 * 32 + 32])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 : v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 + 32, v3 * 64 + v4 * 32 : v3 * 64 + v4 * 32 + 32])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(32, 32):
                                        B_reindex_shared_wmma_matrix_b[v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 + ax0, v3 * 64 + v4 * 32 + ax1] = B_reindex_shared[v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 + ax0, v3 * 64 + v4 * 32 + ax1]
                                for ax_0_3, ax_0_3_1, ax_0_2, ax_0_4, ax_0_4_1 in T.grid(1, 1, 2, 2, 2):
                                    with T.block("update_o_update"):
                                        bv_o_2 = T.axis.spatial(256, ax_0_0_ax_0_0_fused // 8 * 4 + ax_0_2_ax_0_2_fused // 2 * 2 + ax_0_4)
                                        bv_o_3 = T.axis.spatial(256, ax_0_0_ax_0_0_fused % 8 * 32 + ax_0_1_ax_0_1_fused * 4 + ax_0_2_ax_0_2_fused % 2 * 2 + ax_0_4_1)
                                        bv_o_4 = T.axis.reduce(256, ax_0_0 * 4 + ax_0_1 * 2 + ax_0_2)
                                        T.reads(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16], A_reindex_shared_wmma_matrix_a[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16], B_reindex_shared_wmma_matrix_b[bv_o_3 * 16 : bv_o_3 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16])
                                        T.writes(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16])
                                        T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32})
                                        for ax_1_0, ax_1_0_2, ax_1_0_3 in T.grid(1, 1, 1):
                                            with T.block("update_o"):
                                                bv_o_5 = T.axis.spatial(1, 0)
                                                bv_o_6 = T.axis.spatial(1, 0)
                                                bv_o_7 = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16], A_reindex_shared_wmma_matrix_a[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16], B_reindex_shared_wmma_matrix_b[bv_o_3 * 16 : bv_o_3 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16])
                                                T.writes(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16])
                                                A_1 = T.match_buffer(A_reindex_shared_wmma_matrix_a[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                                B_1 = T.match_buffer(B_reindex_shared_wmma_matrix_b[bv_o_3 * 16 : bv_o_3 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_b", offset_factor=16)
                                                C_2 = T.match_buffer(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                                                T.evaluate(T.tvm_mma_sync(C_2.data, C_2.elem_offset // 256 + C_2.elem_offset % 256 // 16, A_1.data, A_1.elem_offset // 256, B_1.data, B_1.elem_offset // 256, C_2.data, C_2.elem_offset // 256 + C_2.elem_offset % 256 // 16, dtype="handle"))
                        with T.block("C_reindex_wmma.accumulator"):
                            v0, v1, v2 = T.axis.remap("SSS", [ax_0_0_ax_0_0_fused, ax_0_2_ax_0_2_fused, ax_0_1_ax_0_1_fused])
                            T.reads(C_reindex_wmma_accumulator[v0 // 8 * 64 + v1 // 2 * 32 : v0 // 8 * 64 + v1 // 2 * 32 + 32, v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 : v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 + 32])
                            T.writes(C[v0 // 8 * 64 + v1 // 2 * 32 : v0 // 8 * 64 + v1 // 2 * 32 + 32, v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 : v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 + 32])
                            T.block_attr({"auto_copy":1, "vector_bytes":8})
                            for ax0, ax1 in T.grid(32, 32):
                                C[v0 // 8 * 64 + v1 // 2 * 32 + ax0, v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 + ax1] = C_reindex_wmma_accumulator[v0 // 8 * 64 + v1 // 2 * 32 + ax0, v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 + ax1]


# from tvm.script import tir as T
@tvm.script.ir_module
class Module_nested_pipeline:
    @T.prim_func
    def main(A: T.Buffer[(4096, 4096), "float16"], B: T.Buffer[(4096, 4096), "float16"], C: T.Buffer[(4096, 4096), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.tensor_core_enabled":"1", "warp_execution":1})
            C_reindex_wmma_accumulator = T.alloc_buffer([4096, 4096], dtype="float32", scope="wmma.accumulator")
            A_reindex_shared = T.alloc_buffer([4096, 4096], dtype="float16", scope="shared")
            B_reindex_shared = T.alloc_buffer([4096, 4096], dtype="float16", scope="shared")
            A_reindex_shared_wmma_matrix_a = T.alloc_buffer([4096, 4096], dtype="float16", scope="wmma.matrix_a")
            B_reindex_shared_wmma_matrix_b = T.alloc_buffer([4096, 4096], dtype="float16", scope="wmma.matrix_b")
            for ax_0_0_ax_0_0_fused in T.thread_binding(512, thread="blockIdx.x", annotations={"pragma_auto_unroll_max_step":512, "pragma_unroll_explicit":1}):
                for ax_0_1_ax_0_1_fused in T.thread_binding(8, thread="blockIdx.y"):
                    for ax_0_2_ax_0_2_fused in T.thread_binding(4, thread="threadIdx.y"):
                        for ax_0_4_init, ax_0_4_init_1 in T.grid(2, 2):
                            with T.block("update_o_init"):
                                bv_o = T.axis.spatial(256, ax_0_0_ax_0_0_fused // 8 * 4 + ax_0_2_ax_0_2_fused // 2 * 2 + ax_0_4_init)
                                bv_o_1 = T.axis.spatial(256, ax_0_0_ax_0_0_fused % 8 * 32 + ax_0_1_ax_0_1_fused * 4 + ax_0_2_ax_0_2_fused % 2 * 2 + ax_0_4_init_1)
                                T.reads()
                                T.writes(C_reindex_wmma_accumulator[bv_o * 16 : bv_o * 16 + 16, bv_o_1 * 16 : bv_o_1 * 16 + 16])
                                T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32})
                                for ax_1_0, ax_1_0_1 in T.grid(1, 1):
                                    with T.block("update_init_o"):
                                        bv_init_o = T.axis.spatial(1, 0)
                                        bv_init_o_1 = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_wmma_accumulator[bv_o * 16 : bv_o * 16 + 16, bv_o_1 * 16 : bv_o_1 * 16 + 16])
                                        C_1 = T.match_buffer(C_reindex_wmma_accumulator[bv_o * 16 : bv_o * 16 + 16, bv_o_1 * 16 : bv_o_1 * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                                        T.evaluate(T.tvm_fill_fragment(C_1.data, 16, 16, 16, C_1.elem_offset // 256 + C_1.elem_offset % 256 // 16, T.float32(0), dtype="handle"))
                        for ax_0_0 in T.serial(64, annotations={"software_pipeline_order":[0, 3, 1, 4, 5, 2, 6], "software_pipeline_stage":[0, 0, 0, 0, 0, 1, 1]}):
                            with T.block("A_reindex_shared"):
                                v0, v1 = T.axis.remap("SS", [ax_0_0_ax_0_0_fused, ax_0_0])
                                T.reads(A[v0 // 8 * 64 : v0 // 8 * 64 + 64, v1 * 64 : v1 * 64 + 64])
                                T.writes(A_reindex_shared[v0 // 8 * 64 : v0 // 8 * 64 + 64, v1 * 64 : v1 * 64 + 64])
                                T.block_attr({"auto_copy":1, "double_buffer_scope":0, "local_stage":1, "vector_bytes":16})
                                for ax0, ax1 in T.grid(64, 64):
                                    A_reindex_shared[v0 // 8 * 64 + ax0, v1 * 64 + ax1] = A[v0 // 8 * 64 + ax0, v1 * 64 + ax1]
                            with T.block("B_reindex_shared"):
                                v0, v1, v2 = T.axis.remap("SSS", [ax_0_0_ax_0_0_fused, ax_0_1_ax_0_1_fused, ax_0_0])
                                T.reads(B[v0 % 8 * 512 + v1 * 64 : v0 % 8 * 512 + v1 * 64 + 64, v2 * 64 : v2 * 64 + 64])
                                T.writes(B_reindex_shared[v0 % 8 * 512 + v1 * 64 : v0 % 8 * 512 + v1 * 64 + 64, v2 * 64 : v2 * 64 + 64])
                                T.block_attr({"auto_copy":1, "double_buffer_scope":0, "local_stage":1, "vector_bytes":16})
                                for ax0, ax1 in T.grid(64, 64):
                                    B_reindex_shared[v0 % 8 * 512 + v1 * 64 + ax0, v2 * 64 + ax1] = B[v0 % 8 * 512 + v1 * 64 + ax0, v2 * 64 + ax1]
                            for ax_0_1 in T.serial(2, annotations={"software_pipeline_order":[0, 1, 2], "software_pipeline_stage":[0, 0, 1]}):
                                with T.block("A_reindex_shared_wmma.matrix_a"):
                                    v0, v1, v2, v3 = T.axis.remap("SSSS", [ax_0_0_ax_0_0_fused, ax_0_2_ax_0_2_fused, ax_0_0, ax_0_1])
                                    T.reads(A_reindex_shared[v0 // 8 * 64 + v1 // 2 * 32 : v0 // 8 * 64 + v1 // 2 * 32 + 32, v2 * 64 + v3 * 32 : v2 * 64 + v3 * 32 + 32])
                                    T.writes(A_reindex_shared_wmma_matrix_a[v0 // 8 * 64 + v1 // 2 * 32 : v0 // 8 * 64 + v1 // 2 * 32 + 32, v2 * 64 + v3 * 32 : v2 * 64 + v3 * 32 + 32])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(32, 32):
                                        A_reindex_shared_wmma_matrix_a[v0 // 8 * 64 + v1 // 2 * 32 + ax0, v2 * 64 + v3 * 32 + ax1] = A_reindex_shared[v0 // 8 * 64 + v1 // 2 * 32 + ax0, v2 * 64 + v3 * 32 + ax1]
                                with T.block("B_reindex_shared_wmma.matrix_b"):
                                    v0, v1, v2, v3, v4 = T.axis.remap("SSSSS", [ax_0_0_ax_0_0_fused, ax_0_1_ax_0_1_fused, ax_0_2_ax_0_2_fused, ax_0_0, ax_0_1])
                                    T.reads(B_reindex_shared[v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 : v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 + 32, v3 * 64 + v4 * 32 : v3 * 64 + v4 * 32 + 32])
                                    T.writes(B_reindex_shared_wmma_matrix_b[v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 : v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 + 32, v3 * 64 + v4 * 32 : v3 * 64 + v4 * 32 + 32])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(32, 32):
                                        B_reindex_shared_wmma_matrix_b[v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 + ax0, v3 * 64 + v4 * 32 + ax1] = B_reindex_shared[v0 % 8 * 512 + v1 * 64 + v2 % 2 * 32 + ax0, v3 * 64 + v4 * 32 + ax1]
                                for ax_0_3, ax_0_3_1, ax_0_2, ax_0_4, ax_0_4_1 in T.grid(1, 1, 2, 2, 2):
                                    with T.block("update_o_update"):
                                        bv_o_2 = T.axis.spatial(256, ax_0_0_ax_0_0_fused // 8 * 4 + ax_0_2_ax_0_2_fused // 2 * 2 + ax_0_4)
                                        bv_o_3 = T.axis.spatial(256, ax_0_0_ax_0_0_fused % 8 * 32 + ax_0_1_ax_0_1_fused * 4 + ax_0_2_ax_0_2_fused % 2 * 2 + ax_0_4_1)
                                        bv_o_4 = T.axis.reduce(256, ax_0_0 * 4 + ax_0_1 * 2 + ax_0_2)
                                        T.reads(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16], A_reindex_shared_wmma_matrix_a[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16], B_reindex_shared_wmma_matrix_b[bv_o_3 * 16 : bv_o_3 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16])
                                        T.writes(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16])
                                        T.block_attr({"meta_schedule.thread_extent_high_inclusive":1024, "meta_schedule.thread_extent_low_inclusive":32})
                                        for ax_1_0, ax_1_0_2, ax_1_0_3 in T.grid(1, 1, 1):
                                            with T.block("update_o"):
                                                bv_o_5 = T.axis.spatial(1, 0)
                                                bv_o_6 = T.axis.spatial(1, 0)
                                                bv_o_7 = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16], A_reindex_shared_wmma_matrix_a[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16], B_reindex_shared_wmma_matrix_b[bv_o_3 * 16 : bv_o_3 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16])
                                                T.writes(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16])
                                                A_1 = T.match_buffer(A_reindex_shared_wmma_matrix_a[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                                B_1 = T.match_buffer(B_reindex_shared_wmma_matrix_b[bv_o_3 * 16 : bv_o_3 * 16 + 16, bv_o_4 * 16 : bv_o_4 * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_b", offset_factor=16)
                                                C_2 = T.match_buffer(C_reindex_wmma_accumulator[bv_o_2 * 16 : bv_o_2 * 16 + 16, bv_o_3 * 16 : bv_o_3 * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                                                T.evaluate(T.tvm_mma_sync(C_2.data, C_2.elem_offset // 256 + C_2.elem_offset % 256 // 16, A_1.data, A_1.elem_offset // 256, B_1.data, B_1.elem_offset // 256, C_2.data, C_2.elem_offset // 256 + C_2.elem_offset % 256 // 16, dtype="handle"))
                        with T.block("C_reindex_wmma.accumulator"):
                            v0, v1, v2 = T.axis.remap("SSS", [ax_0_0_ax_0_0_fused, ax_0_2_ax_0_2_fused, ax_0_1_ax_0_1_fused])
                            T.reads(C_reindex_wmma_accumulator[v0 // 8 * 64 + v1 // 2 * 32 : v0 // 8 * 64 + v1 // 2 * 32 + 32, v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 : v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 + 32])
                            T.writes(C[v0 // 8 * 64 + v1 // 2 * 32 : v0 // 8 * 64 + v1 // 2 * 32 + 32, v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 : v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 + 32])
                            T.block_attr({"auto_copy":1, "vector_bytes":8})
                            for ax0, ax1 in T.grid(32, 32):
                                C[v0 // 8 * 64 + v1 // 2 * 32 + ax0, v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 + ax1] = C_reindex_wmma_accumulator[v0 // 8 * 64 + v1 // 2 * 32 + ax0, v0 % 8 * 512 + v2 * 64 + v1 % 2 * 32 + ax1]


# f = tvm.build(Module_nested_pipeline, target="cuda", name="dense")
f = tvm.build(Module, target="cuda", name="dense")

N = K = M = 4096
dev = tvm.device("cuda", 0)
a_np = np.random.uniform(size=(N, K)).astype("float16")
b_np = np.random.uniform(size=(K, M)).astype("float16")
c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
# print(f.imported_modules[0].get_source())
f(a, b, c)
tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)
print("ok")

evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
gflops = (N*M*K) * 2 / 1e9
time_ms = evaluator(a, b, c).mean * 1e3
print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))

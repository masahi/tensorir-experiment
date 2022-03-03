# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

import math
import sys
from typing import List

import pytest
import tvm
from tvm.tir.schedule import BlockRV, Schedule
from tvm.error import TVMError
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.schedule_rule import PyScheduleRule
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.script import tir as T
from tvm.target import Target
from tvm import register_func

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MatmulCustomized:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        with T.block("root"):
            for i, j, k in T.grid(1024, 1024, 1024):
                with T.block("matmul"):
                    T.block_attr({"schedule_rule": "tvm.meta_schedule.test.custom_search_space"})
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MatmulCustomizedNoneRule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        with T.block("root"):
            T.block_attr({"schedule_rule": "None"})
            for i, j, k in T.grid(1024, 1024, 1024):
                with T.block("matmul"):
                    T.block_attr({"schedule_rule": "None"})
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

@tvm.script.ir_module
class DuplicateMatmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class TrinityMatmul:
    @T.prim_func
    def main(a: T.handle, d: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.alloc_buffer((1024, 1024), "float32")
        C = T.alloc_buffer((1024, 1024), "float32")
        D = T.match_buffer(d, (1024, 1024), "float32")
        for i, j in T.grid(1024, 1024):
            with T.block("A"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0
        for i, j in T.grid(1024, 1024):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + 3.0
        for i, j in T.grid(1024, 1024):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = C[vi, vj] * 5.0


@tvm.script.ir_module
class TrinityMatmulProcessedForReference:
    @T.prim_func
    def main(a: T.handle, d: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, [1024, 1024], dtype="float32")
        D = T.match_buffer(d, [1024, 1024], dtype="float32")
        # body
        # with T.block("root")
        B = T.alloc_buffer([1024, 1024], dtype="float32")
        for i0_0, i1_0, i0_1, i1_1 in T.grid(16, 64, 64, 16):
            with T.block("A"):
                vi = T.axis.S(1024, i0_0 * 64 + i0_1)
                vj = T.axis.S(1024, i1_0 * 16 + i1_1)
                T.reads([A[vi, vj]])
                T.writes([B[vi, vj]])
                B[vi, vj] = A[vi, vj] * T.float32(2)
        for i0_0, i1_0, i0_1, i1_1 in T.grid(16, 64, 64, 16):
            with T.block("C"):
                vi = T.axis.S(1024, i0_0 * 64 + i0_1)
                vj = T.axis.S(1024, i1_0 * 16 + i1_1)
                T.reads([B[vi, vj]])
                T.writes([D[vi, vj]])
                D[vi, vj] = (B[vi, vj] + T.float32(3)) * T.float32(5)


# with T.block("root"):

#     with T.block("A"):
#         # template: meta_schedule.testing.some_rule
#         ...
#     with T.block("B"):
#         # ReLU
#         ...
#     with T.block("C"):
#         # bias_add
#         ...



@tvm.script.ir_module
class Conv2d_Winograd:
    @T.prim_func
    def main(var_placeholder: T.handle, var_placeholder_1: T.handle, var_conv2d_winograd: T.handle) -> None:
        # function attr dict
        T.func_attr({"layout_free_placeholders": [var_placeholder_1]})
        placeholder = T.match_buffer(var_placeholder, [1, 14, 14, 128], elem_offset=0, align=128, offset_factor=1)
        placeholder_1 = T.match_buffer(var_placeholder_1, [6, 6, 128, 128], elem_offset=0, align=128, offset_factor=1)
        conv2d_winograd = T.match_buffer(var_conv2d_winograd, [1, 12, 12, 128], elem_offset=0, align=128, offset_factor=1)
        # body
        with T.block("root"):
            T.block_attr({"schedule_rule": "tvm.meta_schedule.test.custom_search_space.winograd"})
            data_pad = T.alloc_buffer([1, 16, 16, 128], elem_offset=0, align=128, offset_factor=1)
            input_tile = T.alloc_buffer([6, 6, 9, 128], elem_offset=0, align=128, offset_factor=1)
            B = T.alloc_buffer([6, 6], elem_offset=0, align=128, offset_factor=1)
            data_pack = T.alloc_buffer([6, 6, 9, 128], elem_offset=0, align=128, offset_factor=1)
            bgemm = T.alloc_buffer([6, 6, 9, 128], elem_offset=0, align=128, offset_factor=1)
            A = T.alloc_buffer([6, 4], elem_offset=0, align=128, offset_factor=1)
            inverse = T.alloc_buffer([4, 4, 9, 128], elem_offset=0, align=128, offset_factor=1)
            for i0, i1, i2, i3 in T.grid(1, 16, 16, 128):
                with T.block("data_pad"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads([placeholder[i0_1, i1_1, i2_1, i3_1]])
                    T.writes([data_pad[i0_1, i1_1, i2_1, i3_1]])
                    T.block_attr({
                        "schedule_rule": "None",
                    })
                    data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(((((0 <= i1_1) and (i1_1 < 14)) and (0 <= i2_1)) and (i2_1 < 14)), placeholder[i0_1, i1_1, i2_1, i3_1], T.float32(0), dtype="float32")
            for i0_2, i1_2, i2_2, i3_2 in T.grid(6, 6, 9, 128):
                with T.block("input_tile"):
                    eps, nu, p, ci = T.axis.remap("SSSS", [i0_2, i1_2, i2_2, i3_2])
                    T.reads([data_pad[T.floordiv(p, 9), ((T.floordiv(T.floormod(p, 9), 3)*4) + eps), ((T.floormod(p, 3)*4) + nu), ci]])
                    T.writes([input_tile[eps, nu, p, ci]])
                    T.block_attr({
                        "schedule_rule": "None",
                    })
                    input_tile[eps, nu, p, ci] = data_pad[T.floordiv(p, 9), ((T.floordiv(T.floormod(p, 9), 3)*4) + eps), ((T.floormod(p, 3)*4) + nu), ci]
            for i0_3, i1_3 in T.grid(6, 6):
                with T.block("B"):
                    i, j = T.axis.remap("SS", [i0_3, i1_3])
                    T.writes([B[i, j]])
                    T.block_attr({
                        "const_matrix" : True,
                        "schedule_rule": "None",
                    })
                    B[i, j] = T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 5)), T.float32(1), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 4)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 3)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 2)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 1)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 0)), T.float32(0), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 5)), T.float32(1.5), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 4)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 3)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 2)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 1)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 0)), T.float32(1), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 5)), T.float32(-2), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 4)), T.float32(-0.5), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 3)), T.float32(2), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 2)), T.float32(2.5), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 1)), T.float32(0.5), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 0)), T.float32(1.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 5)), T.float32(-1.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 4)), T.float32(-1), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 3)), T.float32(-1), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 2)), T.float32(0.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 1)), T.float32(-2.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 0)), T.float32(-2), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 5)), T.float32(1), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 4)), T.float32(0.5), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 3)), T.float32(-2), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 2)), T.float32(-1), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 1)), T.float32(1), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 0)), T.float32(-1.5), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 5)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 4)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 3)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 2)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 1)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 0)), T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
            for i0_4, i1_4, i2_3, i3_3, i4, i5 in T.grid(6, 6, 9, 128, 6, 6):
                with T.block("data_pack"):
                    eps_1, nu_1, p_1, ci_1, r_a, r_b = T.axis.remap("SSSSRR", [i0_4, i1_4, i2_3, i3_3, i4, i5])
                    T.reads([data_pack[eps_1, nu_1, p_1, ci_1], input_tile[r_a, r_b, p_1, ci_1], B[T.min(r_a, r_b):(T.min(r_a, r_b) + ((T.max(r_a, r_b) + 1) - T.min(r_a, r_b))), T.min(eps_1, nu_1):(T.min(eps_1, nu_1) + ((T.max(eps_1, nu_1) + 1) - T.min(eps_1, nu_1)))]])
                    T.writes([data_pack[eps_1, nu_1, p_1, ci_1]])
                    T.block_attr({
                        "auto_scheduler_simplify_const_tensor_indices":["eps", "nu", "r_a", "r_b"],
                        "schedule_rule": "None",
                    })
                    with T.init():
                        data_pack[eps_1, nu_1, p_1, ci_1] = T.float32(0)
                    data_pack[eps_1, nu_1, p_1, ci_1] = (data_pack[eps_1, nu_1, p_1, ci_1] + ((input_tile[r_a, r_b, p_1, ci_1]*B[r_a, eps_1])*B[r_b, nu_1]))
            for i0_5, i1_5, i2_4, i3_4, i4_1 in T.grid(6, 6, 9, 128, 128):
                with T.block("bgemm"):
                    eps_2, nu_2, p_2, co, ci_2 = T.axis.remap("SSSSR", [i0_5, i1_5, i2_4, i3_4, i4_1])
                    T.reads([bgemm[eps_2, nu_2, p_2, co], data_pack[eps_2, nu_2, p_2, ci_2], placeholder_1[eps_2, nu_2, co, ci_2]])
                    T.writes([bgemm[eps_2, nu_2, p_2, co]])
                    T.block_attr({
                        "schedule_rule": "None",
                    })
                    with T.init():
                        bgemm[eps_2, nu_2, p_2, co] = T.float32(0)
                    bgemm[eps_2, nu_2, p_2, co] = (bgemm[eps_2, nu_2, p_2, co] + (data_pack[eps_2, nu_2, p_2, ci_2]*placeholder_1[eps_2, nu_2, co, ci_2]))
            for i0_6, i1_6 in T.grid(6, 4):
                with T.block("A"):
                    i_1, j_1 = T.axis.remap("SS", [i0_6, i1_6])
                    T.writes([A[i_1, j_1]])
                    T.block_attr({
                        "const_matrix" : True,
                        "schedule_rule": "None",
                    })
                    A[i_1, j_1] = T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 3)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 2)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 1)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 0)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 3)), T.float32(-8), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 2)), T.float32(4), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 1)), T.float32(-2), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 3)), T.float32(0.125), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 2)), T.float32(0.25), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 1)), T.float32(0.5), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 3)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 2)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 1)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 3)), T.float32(-1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 2)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 1)), T.float32(-1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 3)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 2)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 1)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.float32(0)))))))))))))))))))))))))
            for i0_7, i1_7, i2_5, i3_5, i4_2, i5_1 in T.grid(4, 4, 9, 128, 6, 6):
                with T.block("inverse"):
                    vh, vw, p_3, co_1, r_a_1, r_b_1 = T.axis.remap("SSSSRR", [i0_7, i1_7, i2_5, i3_5, i4_2, i5_1])
                    T.reads([inverse[vh, vw, p_3, co_1], bgemm[r_a_1, r_b_1, p_3, co_1], A[T.min(r_a_1, r_b_1):(T.min(r_a_1, r_b_1) + ((T.max(r_a_1, r_b_1) + 1) - T.min(r_a_1, r_b_1))), T.min(vh, vw):(T.min(vh, vw) + ((T.max(vh, vw) + 1) - T.min(vh, vw)))]])
                    T.writes([inverse[vh, vw, p_3, co_1]])
                    T.block_attr({
                        "schedule_rule": "None",
                        "auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"],
                    })
                    with T.init():
                        inverse[vh, vw, p_3, co_1] = T.float32(0)
                    inverse[vh, vw, p_3, co_1] = (inverse[vh, vw, p_3, co_1] + ((bgemm[r_a_1, r_b_1, p_3, co_1]*A[r_a_1, vh])*A[r_b_1, vw]))
            for i0_8, i1_8, i2_6, i3_6 in T.grid(1, 12, 12, 128):
                with T.block("conv2d_winograd"):
                    n, h, w, co_2 = T.axis.remap("SSSS", [i0_8, i1_8, i2_6, i3_6])
                    T.reads([inverse[T.floormod(h, 4), T.floormod(w, 4), (((n*9) + (T.floordiv(h, 4)*3)) + T.floordiv(w, 4)), co_2]])
                    T.writes([conv2d_winograd[n, h, w, co_2]])
                    T.block_attr({
                        "schedule_rule": "None"
                    })
                    conv2d_winograd[n, h, w, co_2] = inverse[T.floormod(h, 4), T.floormod(w, 4), (((n*9) + (T.floordiv(h, 4)*3)) + T.floordiv(w, 4)), co_2]

@tvm.script.ir_module
class Conv2d_Winograd_Cuda:
    @T.prim_func
    def main(var_placeholder: T.handle, var_placeholder_1: T.handle, var_conv2d_winograd: T.handle) -> None:
        # function attr dict
        T.func_attr({"layout_free_placeholders": [var_placeholder_1]})
        placeholder = T.match_buffer(var_placeholder, [1, 14, 14, 128], elem_offset=0, align=128, offset_factor=1)
        placeholder_1 = T.match_buffer(var_placeholder_1, [6, 6, 128, 128], elem_offset=0, align=128, offset_factor=1)
        conv2d_winograd = T.match_buffer(var_conv2d_winograd, [1, 12, 12, 128], elem_offset=0, align=128, offset_factor=1)
        # body
        with T.block("root"):
            data_pad = T.alloc_buffer([1, 16, 16, 128], elem_offset=0, align=128, offset_factor=1)
            input_tile = T.alloc_buffer([6, 6, 9, 128], elem_offset=0, align=128, offset_factor=1)
            B = T.alloc_buffer([6, 6], elem_offset=0, align=128, offset_factor=1)
            data_pack = T.alloc_buffer([6, 6, 9, 128], elem_offset=0, align=128, offset_factor=1)
            bgemm = T.alloc_buffer([6, 6, 9, 128], elem_offset=0, align=128, offset_factor=1)
            A = T.alloc_buffer([6, 4], elem_offset=0, align=128, offset_factor=1)
            inverse = T.alloc_buffer([4, 4, 9, 128], elem_offset=0, align=128, offset_factor=1)
            for i0, i1, i2, i3 in T.grid(1, 16, 16, 128):
                with T.block("data_pad"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.block_attr({
                        "schedule_rule": "None",
                    })
                    T.reads([placeholder[i0_1, i1_1, i2_1, i3_1]])
                    T.writes([data_pad[i0_1, i1_1, i2_1, i3_1]])
                    data_pad[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(((((0 <= i1_1) and (i1_1 < 14)) and (0 <= i2_1)) and (i2_1 < 14)), placeholder[i0_1, i1_1, i2_1, i3_1], T.float32(0), dtype="float32")
            for i0_2, i1_2, i2_2, i3_2 in T.grid(6, 6, 9, 128):
                with T.block("input_tile"):
                    eps, nu, p, ci = T.axis.remap("SSSS", [i0_2, i1_2, i2_2, i3_2])
                    T.block_attr({
                        "schedule_rule": "None",
                    })
                    T.reads([data_pad[T.floordiv(p, 9), ((T.floordiv(T.floormod(p, 9), 3)*4) + eps), ((T.floormod(p, 3)*4) + nu), ci]])
                    T.writes([input_tile[eps, nu, p, ci]])
                    input_tile[eps, nu, p, ci] = data_pad[T.floordiv(p, 9), ((T.floordiv(T.floormod(p, 9), 3)*4) + eps), ((T.floormod(p, 3)*4) + nu), ci]
            for i0_3, i1_3 in T.grid(6, 6):
                with T.block("B"):
                    i, j = T.axis.remap("SS", [i0_3, i1_3])
                    T.writes([B[i, j]])
                    T.block_attr({
                        "const_matrix":True,
                        "schedule_rule": "None",
                    })
                    B[i, j] = T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 5)), T.float32(1), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 4)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 3)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 2)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 1)), T.float32(0), T.Select(((T.floormod(i, 6) == 5) and (T.floormod(j, 6) == 0)), T.float32(0), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 5)), T.float32(1.5), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 4)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 3)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 2)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 1)), T.float32(1), T.Select(((T.floormod(i, 6) == 4) and (T.floormod(j, 6) == 0)), T.float32(1), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 5)), T.float32(-2), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 4)), T.float32(-0.5), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 3)), T.float32(2), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 2)), T.float32(2.5), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 1)), T.float32(0.5), T.Select(((T.floormod(i, 6) == 3) and (T.floormod(j, 6) == 0)), T.float32(1.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 5)), T.float32(-1.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 4)), T.float32(-1), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 3)), T.float32(-1), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 2)), T.float32(0.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 1)), T.float32(-2.5), T.Select(((T.floormod(i, 6) == 2) and (T.floormod(j, 6) == 0)), T.float32(-2), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 5)), T.float32(1), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 4)), T.float32(0.5), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 3)), T.float32(-2), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 2)), T.float32(-1), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 1)), T.float32(1), T.Select(((T.floormod(i, 6) == 1) and (T.floormod(j, 6) == 0)), T.float32(-1.5), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 5)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 4)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 3)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 2)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 1)), T.float32(0), T.Select(((T.floormod(i, 6) == 0) and (T.floormod(j, 6) == 0)), T.float32(1), T.float32(0)))))))))))))))))))))))))))))))))))))
            for i0_4, i1_4, i2_3, i3_3, i4, i5 in T.grid(6, 6, 9, 128, 6, 6):
                with T.block("data_pack"):
                    eps_1, nu_1, p_1, ci_1, r_a, r_b = T.axis.remap("SSSSRR", [i0_4, i1_4, i2_3, i3_3, i4, i5])
                    T.reads([data_pack[eps_1, nu_1, p_1, ci_1], input_tile[r_a, r_b, p_1, ci_1], B[T.min(r_a, r_b):(T.min(r_a, r_b) + ((T.max(r_a, r_b) + 1) - T.min(r_a, r_b))), T.min(eps_1, nu_1):(T.min(eps_1, nu_1) + ((T.max(eps_1, nu_1) + 1) - T.min(eps_1, nu_1)))]])
                    T.writes([data_pack[eps_1, nu_1, p_1, ci_1]])
                    T.block_attr({
                        "auto_scheduler_simplify_const_tensor_indices": ["eps", "nu", "r_a", "r_b"],
                        "schedule_rule": "None",
                    })
                    with T.init():
                        data_pack[eps_1, nu_1, p_1, ci_1] = T.float32(0)
                    data_pack[eps_1, nu_1, p_1, ci_1] = (data_pack[eps_1, nu_1, p_1, ci_1] + ((input_tile[r_a, r_b, p_1, ci_1]*B[r_a, eps_1])*B[r_b, nu_1]))
            for i0_5, i1_5, i2_4, i3_4, i4_1 in T.grid(6, 6, 9, 128, 128):
                with T.block("bgemm"):
                    T.block_attr({
                        "schedule_rule": "None",
                    })
                    eps_2, nu_2, p_2, co, ci_2 = T.axis.remap("SSSSR", [i0_5, i1_5, i2_4, i3_4, i4_1])
                    T.reads([bgemm[eps_2, nu_2, p_2, co], data_pack[eps_2, nu_2, p_2, ci_2], placeholder_1[eps_2, nu_2, co, ci_2]])
                    T.writes([bgemm[eps_2, nu_2, p_2, co]])
                    with T.init():
                        bgemm[eps_2, nu_2, p_2, co] = T.float32(0)
                    bgemm[eps_2, nu_2, p_2, co] = (bgemm[eps_2, nu_2, p_2, co] + (data_pack[eps_2, nu_2, p_2, ci_2]*placeholder_1[eps_2, nu_2, co, ci_2]))
            for i0_6, i1_6 in T.grid(6, 4):
                with T.block("A"):
                    i_1, j_1 = T.axis.remap("SS", [i0_6, i1_6])
                    T.writes([A[i_1, j_1]])
                    T.block_attr({
                        "const_matrix":True,
                        "schedule_rule": "None",
                    })
                    A[i_1, j_1] = T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 3)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 2)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 1)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 5) and (T.floormod(j_1, 4) == 0)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 3)), T.float32(-8), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 2)), T.float32(4), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 1)), T.float32(-2), T.Select(((T.floormod(i_1, 6) == 4) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 3)), T.float32(0.125), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 2)), T.float32(0.25), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 1)), T.float32(0.5), T.Select(((T.floormod(i_1, 6) == 3) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 3)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 2)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 1)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 2) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 3)), T.float32(-1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 2)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 1)), T.float32(-1), T.Select(((T.floormod(i_1, 6) == 1) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 3)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 2)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 1)), T.float32(0), T.Select(((T.floormod(i_1, 6) == 0) and (T.floormod(j_1, 4) == 0)), T.float32(1), T.float32(0)))))))))))))))))))))))))
            for i0_7, i1_7, i2_5, i3_5, i4_2, i5_1 in T.grid(4, 4, 9, 128, 6, 6):
                with T.block("inverse"):
                    vh, vw, p_3, co_1, r_a_1, r_b_1 = T.axis.remap("SSSSRR", [i0_7, i1_7, i2_5, i3_5, i4_2, i5_1])
                    T.reads([inverse[vh, vw, p_3, co_1], bgemm[r_a_1, r_b_1, p_3, co_1], A[T.min(r_a_1, r_b_1):(T.min(r_a_1, r_b_1) + ((T.max(r_a_1, r_b_1) + 1) - T.min(r_a_1, r_b_1))), T.min(vh, vw):(T.min(vh, vw) + ((T.max(vh, vw) + 1) - T.min(vh, vw)))]])
                    T.writes([inverse[vh, vw, p_3, co_1]])
                    T.block_attr({
                        "auto_scheduler_simplify_const_tensor_indices":["vh", "vw", "r_a", "r_b"],
                        "schedule_rule": "None",
                    })
                    with T.init():
                        inverse[vh, vw, p_3, co_1] = T.float32(0)
                    inverse[vh, vw, p_3, co_1] = (inverse[vh, vw, p_3, co_1] + ((bgemm[r_a_1, r_b_1, p_3, co_1]*A[r_a_1, vh])*A[r_b_1, vw]))
            for i0_8, i1_8, i2_6, i3_6 in T.grid(1, 12, 12, 128):
                with T.block("conv2d_winograd"):
                    n, h, w, co_2 = T.axis.remap("SSSS", [i0_8, i1_8, i2_6, i3_6])
                    T.block_attr({
                        "schedule_rule": "None",
                    })
                    T.reads([inverse[T.floormod(h, 4), T.floormod(w, 4), (((n*9) + (T.floordiv(h, 4)*3)) + T.floordiv(w, 4)), co_2]])
                    T.writes([conv2d_winograd[n, h, w, co_2]])
                    conv2d_winograd[n, h, w, co_2] = inverse[T.floormod(h, 4), T.floormod(w, 4), (((n*9) + (T.floordiv(h, 4)*3)) + T.floordiv(w, 4)), co_2]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _is_root(sch: Schedule, block: BlockRV) -> bool:
    return sch.get_sref(block).parent is None


def _check_correct(schedule: Schedule):
    trace = schedule.trace
    for inst in trace.decisions:
        assert math.prod(trace.decisions[inst]) == 1024


class WowSoFancyScheduleRule(PyScheduleRule):
    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        if _is_root(sch, block):
            return [sch]
        new_sch = sch.copy()
        i, j, k = new_sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[2, 4, 64, 2])
        j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[4, 64, 2, 2])
        k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
        new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        return [new_sch]


class DoubleScheduleRule(PyScheduleRule):
    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        if _is_root(sch, block):
            return [sch]
        new_sch = sch.copy()
        i, j, k = new_sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[4, 64, 2, 2])
        j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[2, 4, 64, 2])
        k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
        new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        result = [new_sch]
        new_sch = sch.copy()
        i, j, k = new_sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[4, 64, 2, 2])
        j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[2, 4, 64, 2])
        k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
        new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        result.append(new_sch)
        return result


class ReorderScheduleRule(PyScheduleRule):
    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        if _is_root(sch, block):
            return [sch]
        new_sch = sch.copy()
        i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = new_sch.get_loops(block=block)
        new_sch.reorder(i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3, i_0, j_0)
        result = [new_sch]
        new_sch = sch.copy()
        i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = new_sch.get_loops(block=block)
        new_sch.reorder(i_1, j_3, i_0, j_0, j_1, k_0, i_2, j_2, k_1, i_3)
        result.append(new_sch)
        return result


def test_meta_schedule_post_order_apply():
    mod = Matmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Test Task",
        sch_rules=[WowSoFancyScheduleRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 1
    assert not tvm.ir.structural_equal(schs[0].mod, mod)
    _check_correct(schs[0])


def test_meta_schedule_post_order_apply_double():
    mod = Matmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Double Rules Task",
        sch_rules=[DoubleScheduleRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 2
    for sch in schs:
        assert not tvm.ir.structural_equal(sch.mod, mod)
        _check_correct(sch)


def test_meta_schedule_post_order_apply_multiple():
    mod = Matmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Double Rules Task",
        sch_rules=[DoubleScheduleRule(), ReorderScheduleRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 4
    for sch in schs:
        assert not tvm.ir.structural_equal(sch.mod, mod)
        _check_correct(sch)


def test_meta_schedule_post_order_apply_duplicate_matmul():
    mod = DuplicateMatmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Duplicate Matmul Task",
        sch_rules=[WowSoFancyScheduleRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    with pytest.raises(
        TVMError,
        match=r".*TVMError: Check failed: \(block_names_.count\(block->name_hint\) == 0\)"
        r" is false: Duplicated block name matmul in function main not supported!",
    ):
        post_order_apply.generate_design_space(mod)


def test_meta_schedule_post_order_apply_remove_block():
    class TrinityDouble(PyScheduleRule):
        def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
            pass

        def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
            if _is_root(sch, block):
                return [sch]
            new_sch = sch.copy()
            i, j = new_sch.get_loops(block=block)
            i_0, i_1 = new_sch.split(loop=i, factors=[16, 64])
            j_0, j_1 = new_sch.split(loop=j, factors=[64, 16])
            new_sch.reorder(i_0, j_0, i_1, j_1)
            result = [new_sch]
            new_sch = sch.copy()
            i, j = new_sch.get_loops(block=block)
            i_0, i_1 = new_sch.split(loop=i, factors=[2, 512])
            j_0, j_1 = new_sch.split(loop=j, factors=[2, 512])
            new_sch.reorder(i_0, j_0, i_1, j_1)
            result.append(new_sch)
            return result

    class RemoveBlock(PyScheduleRule):
        def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
            pass

        def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
            if _is_root(sch, block):
                return [sch]
            sch = sch.copy()
            if sch.get(block).name_hint == "B":
                sch.compute_inline(block)
            return [sch]

    def correct_trace(a, b, c, d):
        return "\n".join(
            [
                'b0 = sch.get_block(name="A", func_name="main")',
                'b1 = sch.get_block(name="B", func_name="main")',
                'b2 = sch.get_block(name="C", func_name="main")',
                "sch.compute_inline(block=b1)",
                "l3, l4 = sch.get_loops(block=b2)",
                "l5, l6 = sch.split(loop=l3, factors=" + str(a) + ")",
                "l7, l8 = sch.split(loop=l4, factors=" + str(b) + ")",
                "sch.reorder(l5, l7, l6, l8)",
                "l9, l10 = sch.get_loops(block=b0)",
                "l11, l12 = sch.split(loop=l9, factors=" + str(c) + ")",
                "l13, l14 = sch.split(loop=l10, factors=" + str(d) + ")",
                "sch.reorder(l11, l13, l12, l14)",
            ]
        )

    mod = TrinityMatmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Remove Block Task",
        sch_rules=[RemoveBlock(), TrinityDouble()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 4
    for sch in schs:
        with pytest.raises(
            tvm.tir.schedule.schedule.ScheduleError,
            match="ScheduleError: An error occurred in the schedule primitive 'get-block'.",
        ):
            sch.get_block("B", "main")
        sch_trace = sch.trace.simplified(True)
        assert (
            str(sch_trace) == correct_trace([16, 64], [64, 16], [2, 512], [2, 512])
            or str(sch_trace) == correct_trace([2, 512], [2, 512], [2, 512], [2, 512])
            or str(sch_trace) == correct_trace([16, 64], [64, 16], [16, 64], [64, 16])
            or str(sch_trace) == correct_trace([2, 512], [2, 512], [16, 64], [64, 16])
        )


def test_meta_schedule_post_order_apply_custom_search_space():
    @register_func("tvm.meta_schedule.test.custom_search_space")
    def custom_search_space_func(sch: Schedule, block: BlockRV):
        raise ValueError("Customized search space triggered!")

    mod = MatmulCustomized
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Custom Search Space Task",
        sch_rules=[],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    with pytest.raises(ValueError, match="Customized search space triggered!"):
        _ = post_order_apply.generate_design_space(mod)


class DontCallThisRule(PyScheduleRule):
    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        print(sch.get(block))
        raise RuntimeError("This schedule rule should not be called!")


def test_meta_schedule_post_order_apply_custom_search_space_none_rule():
    mod = MatmulCustomizedNoneRule
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Custom Search Space Task",
        sch_rules=[DontCallThisRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    _ = post_order_apply.generate_design_space(mod)


def test_meta_schedule_post_order_apply_custom_search_space_winograd():
    @register_func("tvm.meta_schedule.test.custom_search_space.winograd")
    def custom_search_space_winograd_func(sch: Schedule, block: BlockRV) -> List[Schedule]:
        b1 = sch.get_block(name="A")
        sch.compute_inline(block=b1)
        b2 = sch.get_block(name="B")
        sch.compute_inline(block=b2)
        b3 = sch.get_block(name="inverse")
        l4, l5, l6, l7, l8, l9 = sch.get_loops(block=b3)
        sch.unroll(loop=l4)
        sch.unroll(loop=l5)
        sch.unroll(loop=l8)
        sch.unroll(loop=l9)
        v10, v11 = sch.sample_perfect_tile(n=2, loop=l6, max_innermost_factor=64, decision=[1, 9])
        l12, l13 = sch.split(loop=l6, factors=[v10, v11])
        v14, v15 = sch.sample_perfect_tile(n=2, loop=l7, max_innermost_factor=64, decision=[2, 64])
        l16, l17 = sch.split(loop=l7, factors=[v14, v15])
        sch.reorder(l12, l16, l13, l17, l4, l5, l8, l9)
        b18 = sch.get_block(name="data_pack")
        l19, l20, l21, l22, l23, l24 = sch.get_loops(block=b18)
        sch.unroll(loop=l19)
        sch.unroll(loop=l20)
        sch.unroll(loop=l23)
        sch.unroll(loop=l24)
        v25, v26 = sch.sample_perfect_tile(n=2, loop=l21, max_innermost_factor=64, decision=[9, 1])
        l27, l28 = sch.split(loop=l21, factors=[v25, v26])
        v29, v30 = sch.sample_perfect_tile(n=2, loop=l22, max_innermost_factor=64, decision=[32, 4])
        l31, l32 = sch.split(loop=l22, factors=[v29, v30])
        sch.reorder(l27, l31, l28, l32, l19, l20, l23, l24)
        b33 = sch.get_block(name="bgemm")
        b34 = sch.cache_write(block=b33, write_buffer_index=0, storage_scope="global")
        b33, b34 = b34, b33
        l35, l36, l37, l38, l39 = sch.get_loops(block=b34)
        v40, v41, v42, v43 = sch.sample_perfect_tile(
            n=4, loop=l35, max_innermost_factor=64, decision=[1, 2, 3, 1]
        )
        l44, l45, l46, l47 = sch.split(loop=l35, factors=[v40, v41, v42, v43])
        v48, v49, v50, v51 = sch.sample_perfect_tile(
            n=4, loop=l36, max_innermost_factor=64, decision=[1, 1, 1, 6]
        )
        l52, l53, l54, l55 = sch.split(loop=l36, factors=[v48, v49, v50, v51])
        v56, v57, v58, v59 = sch.sample_perfect_tile(
            n=4, loop=l37, max_innermost_factor=64, decision=[1, 1, 1, 9]
        )
        l60, l61, l62, l63 = sch.split(loop=l37, factors=[v56, v57, v58, v59])
        v64, v65, v66, v67 = sch.sample_perfect_tile(
            n=4, loop=l38, max_innermost_factor=64, decision=[2, 1, 16, 4]
        )
        l68, l69, l70, l71 = sch.split(loop=l38, factors=[v64, v65, v66, v67])
        v72, v73 = sch.sample_perfect_tile(n=2, loop=l39, max_innermost_factor=64, decision=[16, 8])
        l74, l75 = sch.split(loop=l39, factors=[v72, v73])
        sch.reorder(
            l44, l52, l60, l68, l45, l53, l61, l69, l74, l46, l54, l62, l70, l75, l47, l55, l63, l71
        )
        sch.reverse_compute_at(block=b33, loop=l69, preserve_unit_loops=True)
        b76 = sch.get_block(name="root")
        sch.annotate(block_or_loop=b76, ann_key="auto_parallel_extent", ann_val=64)
        sch.annotate(block_or_loop=b76, ann_key="auto_vectorize_extent", ann_val=32)
        v77 = sch.sample_categorical(
            candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1
        )
        sch.annotate(block_or_loop=b76, ann_key="auto_unroll_explicit", ann_val=v77)

        b78 = sch.get_block(name="input_tile")
        l80 = sch.sample_compute_location(block=b78, decision=4)
        sch.compute_at(block=b78, loop=l80, preserve_unit_loops=True)

        b81 = sch.get_block(name="data_pad")
        (b82,) = sch.get_consumers(block=b81)
        l83 = sch.sample_compute_location(block=b82, decision=-2)
        sch.compute_at(block=b81, loop=l83, preserve_unit_loops=True)
        return [sch]

    mod = Conv2d_Winograd

    # Add annotation
    sch = Schedule(mod)
    sch.annotate(
        sch.get_block("root"),
        "schedule_rule",
        "tvm.meta_schedule.test.custom_search_space.winograd",
    )
    mod = sch.mod
    context = TuneContext(
        mod=mod,
        target=Target("llvm --num-cores=16"),
        task_name="Custom Search Space Task",
        sch_rules=[DontCallThisRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 1
    (sch,) = schs
    assert str(sch.trace) == "\n".join(
        [
            'b0 = sch.get_block(name="data_pad", func_name="main")',
            'b1 = sch.get_block(name="input_tile", func_name="main")',
            'b2 = sch.get_block(name="B", func_name="main")',
            'b3 = sch.get_block(name="data_pack", func_name="main")',
            'b4 = sch.get_block(name="bgemm", func_name="main")',
            'b5 = sch.get_block(name="A", func_name="main")',
            'b6 = sch.get_block(name="inverse", func_name="main")',
            'b7 = sch.get_block(name="conv2d_winograd", func_name="main")',
            'b8 = sch.get_block(name="root", func_name="main")',
            'b9 = sch.get_block(name="A", func_name="main")',
            "sch.compute_inline(block=b9)",
            'b10 = sch.get_block(name="B", func_name="main")',
            "sch.compute_inline(block=b10)",
            'b11 = sch.get_block(name="inverse", func_name="main")',
            "l12, l13, l14, l15, l16, l17 = sch.get_loops(block=b11)",
            "sch.unroll(loop=l12)",
            "sch.unroll(loop=l13)",
            "sch.unroll(loop=l16)",
            "sch.unroll(loop=l17)",
            "v18, v19 = sch.sample_perfect_tile(loop=l14, n=2, max_innermost_factor=64, decision=[1, 9])",
            "l20, l21 = sch.split(loop=l14, factors=[v18, v19])",
            "v22, v23 = sch.sample_perfect_tile(loop=l15, n=2, max_innermost_factor=64, decision=[2, 64])",
            "l24, l25 = sch.split(loop=l15, factors=[v22, v23])",
            "sch.reorder(l20, l24, l21, l25, l12, l13, l16, l17)",
            'b26 = sch.get_block(name="data_pack", func_name="main")',
            "l27, l28, l29, l30, l31, l32 = sch.get_loops(block=b26)",
            "sch.unroll(loop=l27)",
            "sch.unroll(loop=l28)",
            "sch.unroll(loop=l31)",
            "sch.unroll(loop=l32)",
            "v33, v34 = sch.sample_perfect_tile(loop=l29, n=2, max_innermost_factor=64, decision=[9, 1])",
            "l35, l36 = sch.split(loop=l29, factors=[v33, v34])",
            "v37, v38 = sch.sample_perfect_tile(loop=l30, n=2, max_innermost_factor=64, decision=[32, 4])",
            "l39, l40 = sch.split(loop=l30, factors=[v37, v38])",
            "sch.reorder(l35, l39, l36, l40, l27, l28, l31, l32)",
            'b41 = sch.get_block(name="bgemm", func_name="main")',
            'b42 = sch.cache_write(block=b41, write_buffer_index=0, storage_scope="global")',
            "l43, l44, l45, l46, l47 = sch.get_loops(block=b41)",
            "v48, v49, v50, v51 = sch.sample_perfect_tile(loop=l43, n=4, max_innermost_factor=64, decision=[1, 2, 3, 1])",
            "l52, l53, l54, l55 = sch.split(loop=l43, factors=[v48, v49, v50, v51])",
            "v56, v57, v58, v59 = sch.sample_perfect_tile(loop=l44, n=4, max_innermost_factor=64, decision=[1, 1, 1, 6])",
            "l60, l61, l62, l63 = sch.split(loop=l44, factors=[v56, v57, v58, v59])",
            "v64, v65, v66, v67 = sch.sample_perfect_tile(loop=l45, n=4, max_innermost_factor=64, decision=[1, 1, 1, 9])",
            "l68, l69, l70, l71 = sch.split(loop=l45, factors=[v64, v65, v66, v67])",
            "v72, v73, v74, v75 = sch.sample_perfect_tile(loop=l46, n=4, max_innermost_factor=64, decision=[2, 1, 16, 4])",
            "l76, l77, l78, l79 = sch.split(loop=l46, factors=[v72, v73, v74, v75])",
            "v80, v81 = sch.sample_perfect_tile(loop=l47, n=2, max_innermost_factor=64, decision=[16, 8])",
            "l82, l83 = sch.split(loop=l47, factors=[v80, v81])",
            "sch.reorder(l52, l60, l68, l76, l53, l61, l69, l77, l82, l54, l62, l70, l78, l83, l55, l63, l71, l79)",
            "sch.reverse_compute_at(block=b42, loop=l77, preserve_unit_loops=True)",
            'b84 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b84, ann_key="auto_parallel_extent", ann_val=64)',
            'sch.annotate(block_or_loop=b84, ann_key="auto_vectorize_extent", ann_val=32)',
            "v85 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=1)",
            'sch.annotate(block_or_loop=b84, ann_key="auto_unroll_explicit", ann_val=v85)',
            'b86 = sch.get_block(name="input_tile", func_name="main")',
            "l87 = sch.sample_compute_location(block=b86, decision=4)",
            "sch.compute_at(block=b86, loop=l87, preserve_unit_loops=True)",
            'b88 = sch.get_block(name="data_pad", func_name="main")',
            "b89, = sch.get_consumers(block=b88)",
            "l90 = sch.sample_compute_location(block=b89, decision=-2)",
            "sch.compute_at(block=b88, loop=l90, preserve_unit_loops=True)",
        ],
    )


def test_meta_schedule_post_order_apply_custom_search_space_winograd_cuda():
    @register_func("tvm.meta_schedule.test.custom_search_space.winograd.cuda")
    def custom_search_space_winograd_func_cuda(sch: Schedule, block: BlockRV) -> List[Schedule]:
        b1 = sch.get_block(name="inverse")
        l2, l3, l4, l5, l6, l7 = sch.get_loops(block=b1)
        sch.unroll(loop=l2)
        sch.unroll(loop=l3)
        sch.unroll(loop=l6)
        sch.unroll(loop=l7)
        v8, v9 = sch.sample_perfect_tile(n=2, loop=l4, max_innermost_factor=64, decision=[3, 3])
        l10, l11 = sch.split(loop=l4, factors=[v8, v9])
        v12, v13 = sch.sample_perfect_tile(n=2, loop=l5, max_innermost_factor=64, decision=[2, 64])
        l14, l15 = sch.split(loop=l5, factors=[v12, v13])
        sch.reorder(l10, l14, l11, l15, l2, l3, l6, l7)
        b16 = sch.get_block(name="data_pack")
        l17, l18, l19, l20, l21, l22 = sch.get_loops(block=b16)
        sch.unroll(loop=l17)
        sch.unroll(loop=l18)
        sch.unroll(loop=l21)
        sch.unroll(loop=l22)
        v23, v24 = sch.sample_perfect_tile(n=2, loop=l19, max_innermost_factor=64, decision=[3, 3])
        l25, l26 = sch.split(loop=l19, factors=[v23, v24])
        v27, v28 = sch.sample_perfect_tile(n=2, loop=l20, max_innermost_factor=64, decision=[64, 2])
        l29, l30 = sch.split(loop=l20, factors=[v27, v28])
        sch.reorder(l25, l29, l26, l30, l17, l18, l21, l22)
        b31 = sch.get_block(name="bgemm")
        b32 = sch.cache_write(block=b31, write_buffer_index=0, storage_scope="local")
        b31, b32 = b32, b31
        l33, l34, l35, l36, l37 = sch.get_loops(block=b32)
        v38, v39, v40, v41, v42 = sch.sample_perfect_tile(
            n=5, loop=l33, max_innermost_factor=64, decision=[1, 1, 1, 1, 6]
        )
        l43, l44, l45, l46, l47 = sch.split(loop=l33, factors=[v38, v39, v40, v41, v42])
        v48, v49, v50, v51, v52 = sch.sample_perfect_tile(
            n=5, loop=l34, max_innermost_factor=64, decision=[1, 1, 1, 3, 2]
        )
        l53, l54, l55, l56, l57 = sch.split(loop=l34, factors=[v48, v49, v50, v51, v52])
        v58, v59, v60, v61, v62 = sch.sample_perfect_tile(
            n=5, loop=l35, max_innermost_factor=64, decision=[3, 1, 1, 1, 3]
        )
        l63, l64, l65, l66, l67 = sch.split(loop=l35, factors=[v58, v59, v60, v61, v62])
        v68, v69, v70, v71, v72 = sch.sample_perfect_tile(
            n=5, loop=l36, max_innermost_factor=64, decision=[4, 2, 1, 4, 4]
        )
        l73, l74, l75, l76, l77 = sch.split(loop=l36, factors=[v68, v69, v70, v71, v72])
        v78, v79, v80 = sch.sample_perfect_tile(
            n=3, loop=l37, max_innermost_factor=64, decision=[32, 1, 4]
        )
        l81, l82, l83 = sch.split(loop=l37, factors=[v78, v79, v80])
        sch.reorder(
            l43,
            l53,
            l63,
            l73,
            l44,
            l54,
            l64,
            l74,
            l45,
            l55,
            l65,
            l75,
            l81,
            l82,
            l46,
            l56,
            l66,
            l76,
            l83,
            l47,
            l57,
            l67,
            l77,
        )
        l84 = sch.fuse(l43, l53, l63, l73)
        sch.bind(loop=l84, thread_axis="blockIdx.x")
        l85 = sch.fuse(l44, l54, l64, l74)
        sch.bind(loop=l85, thread_axis="vthread.x")
        l86 = sch.fuse(l45, l55, l65, l75)
        sch.bind(loop=l86, thread_axis="threadIdx.x")
        b87 = sch.cache_read(block=b32, read_buffer_index=2, storage_scope="shared")
        sch.compute_at(block=b87, loop=l81, preserve_unit_loops=True)
        l88, l89, l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b87)
        l96 = sch.fuse(l92, l93, l94, l95)
        v97, v98 = sch.sample_perfect_tile(
            n=2, loop=l96, max_innermost_factor=4, decision=[1536, 3]
        )
        l99, l100 = sch.split(loop=l96, factors=[v97, v98])
        sch.vectorize(loop=l100)
        sch.annotate(block_or_loop=l99, ann_key="loop_type", ann_val="lazy_cooperative_fetch")
        b101 = sch.cache_read(block=b32, read_buffer_index=1, storage_scope="shared")
        sch.compute_at(block=b101, loop=l81, preserve_unit_loops=True)
        l102, l103, l104, l105, l106, l107, l108, l109 = sch.get_loops(block=b101)
        l110 = sch.fuse(l106, l107, l108, l109)
        v111, v112 = sch.sample_perfect_tile(
            n=2, loop=l110, max_innermost_factor=4, decision=[432, 1]
        )
        l113, l114 = sch.split(loop=l110, factors=[v111, v112])
        sch.vectorize(loop=l114)
        sch.annotate(block_or_loop=l113, ann_key="loop_type", ann_val="lazy_cooperative_fetch")
        sch.reverse_compute_at(block=b31, loop=l86, preserve_unit_loops=True)
        b115 = sch.get_block(name="input_tile")
        (b116,) = sch.get_consumers(block=b115)
        l117, l118, l119, l120, l121, l122, l123, l124 = sch.get_loops(block=b116)
        sch.compute_at(block=b115, loop=l120, preserve_unit_loops=True)
        sch.set_scope(block=b115, buffer_index=0, storage_scope="local")
        b125 = sch.get_block(name="A")
        sch.compute_inline(block=b125)
        b126 = sch.get_block(name="B")
        sch.compute_inline(block=b126)
        b127 = sch.get_block(name="data_pad")
        sch.compute_inline(block=b127)
        b128 = sch.get_block(name="root")
        v129 = sch.sample_categorical(
            candidates=[0, 16, 64, 512, 1024], probs=[0.2, 0.2, 0.2, 0.2, 0.2], decision=0
        )
        sch.annotate(block_or_loop=b128, ann_key="auto_unroll_explicit", ann_val=v129)
        return [sch]

    mod = Conv2d_Winograd_Cuda

    # Add annotation
    sch = Schedule(mod)
    sch.annotate(
        sch.get_block("root"),
        "schedule_rule",
        "tvm.meta_schedule.test.custom_search_space.winograd.cuda",
    )
    mod = sch.mod
    context = TuneContext(
        mod=mod,
        target=Target("nvidia/geforce-rtx-3070"),
        task_name="Custom Search Space Task",
        sch_rules=[DontCallThisRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 1
    (sch,) = schs
    assert str(sch.trace) == "\n".join(
        [
            'b0 = sch.get_block(name="data_pad", func_name="main")',
            'b1 = sch.get_block(name="input_tile", func_name="main")',
            'b2 = sch.get_block(name="B", func_name="main")',
            'b3 = sch.get_block(name="data_pack", func_name="main")',
            'b4 = sch.get_block(name="bgemm", func_name="main")',
            'b5 = sch.get_block(name="A", func_name="main")',
            'b6 = sch.get_block(name="inverse", func_name="main")',
            'b7 = sch.get_block(name="conv2d_winograd", func_name="main")',
            'b8 = sch.get_block(name="root", func_name="main")',
            'b9 = sch.get_block(name="inverse", func_name="main")',
            "l10, l11, l12, l13, l14, l15 = sch.get_loops(block=b9)",
            "sch.unroll(loop=l10)",
            "sch.unroll(loop=l11)",
            "sch.unroll(loop=l14)",
            "sch.unroll(loop=l15)",
            "v16, v17 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64, decision=[3, 3])",
            "l18, l19 = sch.split(loop=l12, factors=[v16, v17])",
            "v20, v21 = sch.sample_perfect_tile(loop=l13, n=2, max_innermost_factor=64, decision=[2, 64])",
            "l22, l23 = sch.split(loop=l13, factors=[v20, v21])",
            "sch.reorder(l18, l22, l19, l23, l10, l11, l14, l15)",
            'b24 = sch.get_block(name="data_pack", func_name="main")',
            "l25, l26, l27, l28, l29, l30 = sch.get_loops(block=b24)",
            "sch.unroll(loop=l25)",
            "sch.unroll(loop=l26)",
            "sch.unroll(loop=l29)",
            "sch.unroll(loop=l30)",
            "v31, v32 = sch.sample_perfect_tile(loop=l27, n=2, max_innermost_factor=64, decision=[3, 3])",
            "l33, l34 = sch.split(loop=l27, factors=[v31, v32])",
            "v35, v36 = sch.sample_perfect_tile(loop=l28, n=2, max_innermost_factor=64, decision=[64, 2])",
            "l37, l38 = sch.split(loop=l28, factors=[v35, v36])",
            "sch.reorder(l33, l37, l34, l38, l25, l26, l29, l30)",
            'b39 = sch.get_block(name="bgemm", func_name="main")',
            'b40 = sch.cache_write(block=b39, write_buffer_index=0, storage_scope="local")',
            "l41, l42, l43, l44, l45 = sch.get_loops(block=b39)",
            "v46, v47, v48, v49, v50 = sch.sample_perfect_tile(loop=l41, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 6])",
            "l51, l52, l53, l54, l55 = sch.split(loop=l41, factors=[v46, v47, v48, v49, v50])",
            "v56, v57, v58, v59, v60 = sch.sample_perfect_tile(loop=l42, n=5, max_innermost_factor=64, decision=[1, 1, 1, 3, 2])",
            "l61, l62, l63, l64, l65 = sch.split(loop=l42, factors=[v56, v57, v58, v59, v60])",
            "v66, v67, v68, v69, v70 = sch.sample_perfect_tile(loop=l43, n=5, max_innermost_factor=64, decision=[3, 1, 1, 1, 3])",
            "l71, l72, l73, l74, l75 = sch.split(loop=l43, factors=[v66, v67, v68, v69, v70])",
            "v76, v77, v78, v79, v80 = sch.sample_perfect_tile(loop=l44, n=5, max_innermost_factor=64, decision=[4, 2, 1, 4, 4])",
            "l81, l82, l83, l84, l85 = sch.split(loop=l44, factors=[v76, v77, v78, v79, v80])",
            "v86, v87, v88 = sch.sample_perfect_tile(loop=l45, n=3, max_innermost_factor=64, decision=[32, 1, 4])",
            "l89, l90, l91 = sch.split(loop=l45, factors=[v86, v87, v88])",
            "sch.reorder(l51, l61, l71, l81, l52, l62, l72, l82, l53, l63, l73, l83, l89, l90, l54, l64, l74, l84, l91, l55, l65, l75, l85)",
            "l92 = sch.fuse(l51, l61, l71, l81)",
            'sch.bind(loop=l92, thread_axis="blockIdx.x")',
            "l93 = sch.fuse(l52, l62, l72, l82)",
            'sch.bind(loop=l93, thread_axis="vthread.x")',
            "l94 = sch.fuse(l53, l63, l73, l83)",
            'sch.bind(loop=l94, thread_axis="threadIdx.x")',
            'b95 = sch.cache_read(block=b39, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b95, loop=l89, preserve_unit_loops=True)",
            "l96, l97, l98, l99, l100, l101, l102, l103 = sch.get_loops(block=b95)",
            "l104 = sch.fuse(l100, l101, l102, l103)",
            "v105, v106 = sch.sample_perfect_tile(loop=l104, n=2, max_innermost_factor=4, decision=[1536, 3])",
            "l107, l108 = sch.split(loop=l104, factors=[v105, v106])",
            "sch.vectorize(loop=l108)",
            'sch.annotate(block_or_loop=l107, ann_key="loop_type", ann_val="lazy_cooperative_fetch")',
            'b109 = sch.cache_read(block=b39, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b109, loop=l89, preserve_unit_loops=True)",
            "l110, l111, l112, l113, l114, l115, l116, l117 = sch.get_loops(block=b109)",
            "l118 = sch.fuse(l114, l115, l116, l117)",
            "v119, v120 = sch.sample_perfect_tile(loop=l118, n=2, max_innermost_factor=4, decision=[432, 1])",
            "l121, l122 = sch.split(loop=l118, factors=[v119, v120])",
            "sch.vectorize(loop=l122)",
            'sch.annotate(block_or_loop=l121, ann_key="loop_type", ann_val="lazy_cooperative_fetch")',
            "sch.reverse_compute_at(block=b40, loop=l94, preserve_unit_loops=True)",
            'b123 = sch.get_block(name="input_tile", func_name="main")',
            "b124, = sch.get_consumers(block=b123)",
            "l125, l126, l127, l128, l129, l130, l131, l132 = sch.get_loops(block=b124)",
            "sch.compute_at(block=b123, loop=l128, preserve_unit_loops=True)",
            'sch.set_scope(block=b123, buffer_index=0, storage_scope="local")',
            'b133 = sch.get_block(name="A", func_name="main")',
            "sch.compute_inline(block=b133)",
            'b134 = sch.get_block(name="B", func_name="main")',
            "sch.compute_inline(block=b134)",
            'b135 = sch.get_block(name="data_pad", func_name="main")',
            "sch.compute_inline(block=b135)",
            'b136 = sch.get_block(name="root", func_name="main")',
            "v137 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=0)",
            'sch.annotate(block_or_loop=b136, ann_key="auto_unroll_explicit", ann_val=v137)',
        ]
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))

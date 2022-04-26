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
"""Integration test for CUDA with Tensor Core"""
# pylint: disable=missing-function-docstring
import pytest
import tvm.meta_schedule.testing.te_workload as te_workload
import tvm
from tvm import te, tir
from tvm.script import tir as T
import tvm.testing
import numpy as np
import os
from tvm.contrib import nvcc
import sys

TARGET = tvm.target.Target("nvidia/geforce-rtx-3070")

TASK = "gemm"
USE_MANUAL_CODE = False


@T.prim_func
def mma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [32, 4], dtype="float16", scope="warp")
    B = T.match_buffer(b, [32, 2], dtype="float16", scope="warp")
    C = T.match_buffer(c, [32, 4], dtype="float32", scope="warp")
    with T.block("root"):
        T.reads(C[0 : 32, 0 : 4], A[0 : 32, 0 : 4], B[0 : 32, 0 : 2])
        T.writes(C[0 : 32, 0 : 4])
        for i0, i1, i2 in T.grid(16, 8, 8):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                # T.reads(C[i % 8 * 4 + j // 2, i // 8 * 2 + j % 2], A[i % 8 * 4 + k // 2, i // 8 * 2 + k % 2], B[j * 4 + k // 2, k % 2])
                # T.writes(C[i % 8 * 4 + j // 2, i // 8 * 2 + j % 2])
                # C[i % 8 * 4 + j // 2, i // 8 * 2 + j % 2] = C[i % 8 * 4 + j // 2, i // 8 * 2 + j % 2] + T.cast(A[i % 8 * 4 + k // 2, i // 8 * 2 + k % 2], "float32") * T.cast(B[j * 4 + k // 2, k % 2], "float32")

                T.reads(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2], A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2], B[j * 4 + k // 2, k % 2])
                T.writes(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2])
                C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] = C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] + T.cast(A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2], "float32") * T.cast(B[j * 4 + k // 2, k % 2], "float32")


                # T.reads(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2], A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2], B[k % 8 * 4 + j % 8 // 2, j // 8 * 4 + k // 8 * 2 + j % 2])
                # T.writes(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2])
                # C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] = C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] + T.cast(A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2], "float32") * T.cast(B[k % 8 * 4 + j % 8 // 2, j // 8 * 4 + k // 8 * 2 + j % 2], "float32")




@T.prim_func
def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32, 4), "float16", align=128, offset_factor=1, scope="warp")
    B = T.match_buffer(b, (32, 2), "float16", align=128, offset_factor=1, scope="warp")
    C = T.match_buffer(c, (32, 4), "float32", align=128, offset_factor=1, scope="warp")

    with T.block("root"):
        T.reads(C[0 : 32, 0 : 4], A[0 : 32, 0 : 4], B[0 : 32, 0 : 2])
        T.writes(C[0 : 32, 0 : 4])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)
        T.evaluate(T.call_extern("mma_m16n8k8_row_row_fp16fp16fp32", A.data, A.elem_offset + tx * 4, B.data, B.elem_offset + tx * 2, C.data, C.elem_offset + tx * 4, dtype="float32",))


MMA_SYNC = tir.TensorIntrin.register(
    "mma_sync",
    mma_sync_desc,
    mma_sync_impl,
)


def test_integration_matmul():
    N = 16
    M = 8
    K = 8
    workload = te_workload.matmul_fp16(n=N, m=M, k=K)
    workload = te.create_prim_func(workload)

    # a = mk b = nk / kk c = nn
    # scope = local
    def schedule(sch: tir.Schedule):
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)

        # Step 2. Rule-Multi-Level-Tiling
        i1, i2 = sch.split(i, factors=[None, 16])
        sch.bind(i1, "blockIdx.x")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, i1)
            warp_size = 32
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
            f_0, f_1 = sch.split(fused, factors=[None, warp_size])
            sch.bind(f_1, "threadIdx.x")

        fetch_to_shared(block, 0, 2)
        fetch_to_shared(block, 1, 2)

        # fetch to A_warp 16 * 8 -> 32 * 4
        A_warp = sch.cache_read(block, 0, "warp")
        sch.transform_layout(
            A_warp,
            0,
            "write",
            # index_map=lambda i, j: ((i % 8) * 4 + j // 2, (i // 8) * 2 + j % 2),
            index_map=lambda i, j: ((i % 8) * 4 + (j % 8) // 2, 4 * (j // 8) + (i // 8) * 2 + (j % 8) % 2)
        )

        warp_loop1, warp_loop2 = sch.get_loops(A_warp)[-2:]
        f_0, f_1 = sch.split(warp_loop1, factors=[None, 8])
        f_2, f_3 = sch.split(warp_loop2, factors=[None, 2])
        sch.reorder(f_1, f_2, f_0, f_3)
        fused_1 = sch.fuse(f_1, f_2)
        fused_2 = sch.fuse(f_0, f_3)
        print(fused_1)
        sch.bind(fused_1, "threadIdx.x")

        # f_0, f_1 = sch.split(warp_loop1, factors=[None, 4])
        # sch.reorder(f_1, warp_loop2, f_0)
        # fused_1 = sch.fuse(warp_loop2, f_0)
        # sch.bind(fused_1, "threadIdx.x")

        # fetch to B_warp 8 * 8 -> 32 * 2
        B_warp = sch.cache_read(block, 1, "warp")
        sch.transform_layout(
            B_warp,
            0,
            "write",
            index_map=lambda i, j: (i // 2 + j * 4, i % 2),
            # index_map=lambda i, j: ((i % 8) * 4 + (j % 8) // 2, 4 * (j // 8) + (i // 8) * 2 + (j % 8) % 2)
        )
        warp_loop1, warp_loop2 = sch.get_loops(B_warp)[-2:]
        f_0, f_1 = sch.split(warp_loop1, factors=[4, 2])
        sch.reorder(warp_loop2, f_0, f_1)
        fused_1 = sch.fuse(warp_loop2, f_0)
        sch.bind(fused_1, "threadIdx.x")

        # fetch to C_warp 16 * 8 -> 32 * 4
        C_warp = sch.cache_write(block, 0, "warp")
        sch.reverse_compute_at(C_warp, sch.get_loops(block)[0])
        # need to do a reverse_compute_at to place it under blockidx.x
        sch.transform_layout(
            C_warp,
            0,
            "read",
            # index_map=lambda i, j: ((i % 8) * 4 + j // 2, (i // 8) * 2 + j % 2),
            index_map=lambda i, j: ((i % 8) * 4 + (j % 8) // 2, 4 * (j // 8) + (i // 8) * 2 + (j % 8) % 2)
        )
        warp_loop1, warp_loop2 = sch.get_loops(C_warp)[-2:]
        f_0, f_1 = sch.split(warp_loop1, factors=[None, 8])
        f_2, f_3 = sch.split(warp_loop2, factors=[None, 2])
        sch.reorder(f_1, f_2, f_0, f_3)
        fused_1 = sch.fuse(f_1, f_2)
        fused_2 = sch.fuse(f_0, f_3)
        sch.bind(fused_1, "threadIdx.x")

        # Decompose -> separate C_init from C_warp
        loop = sch.get_loops(block)[1]
        block_init_c = sch.decompose_reduction(block, loop)

        # C_init() 16 * 8 -> 32 * 4
        # as binding is already transformed by previous step
        # only split/reorder/fuse is needed here
        C_init = block_init_c
        init_loop1, init_loop2 = sch.get_loops(C_init)[-2:]
        f_0, f_1 = sch.split(init_loop1, factors=[None, 8])
        f_2, f_3 = sch.split(init_loop2, factors=[None, 2])
        sch.reorder(f_1, f_2, f_0, f_3)
        fused_1 = sch.fuse(f_1, f_2)
        fused_2 = sch.fuse(f_0, f_3)
        sch.bind(fused_1, "threadIdx.x")

        # tensorize
        i0, i1, i2, i3 = sch.get_loops(block)
        sch.tensorize(i1, "mma_sync")

    sch = tir.Schedule(workload)
    schedule(sch)

    if sch is None:
        print("No valid schedule found")
    else:
        print(sch.mod["main"].script())
        # print(tvm.lower(sch.mod["main"], None, simple_mode=True))

    target = "vulkan -from_device=0"
    f = tvm.build(sch.mod["main"], target=target, name="dense")
    # dev = tvm.device("cuda", 0)
    # a_np = np.random.uniform(size=(N, K)).astype("float16")
    # b_np = np.random.uniform(size=(K, M)).astype("float16")
    # c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
    # a = tvm.nd.array(a_np, dev)
    # b = tvm.nd.array(b_np, dev)
    # c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
    # # sys.exit(0)
    # f = tvm.build(sch.mod["main"], target="cuda", name="dense")
    # f(a, b, c)
    # print(f.imported_modules[0].get_source())
    # tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    # evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    # gflops = (N * M * K) * 2 / 1e9
    # time_ms = evaluator(a, b, c).mean * 1e3
    # print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


if __name__ == "__main__":
    test_integration_matmul()

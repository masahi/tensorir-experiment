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
import tempfile
import tvm
from tvm.script import tir as T
from tvm import te, tir
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing import te_workload
import tvm.testing
import numpy as np
import os
from tvm.contrib import nvcc
import sys
import tir_tensor_intrin

TARGET = tvm.target.Target("nvidia/geforce-rtx-3070")


N = 512
M = 512
K = 512
workload = te_workload.matmul_fp16(n=N, m=M, k=K)
workload = te.create_prim_func(workload)

def schedule(sch: tir.Schedule):
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    # Step 1. Rule-Auto-Tensorize
    # pylint: disable=invalid-name
    i, i_tc = sch.split(i, factors=[None, 16])
    j, j_tc = sch.split(j, factors=[None, 16])
    k, k_tc = sch.split(k, factors=[None, 16])
    sch.reorder(
        # fmt: off
        i, j, k,
        # tensor core
        i_tc, j_tc, k_tc,
        # fmt: on
    )
    # print("before blokize")
    # print(sch.mod.script())
    block_inner = sch.blockize(i_tc)
    # print(sch.get(block))

    block_outer, block_inner = block_inner, block

    # print("after blokize")
    # print(sch.mod.script())
    # print(block, block_inner)


    del block
    # Step 2. Rule-Multi-Level-Tiling
    i_factors = sch.sample_perfect_tile(i, n=5)
    j_factors = sch.sample_perfect_tile(j, n=5)
    k_factors = sch.sample_perfect_tile(k, n=3)
    i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
    j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
    k0, k1, k2 = sch.split(k, k_factors)
    # pylint: enable=invalid-name
    sch.reorder(
        # fmt: off
        i0, j0,   # S => blockIdx.x
        i1, j1,   # S => blockIdx.y
        j2, i2,   # S => threadIdx.y
        # cache_write here
        k0,       # R
        # vectorized cooperative fetching here
        k1,       # R
        i3, j3,   # S
        k2,       # R
        i4, j4,
        # S
        # fmt: on
    )

    block_idx = sch.fuse(i0, j0)
    block_idy = sch.fuse(i1, j1)
    thread_idy = sch.fuse(j2, i2)
    sch.bind(block_idx, "blockIdx.x")
    sch.bind(block_idy, "blockIdx.y")
    sch.bind(thread_idy, "threadIdx.y")

    num_ty = sch.get(i_factors[2]) * sch.get(j_factors[2])
    # num_ty = i_factors[2] * j_factors[2]  # NOT SUPPORTED
    def fetch_to_shared(block, idx, ndim):
        block_read = sch.cache_read(block, idx, "shared")
        sch.compute_at(block_read, k0)
        vector_size = 8
        warp_size = 32
        fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
        f_0, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])
        sch.bind(f_2, 'threadIdx.x')
        sch.bind(f_1, 'threadIdx.y')
        sch.vectorize(f_3)

        sch.storage_align(block_read, 0, axis=-2, factor=32, offset=8)
        return block_read


    A_sh = fetch_to_shared(block_outer, 1, 2)

    B_sh = fetch_to_shared(block_outer, 2, 2)

    # Step 3. Postproc-Rewrite-Tensorize
    # Step 3.1. Cache read
    loop = sch.get_loops(block_outer)[-1]
    # print(block_outer.reads)
    # print(sch.get(block_outer))
    block_read_a = sch.cache_read(block_outer, 1, "wmma.matrix_a")
    block_read_b = sch.cache_read(block_outer, 2, "wmma.matrix_b")
    sch.compute_at(block_read_a, k1)
    sch.compute_at(block_read_b, k1)

    # print(sch.get(k1))
    # Step 3.2. Cache write
    block_write_c = sch.cache_write(block_outer, 0, "wmma.accumulator")
    # block_outer, block_write_c = block_write_c, block_outer
    sch.reverse_compute_at(block_write_c, thread_idy)
    # Wuwei: we also need spliting the write back stage.
    ii, jj = sch.get_loops(block_write_c)[-2:]
    io, ii = sch.split(ii, factors=[None, 16])
    jo, ji = sch.split(jj, factors=[None, 16])
    sch.reorder(io, jo, ii, ji)
    # Step 3.3. Decompose
    loop = sch.get_loops(block_outer)[3]
    block_init_c = sch.decompose_reduction(block_outer, loop)
    block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

    # Step 3.4. Tensorize

    loop = sch.get_loops(block_inner)[-3]

    # print(sch.get(block_inner))
    # print(tvm.script.asscript(sch.mod['main']))
    def tile_wmma_fragment(block_read):
        i, j = sch.get_loops(block_read)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)
        return i1

    print("before tensorize")
    print(sch.mod.script())

    sch.tensorize(loop, "wmma.mma_sync")
    loop = tile_wmma_fragment(block_read_a)
    sch.tensorize(loop, "wmma.load_matrix_a")
    loop = tile_wmma_fragment(block_read_b)
    sch.tensorize(loop, "wmma.load_matrix_b")

    print("after tensorize")
    print(sch.mod.script())

    loop = sch.get_loops(block_init_c_inner)[-2]
    sch.tensorize(loop, "wmma.fill")
    loop = sch.get_loops(block_write_c)[-2]
    sch.tensorize(loop, "wmma.store")


# print(workload)
ir_module = tvm.IRModule({"main": workload})
sch = tvm.tir.Schedule(ir_module)
schedule(sch)
# print(sch.mod.script())
# schedule(workload)
# with tempfile.TemporaryDirectory() as work_dir:
#     sch = ms.tune_tir(
#         mod=workload,
#         target=tvm.target.Target("nvidia/geforce-rtx-3070"),
#         # use replay or evolutionary search
#         config=ms.ReplayTraceConfig(
#             num_trials_per_iter=32,
#             num_trials_total=32,
#         ),
#         work_dir=work_dir,
#         space=ms.space_generator.ScheduleFn(schedule)
#         )
#     if sch is None:
#         print("No valid schedule found!")
#     else:
#         print(sch.mod.script())
#         print(sch.trace)

# dev = tvm.device("cuda", 0)
# a_np = np.random.uniform(size=(N, K)).astype("float16")
# b_np = np.random.uniform(size=(K, M)).astype("float16")
# c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
# a = tvm.nd.array(a_np, dev)
# b = tvm.nd.array(b_np, dev)
# c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
# f = tvm.build(sch.mod['main'], target="cuda", name="dense")
# print(f.imported_modules[0].get_source())
# f(a, b, c)
# tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

# evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
# gflops = (N*M*K) * 2 / 1e9
# time_ms = evaluator(a, b, c).mean * 1e3
# print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))

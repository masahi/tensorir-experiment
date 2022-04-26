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
import tvm
from tvm import te, tir
from tvm.script import tir as T
import tvm.testing
import numpy as np


@T.prim_func
def ldmatrix_a_desc(a: T.handle, c: T.handle) -> None:
    A_shared = T.match_buffer(
        a, (16, 8), "float16", align=128, offset_factor=16, scope="shared"
    )
    A_warp = T.match_buffer(
        c, (32, 4), "float16", align=128, offset_factor=16, scope="warp"
    )

    with T.block("root"):
        T.reads(A_shared[0:16, 0:8])
        T.writes(A_warp[0:32, 0:4])

        for ax0, ax1 in T.grid(16, 8):
            with T.block("A_shared_warp"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A_shared[v0, v1])
                T.writes(A_warp[v0 % 8 * 4 + v1 // 2, v0 // 8 * 2 + v1 % 2])
                A_warp[v0 % 8 * 4 + v1 // 2, v0 // 8 * 2 + v1 % 2] = A_shared[v0, v1]

@T.prim_func
def ldmatrix_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A_shared = T.match_buffer(
        a,
        (16, 8),
        "float16",
        align=128,
        offset_factor=16,
        scope="shared",
        strides=[s1, s0],
    )
    A_warp = T.match_buffer(
        c, (32, 4), "float16", align=128, offset_factor=16, scope="warp"
    )
    with T.block("root"):
        T.reads(A_shared[0:16, 0:8])
        T.writes(A_warp[0:32, 0:4])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        T.evaluate(
            T.ptx_ldmatrix(
                0,
                2,
                ".b16",
                A_warp.data,
                4 * tx,
                A_shared.data,
                8 * (tx % 16),
                dtype="float16",
            )
        )


@T.prim_func
def ldmatrix_b_desc(a: T.handle, c: T.handle) -> None:
    B_shared = T.match_buffer(
        a, (8, 8), "float16", align=128, offset_factor=16, scope="shared"
    )
    B_shared_warp = T.match_buffer(
        c, (32, 2), "float16", align=128, offset_factor=16, scope="warp"
    )

    with T.block("root"):
        T.reads(B_shared[0:8, 0:8])
        T.writes(B_shared_warp[0:32, 0:2])

        for ax0, ax1 in T.grid(8, 8):
            with T.block("A_shared_warp"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(B_shared[v0, v1])
                T.writes(B_shared_warp[v1 * 4 + v0 // 2, v0 % 2])
                B_shared_warp[v1 * 4 + v0 // 2, v0 % 2] = B_shared[v0, v1]


@T.prim_func
def ldmatrix_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    B_shared = T.match_buffer(
        a,
        (8, 8),
        "float16",
        align=128,
        offset_factor=16,
        scope="shared",
        strides=[s1, s0],
    )
    B_warp = T.match_buffer(
        c, (32, 2), "float16", align=128, offset_factor=16, scope="warp"
    )
    with T.block("root"):
        T.reads(B_shared[0:8, 0:8])
        T.writes(B_warp[0:32, 0:2])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        T.evaluate(
            T.ptx_ldmatrix(
                0,
                1,
                ".b16",
                B_warp.data,
                2 * tx,
                B_shared.data,
                8 * (tx % 8),
                dtype="float16",
            )
        )


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

                T.reads(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2], A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2], B[k * 4 + j // 2, j % 2])
                T.writes(C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2])
                C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] = C[i % 8 * 4 + j % 8 // 2, j // 8 * 4 + i // 8 * 2 + j % 2] + T.cast(A[i % 8 * 4 + k % 8 // 2, k // 8 * 4 + i // 8 * 2 + k % 2], "float32") * T.cast(B[k * 4 + j // 2, j % 2], "float32")


@T.prim_func
def mma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (32, 4), "float16", align=128, offset_factor=1, scope="warp")
    B = T.match_buffer(b, (32, 2), "float16", align=128, offset_factor=1, scope="warp")
    C = T.match_buffer(c, (32, 4), "float32", align=128, offset_factor=1, scope="warp")

    with T.block("root"):
        T.reads(C[0:32, 0:4], A[0:32, 0:4], B[0:32, 0:2])
        T.writes(C[0:32, 0:4])
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)
        T.evaluate(
            T.ptx_mma(
                "m16n8k8",
                "row",
                "col",
                "fp16",
                "fp16",
                "fp32",
                A.data,
                A.elem_offset + tx * 4,
                B.data,
                B.elem_offset + tx * 2,
                C.data,
                C.elem_offset + tx * 4,
                False,
                dtype="float32",
            )
        )


tir.TensorIntrin.register("mma.ldmatrix_a", ldmatrix_a_desc, ldmatrix_a_impl)
tir.TensorIntrin.register("mma.ldmatrix_b", ldmatrix_b_desc, ldmatrix_b_impl)
tir.TensorIntrin.register("mma_sync", mma_sync_desc, mma_sync_impl)


def dense(n: int, m: int, k: int):
    a = te.placeholder((n, k), name="A", dtype="float16")
    b = te.placeholder((m, k), name="B", dtype="float16")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(
            tvm.tir.Cast("float32", a[i, k]) * tvm.tir.Cast("float32", b[j, k]),
            axis=[k],
        ),
        name="C",
    )
    return (a, b, c)


def test_integration_matmul():
    N = 4096
    M = 4096
    K = 4096

    workload = te.create_prim_func(dense(n=N, m=M, k=K))

    def schedule(sch: tir.Schedule):
        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)

        i, i_tc = sch.split(i, factors=[None, 16])
        j, j_tc = sch.split(j, factors=[None, 8])
        k_outer, k_tc = sch.split(k, factors=[None, 8])

        sch.reorder(
            # fmt: off
            i, j, k_outer,
            # tensor core
            i_tc, j_tc, k_tc
        )

        # block_inner = sch.blockize(i_tc)

        sch.bind(sch.fuse(i, j), "blockIdx.x")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, k_outer)
            warp_size = 32
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
            f_0, f_1 = sch.split(fused, factors=[None, warp_size])
            sch.bind(f_1, "threadIdx.x")

        fetch_to_shared(block, 0, 2)
        fetch_to_shared(block, 1, 2)

        # fetch to A_warp 16 * 8 -> 32 * 4
        A_warp = sch.cache_read(block, 0, "warp")

        def lambda_a(i, j):
            i_0 = i // 16
            j_0 = j // 8

            i = i % 16
            j = j % 8
            return i_0, j_0, (i % 8) * 4 + (j % 8) // 2, 4 * (j // 8) + (i // 8) * 2 + (j % 8) % 2,

        sch.transform_layout(
            A_warp,
            0,
            "write",
            index_map=lambda_a
        )

        sch.tensorize(sch.get_loops(A_warp)[2], "mma.ldmatrix_a")

        def lambda_b(i, j):
            i_0 = i // 8
            j_0 = j // 8
            i = i % 8
            j = j % 8
            return i_0, j_0, i // 2 + j * 4, i % 2

        B_warp = sch.cache_read(block, 1, "warp")
        sch.transform_layout(
            B_warp,
            0,
            "write",
            index_map=lambda_b,
        )
        sch.tensorize(sch.get_loops(B_warp)[2], "mma.ldmatrix_b")

        # fetch to C_warp 16 * 8 -> 32 * 4
        C_warp = sch.cache_write(block, 0, "warp")
        sch.reverse_compute_at(C_warp, sch.get_loops(block)[0])
        # need to do a reverse_compute_at to place it under blockidx.x
        sch.transform_layout(
            C_warp,
            0,
            "read",
            index_map=lambda_a,
        )

        warp_loop1, warp_loop2 = sch.get_loops(C_warp)[-2:]
        f_0, f_1 = sch.split(warp_loop1, factors=[None, 8])
        f_2, f_3 = sch.split(warp_loop2, factors=[None, 2])
        sch.reorder(f_1, f_2, f_0, f_3)
        fused_1 = sch.fuse(f_1, f_2)
        fused_2 = sch.fuse(f_0, f_3)
        sch.bind(fused_1, "threadIdx.x")

        block_init_c = sch.decompose_reduction(block, sch.get_loops(block)[1])
        init_loop1, init_loop2 = sch.get_loops(block_init_c)[-2:]
        f_0, f_1 = sch.split(init_loop1, factors=[None, 8])
        f_2, f_3 = sch.split(init_loop2, factors=[None, 2])
        sch.reorder(f_1, f_2, f_0, f_3)
        fused_1 = sch.fuse(f_1, f_2)
        fused_2 = sch.fuse(f_0, f_3)
        sch.bind(fused_1, "threadIdx.x")

        block = sch.get_block("C_update")
        # tensorize
        i1, _, _ = sch.get_loops(block)[-3:]

        print(sch.get(i1))
        sch.tensorize(i1, "mma_sync")

        # return


    sch = tir.Schedule(workload)
    schedule(sch)

    print(sch.mod["main"].script())
    return

    target = "cuda"
    f = tvm.build(sch.mod["main"], target=target, name="dense")
    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(N, K)).astype("float16")
    b_np = np.random.uniform(size=(M, K)).astype("float16")
    c_np = np.dot(a_np.astype("float32"), b_np.transpose().astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
    # sys.exit(0)
    f = tvm.build(sch.mod["main"], target="cuda", name="dense")
    f(a, b, c)
    print(f.imported_modules[0].get_source())
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)


if __name__ == "__main__":
    test_integration_matmul()

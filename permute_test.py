import numpy as np


VEC_SIZE = 8


def ref():
    res = np.zeros((8, VEC_SIZE * 8), dtype="int32")

    for i in range(VEC_SIZE * 4):
        for j in range(16):
            row_id = (i // VEC_SIZE)
            id_in_row = i % VEC_SIZE
            half_column_id = j // 8
            lane_id = (row_id % 2) * 16 + j
            dst_row = (lane_id % 8) // 2

            dst_row_offset = dst_row + half_column_id * 4

            if row_id == 0:
                dst_col = dst_row
            elif row_id == 1:
                dst_col = dst_row ^ 1
            elif row_id == 2:
                dst_col = dst_row ^ 2
            elif row_id == 3:
                dst_col = (dst_row ^ 2) ^ 1

            dst_col_offset = dst_col * VEC_SIZE + id_in_row + (lane_id % 2) * VEC_SIZE * 4

            res[dst_row_offset, dst_col_offset] = lane_id

    return res


def ldmatrix_permuted_layout():
    res = np.zeros((8, VEC_SIZE * 8), dtype="int32")

    for i in range(VEC_SIZE * 4):
        for j in range(16):
            row_id = (i // VEC_SIZE)
            lane_id = (row_id % 2) * 16 + j
            dst_row = (lane_id % 8) // 2
            dst_col = dst_row ^ row_id

            dst_row_offset = dst_row + (j // 8) * 4
            dst_col_offset = dst_col * VEC_SIZE + (i % VEC_SIZE) + (lane_id % 2) * VEC_SIZE * 4

            res[dst_row_offset, dst_col_offset] = lane_id

    return res


def ldmatrix_permuted_layout2():
    res = np.zeros((8, VEC_SIZE * 8), dtype="int32")

    for lane_id in range(32):
        for row_pair_id in range(2):
            row_id = row_pair_id * 2 + (lane_id // 16)
            dst_row_in_quad = (lane_id % 8) // 2
            dst_col_in_quad = dst_row_in_quad ^ row_id
            quad_hori_id = lane_id % 2
            quad_vert_id = (lane_id % 16) // 8

            dst_row_offset = quad_vert_id * 4 + dst_row_in_quad
            dst_col_offset = (quad_hori_id * 4 + dst_col_in_quad) * VEC_SIZE

            for i in range(8):
                res[dst_row_offset, dst_col_offset + i] = lane_id

    return res


np.testing.assert_equal(ref(), ldmatrix_permuted_layout())
np.testing.assert_equal(ref(), ldmatrix_permuted_layout2())


def shared_store_permuted_layout():
    res = np.zeros((8, VEC_SIZE * 8), dtype="int32")

    for i in range(8):
        for j in range(VEC_SIZE * 8):
            lane_id = j // VEC_SIZE + (i % 4) * 8
            col_id = (lane_id % 8) ^ (i % 4)
            res[i, col_id * VEC_SIZE + j % VEC_SIZE] = lane_id

    return res


def shared_store_permuted_layout2():
    res = np.zeros((8, VEC_SIZE * 8), dtype="int32")

    for i in range(16):
        for j in range(32):
            lane_id = 4 * (i % 8) + j // VEC_SIZE
            col_id = (lane_id % 8) ^ ((i // 2) % 4)
            res[i // 2, col_id * VEC_SIZE + j % VEC_SIZE] = lane_id

    return res


res = shared_store_permuted_layout()
res2 = shared_store_permuted_layout2()

np.testing.assert_equal(res, res2)

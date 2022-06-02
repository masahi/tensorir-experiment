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


np.testing.assert_equal(ref(), ldmatrix_permuted_layout())


def shared_store_permuted_layout():
    res = np.zeros((8, VEC_SIZE * 8), dtype="int32")

    for i in range(8):
        for j in range(VEC_SIZE * 8):
            lane_id = j // VEC_SIZE + (i % 4) * 8
            col_id = (lane_id % 8) ^ (i % 4)
            res[i, col_id * VEC_SIZE + j % VEC_SIZE] = lane_id

    return res


res = shared_store_permuted_layout()

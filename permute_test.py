import numpy as np


VEC_SIZE = 8


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

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 8

#define TM 8
#define TN SG_SZ
#define TK 16

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T1, typename T2, size_t NUM_ROWS_A, size_t NUM_COLS_A,
          size_t NUM_ROWS_B, size_t NUM_COLS_B, size_t NUM_ROWS_C,
          size_t NUM_COLS_C>
void matrix_multiply(big_matrix<T1, NUM_ROWS_C, NUM_COLS_C> &C,
                     big_matrix<T2, NUM_ROWS_A, NUM_COLS_A> &A,
                     big_matrix<T2, NUM_ROWS_B, NUM_COLS_B> &B) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;

  assert(NUM_ROWS_C == NUM_ROWS_A && NUM_COLS_A == NUM_ROWS_B * 2);
  buffer<half, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<half, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<float, 2> bufC(C.get_data(), range<2>(M, N));

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);
     local_accessor<half, 1> slm_A(NUM_ROWS_A * NUM_COLS_A, cgh);
     local_accessor<half, 1> slm_B(NUM_ROWS_B * NUM_COLS_B, cgh);

     cgh.parallel_for<class imatrix>(
         nd_range<2>({2, SG_SZ}, {2, SG_SZ}),
         [accA, accB, slm_A, slm_B, accC, M, N,
          K](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);

           const auto sg_startx = spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

	   for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ax0_ax1_fused_0++) {
	       for (int ax0_ax1_fused_3 = 0; ax0_ax1_fused_3 < 8; ax0_ax1_fused_3++) {
		 slm_A[((((ax0_ax1_fused_0*128) + (spmd_item.get_local_id(0)*64)) + (spmd_item.get_local_id(1)*8)) + ax0_ax1_fused_3)] = accA.get_pointer()[((((ax0_ax1_fused_0*128) + (spmd_item.get_local_id(0)*64)) + (spmd_item.get_local_id(1)*8)) + ax0_ax1_fused_3)];
               }
	   }

	   for (int ax0_ax1_ax2_fused_3 = 0; ax0_ax1_ax2_fused_3 < 8; ++ax0_ax1_ax2_fused_3) {
	     slm_B[(((spmd_item.get_local_id(0)*64) + (spmd_item.get_local_id(1)*8)) + ax0_ax1_ax2_fused_3)] = accB.get_pointer()[(((spmd_item.get_local_id(0)*64) + (spmd_item.get_local_id(1) *8)) + ax0_ax1_ax2_fused_3)];
	   }

	   spmd_item.barrier(access::fence_space::local_space);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<half, TM, TK> sub_a(sg);
           joint_matrix<half, TK, TN, matrix_layout::packed_b> sub_b(sg);
           joint_matrix<float, TM, TN> sub_c(sg);

           joint_matrix_fill(sg, sub_c, 0);
	   joint_matrix_load(sg, sub_a, slm_A.get_pointer() + (sg_startx * TM) * K, K, matrix_layout::row_major);
	   joint_matrix_load(sg, sub_b, slm_B.get_pointer(), N * 2, matrix_layout::packed_b);
	   sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           joint_matrix_store(sg, sub_c, accC.get_pointer() + (sg_startx * TM) * N, N, matrix_layout::row_major);
         }); // parallel for
   }).wait();
}

static constexpr size_t MATRIX_M = 16;
static constexpr size_t MATRIX_N = 8;
static constexpr size_t MATRIX_K = 16;
half A[MATRIX_M][MATRIX_K];
half B[MATRIX_K / 2][MATRIX_N * 2];
float C[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

void matrix_multiply_ref(float *A_mem, float *B_mem, float *C_mem, int M, int N,
                         int K) {
  // tiling
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float acc = 0.0;
      for (int k = 0; k < K; k++) {
        half *va = (half *)(A_mem + m * K + k);
        half *vb = (half *)(B_mem + k * N + n);
        for (int i = 0; i < 2; i++) {
          acc += ((float)va[i] * (float)vb[i]);
        }
      }
      *((float *)(C_mem + m * N + n)) = acc;
    }
}

int main() {
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = i + 2 * j;
    }
  }
  for (int i = 0; i < MATRIX_K / 2; i++) {
    for (int j = 0; j < MATRIX_N * 2; j++) {
      B[i][j] = i + j;
    }
  }

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<half, MATRIX_M, MATRIX_K> MA((half *)&A);
  big_matrix<half, MATRIX_K / 2, MATRIX_N * 2> MB((half *)&B);
  matrix_multiply(MC, MA, MB);
  matrix_multiply_ref((float *)A, (float *)B, (float *)D, MATRIX_M, MATRIX_N,
                      MATRIX_K / 2);

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if (C[i][j] != D[i][j]) {
        res = false;
      }
    }
  }
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}

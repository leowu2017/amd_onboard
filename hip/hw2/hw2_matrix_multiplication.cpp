#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hw_utils.hpp>

using scalar_t = float;

unsigned const dim_1 = 100;
unsigned const dim_2 = 20;
unsigned const dim_3 = 30;

scalar_t matrix_a[dim_1][dim_2];
scalar_t matrix_b[dim_2][dim_3];
scalar_t matrix_output_cpu[dim_1][dim_3];

void init() {
  for (int j = 0; j < dim_1; j++) {
    for (int i = 0; i < dim_2; i++) {
      matrix_a[j][i] = static_cast<scalar_t>(rand() % 100);
    }
  }

  for (int j = 0; j < dim_2; j++) {
    for (int i = 0; i < dim_3; i++) {
      matrix_b[j][i] = static_cast<scalar_t>(rand() % 100);
    }
  }
}

/*************
 *    CPU    *
 *************/
void execute_cpu() {
  for (int j = 0; j < dim_1; j++) {
    for (int i = 0; i < dim_3; i++) {
      matrix_output_cpu[j][i] = 0;
    }
  }

  for (int j = 0; j < dim_1; j++) {
    for (int i = 0; i < dim_3; i++) {
      for (int k = 0; k < dim_2; k++) {
        matrix_output_cpu[j][i] += matrix_a[j][k] * matrix_b[k][i];
      }
    }
  }
}

/*************
 *    GPU    *
 *************/
scalar_t matrix_output_gpu[dim_1][dim_3];

unsigned const block_size = 16;

__global__ void matrix_multiplication_kernel(scalar_t const *matrix_a,
                                             scalar_t const *matrix_b,
                                             scalar_t *matrix_out) {
  const unsigned bx = blockIdx.x;
  const unsigned by = blockIdx.y;
  const unsigned tx = threadIdx.x;
  const unsigned ty = threadIdx.y;

  const unsigned row = by * block_size + ty;
  const unsigned column = bx * block_size + tx;

  if ((row >= dim_1) || (column >= dim_3))
    return;

  scalar_t thread_result = 0;
  for (int i = 0; i < dim_2; i++) {
    const unsigned matrix_a_offset = dim_2 * row + i;
    const unsigned matrix_b_offset = dim_3 * i + column;
    thread_result += matrix_a[matrix_a_offset] * matrix_b[matrix_b_offset];
  }
  const unsigned matrix_out_offset = dim_3 * row + column;
  matrix_out[matrix_out_offset] = thread_result;
}

void execute_gpu() {
  // memory
  unsigned const matrix_a_bytes = dim_1 * dim_2 * sizeof(scalar_t);
  unsigned const matrix_b_bytes = dim_2 * dim_3 * sizeof(scalar_t);
  unsigned const matrix_output_bytes = dim_1 * dim_3 * sizeof(scalar_t);

  scalar_t *matrix_a_mem;
  scalar_t *matrix_b_mem;
  scalar_t *matrix_output_mem;

  HIP_CHECK(hipMalloc(&matrix_a_mem, matrix_a_bytes));
  HIP_CHECK(hipMalloc(&matrix_b_mem, matrix_b_bytes));
  HIP_CHECK(hipMalloc(&matrix_output_mem, matrix_output_bytes));

  HIP_CHECK(hipMemcpy(matrix_a_mem, matrix_a[0], matrix_a_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(matrix_b_mem, matrix_b[0], matrix_b_bytes,
                      hipMemcpyHostToDevice));

  // execute kernel
  dim3 const block_dim = {block_size, block_size};
  dim3 const grid_dim = {(dim_3 + block_size - 1) / block_size,
                         (dim_1 + block_size - 1) / block_size};
  matrix_multiplication_kernel<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      matrix_a_mem, matrix_b_mem, matrix_output_mem);

  HIP_CHECK(hipMemcpy(matrix_output_gpu[0], matrix_output_mem,
                      matrix_output_bytes, hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(matrix_a_mem));
  HIP_CHECK(hipFree(matrix_b_mem));
  HIP_CHECK(hipFree(matrix_output_mem));
}

void validate() {
  constexpr float tolerance = 0.001F;
  for (int i = 0; i < dim_1; i++) {
    for (int j = 0; j < dim_3; j++) {
      bool const pass = tolerance > std::abs(matrix_output_cpu[i][j] -
                                             matrix_output_gpu[i][j]);
      if (!pass) {
        std::cout << "Validation failed." << std::endl;
        return;
      }
    }
  }
  std::cout << "Validation passed." << std::endl;
}

int main() {

  init();
  execute_cpu();
  execute_gpu();
  validate();

  return 0;
}
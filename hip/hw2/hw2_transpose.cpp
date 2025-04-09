#include <iostream>

#include <hip/hip_runtime.h>
#include <hw_utils.hpp>

unsigned const width = 1024;
unsigned const height = 4096;
using scalar_t = float;

scalar_t matrix_in[height][width];
scalar_t matrix_out_cpu[width][height];

void init() {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      matrix_in[i][j] = (scalar_t)rand() / (scalar_t)RAND_MAX;
    }
  }
}

/*************
 *    CPU    *
 *************/
void exec_cpu() {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      matrix_out_cpu[j][i] = matrix_in[i][j];
    }
  }
}

/*************
 *    GPU    *
 *************/

scalar_t matrix_out_gpu[width][height];

unsigned const block_size = 16;

__global__ void transpose_kernel(scalar_t const *in, scalar_t *out) {
  unsigned const bx = blockIdx.x;
  unsigned const by = blockIdx.y;
  unsigned const tx = threadIdx.x;
  unsigned const ty = threadIdx.y;

  unsigned const x = bx * block_size + tx;
  unsigned const y = by * block_size + ty;

  if ((x >= height) || (y >= width))
    return;

  unsigned const output_offset = height * y + x;
  unsigned const input_offset = width * x + y;

  out[output_offset] = in[input_offset];
}

void exec_gpu() {
  // memory
  size_t const matrix_in_bytes = height * width * sizeof(scalar_t);
  size_t const matrix_out_bytes = width * height * sizeof(scalar_t);

  scalar_t *matrix_in_mem;
  scalar_t *matrix_out_mem;

  HIP_CHECK(hipMalloc(&matrix_in_mem, matrix_in_bytes));
  HIP_CHECK(hipMalloc(&matrix_out_mem, matrix_out_bytes));

  HIP_CHECK(hipMemcpy(matrix_in_mem, matrix_in[0], matrix_in_bytes,
                      hipMemcpyHostToDevice));

  // execute kernel
  dim3 const block_dim = {block_size, block_size};
  dim3 const grid_dim = {(height + block_size - 1) / block_size,
                         (width + block_size - 1) / block_size};
  transpose_kernel<<<grid_dim, block_dim, 0, hipStreamDefault>>>(
      matrix_in_mem, matrix_out_mem);

  HIP_CHECK(hipMemcpy(matrix_out_gpu[0], matrix_out_mem, matrix_out_bytes,
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(matrix_in_mem));
  HIP_CHECK(hipFree(matrix_out_mem));
}

void evaluate() {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      if (matrix_out_cpu[i][j] != matrix_out_gpu[i][j]) {
        std::cout << "Validation failed." << std::endl;
        return;
      }
    }
  }
  std::cout << "Validation passed." << std::endl;
}

int main() {
  init();
  exec_cpu();
  exec_gpu();
  evaluate();
  return 0;
}
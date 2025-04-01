#include <iostream>

#include <hip/hip_runtime.h>

int const error_exit_code = -1;

/*
  Inner product of 1024 x 512 matrix and 1024 vector.
  The result is a 512 vector.
*/

size_t const rows = 1024;
size_t const columns = 512;

float matrix_a[rows][columns]; // Matrix A
float vector_b[columns];       // Vector B

void init() {
  // Initialize matrix_a
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++) {
      matrix_a[i][j] = (i + j);
    }
  // Initialize vector_b
  for (int i = 0; i < columns; i++)
    vector_b[i] = (i);
}

/*************
 *    CPU    *
 *************/
float vector_out_cpu[rows]; // Result Vector for CPU

void exec_cpu() {
  // Initialize vector_out_cpu
  for (int i = 0; i < columns; i++)
    vector_out_cpu[i] = 0;

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++) {
      vector_out_cpu[i] += matrix_a[i][j] * vector_b[j];
    }
}

/*************
 *    GPU    *
 *************/
#define HIP_CHECK(condition)                                                   \
  {                                                                            \
    const hipError_t error = condition;                                        \
    if (error != hipSuccess) {                                                 \
      std::cerr << "An error encountered: \"" << hipGetErrorString(error)      \
                << "\" at " << __FILE__ << ':' << __LINE__ << std::endl;       \
      std::exit(error_exit_code);                                              \
    }                                                                          \
  }

const unsigned block_size = 16;

float vector_out_gpu[rows]; // Result Vector for GPU

__global__ void matrix_vector_multiplication_kernel(float const *matrix_a,
                                                    float const *vector_b,
                                                    float *vector_out) {
  const unsigned tx = threadIdx.x;
  const unsigned bx = blockIdx.x;

  const unsigned matrix_a_offset = columns * (bx * block_size + tx);
  const unsigned vector_out_offset = bx * block_size + tx;

  float thread_result = 0;
  for (int column = 0; column < columns; column++) {
    thread_result += matrix_a[matrix_a_offset + column] * vector_b[column];
  }
  vector_out[vector_out_offset] = thread_result;
}

void exec_gpu() {
  // memory
  const size_t matrix_a_bytes = sizeof(float) * rows * columns;
  const size_t vector_b_bytes = sizeof(float) * columns;
  const size_t vector_out_bytes = sizeof(float) * rows;
  float *matrix_a_gmem;
  float *vector_b_gmem;
  float *vector_out_gmem;
  HIP_CHECK(hipMalloc(&matrix_a_gmem, matrix_a_bytes));
  HIP_CHECK(hipMalloc(&vector_b_gmem, vector_b_bytes));
  HIP_CHECK(hipMalloc(&vector_out_gmem, vector_out_bytes));

  HIP_CHECK(hipMemcpy(matrix_a_gmem, matrix_a[0], matrix_a_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(vector_b_gmem, vector_b, vector_b_bytes,
                      hipMemcpyHostToDevice));

  // launch kernel
  const dim3 block_dim(block_size);
  const dim3 grid_dim(rows / block_size);

  matrix_vector_multiplication_kernel<<<grid_dim, block_dim, 0,
                                        hipStreamDefault>>>(
      matrix_a_gmem, vector_b_gmem, vector_out_gmem);
  HIP_CHECK(hipGetLastError());

  // results
  HIP_CHECK(hipMemcpy(vector_out_gpu, vector_out_gmem, vector_out_bytes,
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(matrix_a_gmem));
  HIP_CHECK(hipFree(vector_b_gmem));
  HIP_CHECK(hipFree(vector_out_gmem));
}

void validate() {
  constexpr float tolerance = 0.001F;
  for (int i = 0; i < rows; i++) {
    bool const pass =
        tolerance > std::abs(vector_out_cpu[i] - vector_out_gpu[i]);
    if (!pass) {
      std::cout << "Validation failed." << std::endl;
      return;
    }
  }
  std::cout << "Validation passed." << std::endl;
}

int main() {
  init();
  exec_cpu();
  exec_gpu();
  validate();

  return 0;
}

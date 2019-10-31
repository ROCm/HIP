// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args
// CHECK: #include <hip/hip_runtime.h>
#include <iostream>
// CHECK: #include <hipcub/hipcub.hpp>
#include <cub/cub.cuh>

// using namespace hipcub;
using namespace cub;

// Simple CUDA kernel for computing tiled partial sums
template <int BLOCK_THREADS, int ITEMS_PER_THREAD,
          // CHECK: hipcub::BlockLoadAlgorithm LOAD_ALGO,
          cub::BlockLoadAlgorithm LOAD_ALGO,
          // CHECK: hipcub::BlockScanAlgorithm SCAN_ALGO>
          cub::BlockScanAlgorithm SCAN_ALGO>
__global__ void ScanTilesKernel(int *d_in, int *d_out) {
  // Specialize collective types for problem context
  // CHECK: typedef ::hipcub::BlockLoad<int*, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGO> BlockLoadT;
  typedef ::cub::BlockLoad<int*, BLOCK_THREADS, ITEMS_PER_THREAD, LOAD_ALGO> BlockLoadT;
  typedef BlockScan<int, BLOCK_THREADS, SCAN_ALGO> BlockScanT;
  // Allocate on-chip temporary storage
  __shared__ union {
    typename BlockLoadT::TempStorage load;
    typename BlockScanT::TempStorage reduce;
  } temp_storage;
  // Load data per thread
  int thread_data[ITEMS_PER_THREAD];
  int offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
  BlockLoadT(temp_storage.load).Load(d_in + offset, offset);
  __syncthreads();
  // Compute the block-wide prefix sum
  BlockScanT(temp_storage).Sum(thread_data);
}

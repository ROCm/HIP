/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* HIT_START
 * BUILD: %t %s
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include "hip/hip_runtime.h"
#include "hip/hip_cooperative_groups.h"

#define GRID_X 512
#define GRID_Y 1
#define GRID_Z 1
#define BLOCK_X 256
#define BLOCK_Y 1
#define BLOCK_Z 1

#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#define RESULT_ASSERT(x, y) (assert((x) == (y)))

using namespace cooperative_groups;

__global__ void test_cg(unsigned int* gridSize, unsigned int* blockSize,
                        unsigned int* dTile64Size, unsigned int* sTile64Size,
                        unsigned int* dTile32Size, unsigned int* sTile32Size,
                        unsigned int* dTile16Size, unsigned int* sTile16Size,
                        unsigned int* dTile8Size, unsigned int* sTile8Size,
                        unsigned int* dTile4Size, unsigned int* sTile4Size,
                        unsigned int* dTile2Size, unsigned int* sTile2Size,
                        unsigned int* dTile1Size, unsigned int* sTile1Size)
{
  grid_group g = this_grid();
  if (g.thread_rank() == 0) {
    *gridSize = g.size();
  }

  thread_block tb = this_thread_block();
  if (tb.thread_rank() == 0) {
    *blockSize = tb.size();
  }

  thread_group d_tile64 = tiled_partition(tb, 64);
  if (d_tile64.thread_rank() == 0) {
    *dTile64Size = d_tile64.size();
  }

  thread_group d_tile32 = tiled_partition(d_tile64, 32);
  if (d_tile32.thread_rank() == 0) {
    *dTile32Size = d_tile32.size();
  }

  thread_group d_tile16 = tiled_partition(d_tile32, 16);
  if (d_tile16.thread_rank() == 0) {
    *dTile16Size = d_tile16.size();
  }

  thread_group d_tile8 = tiled_partition(d_tile16, 8);
  if (d_tile8.thread_rank() == 0) {
    *dTile8Size = d_tile8.size();
  }

  thread_group d_tile4 = tiled_partition(d_tile8, 4);
  if (d_tile4.thread_rank() == 0) {
    *dTile4Size = d_tile4.size();
  }

  thread_group d_tile2 = tiled_partition(d_tile4, 2);
  if (d_tile2.thread_rank() == 0) {
    *dTile2Size = d_tile2.size();
  }

  thread_group d_tile1 = tiled_partition(d_tile2, 1);
  if (d_tile1.thread_rank() == 0) {
    *dTile1Size = d_tile1.size();
  }

  thread_block_tile<64> s_tile64 = tiled_partition<64>(tb);
  if (s_tile64.thread_rank() == 0) {
    *sTile64Size = s_tile64.size();
  }

  thread_block_tile<32> s_tile32 = tiled_partition<32>(s_tile64);
  if (s_tile32.thread_rank() == 0) {
    *sTile32Size = s_tile32.size();
  }

  thread_block_tile<16> s_tile16 = tiled_partition<16>(s_tile32);
  if (s_tile16.thread_rank() == 0) {
    *sTile16Size = s_tile16.size();
  }

  thread_block_tile<8> s_tile8 = tiled_partition<8>(s_tile16);
  if (s_tile8.thread_rank() == 0) {
    *sTile8Size = s_tile8.size();
  }

  thread_block_tile<4> s_tile4 = tiled_partition<4>(s_tile8);
  if (s_tile4.thread_rank() == 0) {
    *sTile4Size = s_tile4.size();
  }

  thread_block_tile<2> s_tile2 = tiled_partition<2>(s_tile4);
  if (s_tile2.thread_rank() == 0) {
    *sTile2Size = s_tile2.size();
  }

  thread_block_tile<1> s_tile1 = tiled_partition<1>(s_tile2);
  if (s_tile1.thread_rank() == 0) {
    *sTile1Size = s_tile1.size();
  }
}

int main() {
  unsigned int *h_gridSize, *d_gridSize;
  unsigned int *h_blockSize, *d_blockSize;
  unsigned int *h_dTile64Size, *d_dTile64Size;
  unsigned int *h_dTile32Size, *d_dTile32Size;
  unsigned int *h_dTile16Size, *d_dTile16Size;
  unsigned int *h_dTile8Size, *d_dTile8Size;
  unsigned int *h_dTile4Size, *d_dTile4Size;
  unsigned int *h_dTile2Size, *d_dTile2Size;
  unsigned int *h_dTile1Size, *d_dTile1Size;
  unsigned int *h_sTile64Size, *d_sTile64Size;
  unsigned int *h_sTile32Size, *d_sTile32Size;
  unsigned int *h_sTile16Size, *d_sTile16Size;
  unsigned int *h_sTile8Size, *d_sTile8Size;
  unsigned int *h_sTile4Size, *d_sTile4Size;
  unsigned int *h_sTile2Size, *d_sTile2Size;
  unsigned int *h_sTile1Size, *d_sTile1Size;

  size_t nBytes = sizeof(unsigned int);

  h_gridSize = (unsigned int*) malloc(nBytes);
  h_blockSize = (unsigned int*) malloc(nBytes);
  h_dTile64Size = (unsigned int*) malloc(nBytes);
  h_dTile32Size = (unsigned int*) malloc(nBytes);
  h_dTile16Size = (unsigned int*) malloc(nBytes);
  h_dTile8Size = (unsigned int*) malloc(nBytes);
  h_dTile4Size = (unsigned int*) malloc(nBytes);
  h_dTile2Size = (unsigned int*) malloc(nBytes);
  h_dTile1Size = (unsigned int*) malloc(nBytes);
  h_sTile64Size = (unsigned int*) malloc(nBytes);
  h_sTile32Size = (unsigned int*) malloc(nBytes);
  h_sTile16Size = (unsigned int*) malloc(nBytes);
  h_sTile8Size = (unsigned int*) malloc(nBytes);
  h_sTile4Size = (unsigned int*) malloc(nBytes);
  h_sTile2Size = (unsigned int*) malloc(nBytes);
  h_sTile1Size = (unsigned int*) malloc(nBytes);

  HIP_ASSERT(hipMalloc(&d_gridSize, nBytes));
  HIP_ASSERT(hipMalloc(&d_blockSize, nBytes));
  HIP_ASSERT(hipMalloc(&d_dTile64Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_dTile32Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_dTile16Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_dTile8Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_dTile4Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_dTile2Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_dTile1Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_sTile64Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_sTile32Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_sTile16Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_sTile8Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_sTile4Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_sTile2Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_sTile1Size, nBytes));

  hipLaunchKernelGGL(test_cg,
                     dim3(GRID_X, GRID_Y, GRID_Z),
                     dim3(BLOCK_X, BLOCK_Y, BLOCK_Z),
                     0,
                     0,
                     d_gridSize, d_blockSize,
                     d_dTile64Size, d_sTile64Size,
                     d_dTile32Size, d_sTile32Size,
                     d_dTile16Size, d_sTile16Size,
                     d_dTile8Size, d_sTile8Size,
                     d_dTile4Size, d_sTile4Size,
                     d_dTile2Size, d_sTile2Size,
                     d_dTile1Size, d_sTile1Size);

  HIP_ASSERT(hipMemcpy(h_gridSize, d_gridSize, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_blockSize, d_blockSize, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_dTile64Size, d_dTile64Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_dTile32Size, d_dTile32Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_dTile16Size, d_dTile16Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_dTile8Size, d_dTile8Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_dTile4Size, d_dTile4Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_dTile2Size, d_dTile2Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_dTile1Size, d_dTile1Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_sTile64Size, d_sTile64Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_sTile32Size, d_sTile32Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_sTile16Size, d_sTile16Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_sTile8Size, d_sTile8Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_sTile4Size, d_sTile4Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_sTile2Size, d_sTile2Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_sTile1Size, d_sTile1Size, nBytes, hipMemcpyDeviceToHost));

  unsigned int expectedBlockSize = (BLOCK_X * BLOCK_Y * BLOCK_Z);
  unsigned int expectedGridSize = ((GRID_X * GRID_Y * GRID_Z) * expectedBlockSize);

  RESULT_ASSERT(*h_gridSize, expectedGridSize);
  RESULT_ASSERT(*h_blockSize, expectedBlockSize);
  RESULT_ASSERT(*h_dTile64Size, 64);
  RESULT_ASSERT(*h_dTile32Size, 32);
  RESULT_ASSERT(*h_dTile16Size, 16);
  RESULT_ASSERT(*h_dTile8Size, 8);
  RESULT_ASSERT(*h_dTile4Size, 4);
  RESULT_ASSERT(*h_dTile2Size, 2);
  RESULT_ASSERT(*h_dTile1Size, 1);
  RESULT_ASSERT(*h_sTile64Size, 64);
  RESULT_ASSERT(*h_sTile32Size, 32);
  RESULT_ASSERT(*h_sTile16Size, 16);
  RESULT_ASSERT(*h_sTile8Size, 8);
  RESULT_ASSERT(*h_sTile4Size, 4);
  RESULT_ASSERT(*h_sTile2Size, 2);
  RESULT_ASSERT(*h_sTile1Size, 1);

  free(h_gridSize);
  free(h_blockSize);
  free(h_dTile64Size);
  free(h_dTile32Size);
  free(h_dTile16Size);
  free(h_dTile8Size);
  free(h_dTile4Size);
  free(h_dTile2Size);
  free(h_dTile1Size);
  free(h_sTile64Size);
  free(h_sTile32Size);
  free(h_sTile16Size);
  free(h_sTile8Size);
  free(h_sTile4Size);
  free(h_sTile2Size);
  free(h_sTile1Size);

  HIP_ASSERT(hipFree(d_gridSize));
  HIP_ASSERT(hipFree(d_blockSize));
  HIP_ASSERT(hipFree(d_dTile64Size));
  HIP_ASSERT(hipFree(d_dTile32Size));
  HIP_ASSERT(hipFree(d_dTile16Size));
  HIP_ASSERT(hipFree(d_dTile8Size));
  HIP_ASSERT(hipFree(d_dTile4Size));
  HIP_ASSERT(hipFree(d_dTile2Size));
  HIP_ASSERT(hipFree(d_dTile1Size));
  HIP_ASSERT(hipFree(d_sTile64Size));
  HIP_ASSERT(hipFree(d_sTile32Size));
  HIP_ASSERT(hipFree(d_sTile16Size));
  HIP_ASSERT(hipFree(d_sTile8Size));
  HIP_ASSERT(hipFree(d_sTile4Size));
  HIP_ASSERT(hipFree(d_sTile2Size));
  HIP_ASSERT(hipFree(d_sTile1Size));

  passed();
}

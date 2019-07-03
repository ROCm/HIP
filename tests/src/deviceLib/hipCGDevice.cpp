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

__global__ void test_cg(unsigned int* gridSize,
                        unsigned int* blockSize,
                        unsigned int* dTile64Size,
                        unsigned int* dTile32Size,
                        unsigned int* sTile64Size,
                        unsigned int* sTile32Size)
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

  thread_block_tile<64> s_tile64 = tiled_partition<64>(tb);
  if (s_tile64.thread_rank() == 0) {
    *sTile64Size = s_tile64.size();
  }

  thread_block_tile<32> s_tile32 = tiled_partition<32>(s_tile64);
  if (s_tile32.thread_rank() == 0) {
    *sTile32Size = s_tile32.size();
  }
}

int main() {
  unsigned int *h_gridSize, *d_gridSize;
  unsigned int *h_blockSize, *d_blockSize;
  unsigned int *h_dTile64Size, *d_dTile64Size;
  unsigned int *h_dTile32Size, *d_dTile32Size;
  unsigned int *h_sTile64Size, *d_sTile64Size;
  unsigned int *h_sTile32Size, *d_sTile32Size;
  size_t nBytes = sizeof(unsigned int);

  h_gridSize = (unsigned int*) malloc(nBytes);
  h_blockSize = (unsigned int*) malloc(nBytes);
  h_dTile64Size = (unsigned int*) malloc(nBytes);
  h_dTile32Size = (unsigned int*) malloc(nBytes);
  h_sTile64Size = (unsigned int*) malloc(nBytes);
  h_sTile32Size = (unsigned int*) malloc(nBytes);

  HIP_ASSERT(hipMalloc(&d_gridSize, nBytes));
  HIP_ASSERT(hipMalloc(&d_blockSize, nBytes));
  HIP_ASSERT(hipMalloc(&d_dTile64Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_dTile32Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_sTile64Size, nBytes));
  HIP_ASSERT(hipMalloc(&d_sTile32Size, nBytes));

  hipLaunchKernelGGL(test_cg,
                     dim3(GRID_X, GRID_Y, GRID_Z),
                     dim3(BLOCK_X, BLOCK_Y, BLOCK_Z),
                     0,
                     0,
                     d_gridSize,
                     d_blockSize,
                     d_dTile64Size,
                     d_dTile32Size,
                     d_sTile64Size,
                     d_sTile32Size);

  HIP_ASSERT(hipMemcpy(h_gridSize, d_gridSize, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_blockSize, d_blockSize, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_dTile64Size, d_dTile64Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_dTile32Size, d_dTile32Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_sTile64Size, d_sTile64Size, nBytes, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipMemcpy(h_sTile32Size, d_sTile32Size, nBytes, hipMemcpyDeviceToHost));

  unsigned int exptectBlockSize = (BLOCK_X * BLOCK_Y * BLOCK_Z);
  unsigned int exptectGrdSize = ((GRID_X * GRID_Y * GRID_Z) * exptectBlockSize);

  RESULT_ASSERT(*h_gridSize, exptectGrdSize);
  RESULT_ASSERT(*h_blockSize, exptectBlockSize);
  RESULT_ASSERT(*h_dTile64Size, 64);
  RESULT_ASSERT(*h_dTile32Size, 32);
  RESULT_ASSERT(*h_sTile64Size, 64);
  RESULT_ASSERT(*h_sTile32Size, 32);

  free(h_gridSize);
  free(h_blockSize);
  free(h_dTile64Size);
  free(h_dTile32Size);
  free(h_sTile64Size);
  free(h_sTile32Size);

  HIP_ASSERT(hipFree(d_gridSize));
  HIP_ASSERT(hipFree(d_blockSize));
  HIP_ASSERT(hipFree(d_dTile64Size));
  HIP_ASSERT(hipFree(d_dTile32Size));
  HIP_ASSERT(hipFree(d_sTile64Size));
  HIP_ASSERT(hipFree(d_sTile32Size));

  passed();
}

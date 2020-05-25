/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include "hip/hip_cooperative_groups.h"

#define HIP_ASSERT(lhs, rhs) assert(lhs == rhs)

using namespace cooperative_groups;

static __global__
void kernel_cg_thread_block_type(dim3 *groupIndexD,
                                 dim3 *thdIndexD,
                                 int *sizeD,
                                 int *thdRankD,
                                 int *isValidD,
                                 int *syncValD)
{
  thread_block tb = this_thread_block();

  // Consider the workgroup id (0, 1, 1) and thread id (0, 1, 2) for validation
  // of the test
  int isBlockIdx011 =
    (hipBlockIdx_x == 0 && hipBlockIdx_y == 1 && hipBlockIdx_z == 1);
  int isThreadIdx012 =
    (hipThreadIdx_x == 0 && hipThreadIdx_y == 1 && hipThreadIdx_z == 2);

  if (isBlockIdx011 && isThreadIdx012) {
    *groupIndexD = tb.group_index();
    *thdIndexD = tb.thread_index();
    *sizeD = tb.size();
    *thdRankD = tb.thread_rank();
    *isValidD = tb.is_valid();
  }

  // Consider local thread id (0, 0, 0) and (0, 0, 1) for validation of sync()
  // api
  __shared__ int sVar[2];
  int isThreadIdx000 =
    (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0);
  int isThreadIdx001 =
    (hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 1);

  if (isThreadIdx000)
    sVar[0] = 10;
  if (isThreadIdx001)
    sVar[1] = 20;
  tb.sync();
  if (isBlockIdx011 && isThreadIdx012)
    *syncValD = sVar[0] + sVar[1];
}

static void test_cg_thread_block_type()
{
  dim3 *groupIndexD, *groupIndexH;
  dim3 *thdIndexD, *thdIndexH;
  int *sizeD, *sizeH;
  int *thdRankD, *thdRankH;
  int *isValidD, *isValidH;
  int *syncValD, *syncValH;

  // Allocate device memory
  HIP_ASSERT(hipMalloc((void**)&groupIndexD, sizeof(dim3)), hipSuccess);
  HIP_ASSERT(hipMalloc((void**)&thdIndexD, sizeof(dim3)), hipSuccess);
  HIP_ASSERT(hipMalloc((void**)&sizeD, sizeof(int)), hipSuccess);
  HIP_ASSERT(hipMalloc((void**)&thdRankD, sizeof(int)), hipSuccess);
  HIP_ASSERT(hipMalloc((void**)&isValidD, sizeof(int)), hipSuccess);
  HIP_ASSERT(hipMalloc((void**)&syncValD, sizeof(int)), hipSuccess);

  // Allocate host memory
  HIP_ASSERT(hipHostMalloc((void**)&groupIndexH, sizeof(dim3)), hipSuccess);
  HIP_ASSERT(hipHostMalloc((void**)&thdIndexH, sizeof(dim3)), hipSuccess);
  HIP_ASSERT(hipHostMalloc((void**)&sizeH, sizeof(int)), hipSuccess);
  HIP_ASSERT(hipHostMalloc((void**)&thdRankH, sizeof(int)), hipSuccess);
  HIP_ASSERT(hipHostMalloc((void**)&isValidH, sizeof(int)), hipSuccess);
  HIP_ASSERT(hipHostMalloc((void**)&syncValH, sizeof(int)), hipSuccess);

  // Launch Kernel
  hipLaunchKernelGGL(kernel_cg_thread_block_type,
                     dim3(2, 2, 2),
                     dim3(4, 4, 4),
                     0,
                     0,
                     groupIndexD,
                     thdIndexD,
                     sizeD,
                     thdRankD,
                     isValidD,
                     syncValD);

  // Copy result from device to host
  HIP_ASSERT(hipMemcpy(groupIndexH, groupIndexD, sizeof(dim3), hipMemcpyDeviceToHost), hipSuccess);
  HIP_ASSERT(hipMemcpy(thdIndexH, thdIndexD, sizeof(dim3), hipMemcpyDeviceToHost), hipSuccess);
  HIP_ASSERT(hipMemcpy(sizeH, sizeD, sizeof(int), hipMemcpyDeviceToHost), hipSuccess);
  HIP_ASSERT(hipMemcpy(thdRankH, thdRankD, sizeof(int), hipMemcpyDeviceToHost), hipSuccess);
  HIP_ASSERT(hipMemcpy(isValidH, isValidD, sizeof(int), hipMemcpyDeviceToHost), hipSuccess);
  HIP_ASSERT(hipMemcpy(syncValH, syncValD, sizeof(int), hipMemcpyDeviceToHost), hipSuccess);

  // Validate result
  // Group index should be (0, 1, 1)
  HIP_ASSERT(groupIndexH->x, 0);
  HIP_ASSERT(groupIndexH->y, 1);
  HIP_ASSERT(groupIndexH->z, 1);
  // Thread index should be (0, 1, 2)
  HIP_ASSERT(thdIndexH->x, 0);
  HIP_ASSERT(thdIndexH->y, 1);
  HIP_ASSERT(thdIndexH->z, 2);
  // Workgroup size should be 64
  HIP_ASSERT(*sizeH, 64);
  // Thread rank should be 36
  HIP_ASSERT(*thdRankH, 36);
  // Call to is_valid() should return true
  HIP_ASSERT(*isValidH, 1);
  // syncVal should be equal to 30
  HIP_ASSERT(*syncValH, 30);

  // Free device memory
  HIP_ASSERT(hipFree(groupIndexD), hipSuccess);
  HIP_ASSERT(hipFree(thdIndexD), hipSuccess);
  HIP_ASSERT(hipFree(sizeD), hipSuccess);
  HIP_ASSERT(hipFree(thdRankD), hipSuccess);
  HIP_ASSERT(hipFree(isValidD), hipSuccess);
  HIP_ASSERT(hipFree(syncValD), hipSuccess);

  //Free host memory
  HIP_ASSERT(hipHostFree(groupIndexH), hipSuccess);
  HIP_ASSERT(hipHostFree(thdIndexH), hipSuccess);
  HIP_ASSERT(hipHostFree(sizeH), hipSuccess);
  HIP_ASSERT(hipHostFree(thdRankH), hipSuccess);
  HIP_ASSERT(hipHostFree(isValidH), hipSuccess);
  HIP_ASSERT(hipHostFree(syncValH), hipSuccess);
}

int main()
{
  test_cg_thread_block_type();
  passed();
}

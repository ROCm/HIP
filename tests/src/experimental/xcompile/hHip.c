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


#include "gHipApi.h"
#include "hip/hip_runtime.h"

#define LEN 1024*1024
#define SIZE LEN * sizeof(float)

__global__ void Add(hipLaunchParm lp, float *Ad, float *Bd, float *Cd, size_t len)
{
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if(tx < len)
    {
        Cd[tx] = Ad[tx] + Bd[tx];
    }
}

int main()
{
    mem_manager *a, *b, *c;
    a = mem_manager_start(SIZE);
    b = mem_manager_start(SIZE);
    c = mem_manager_start(SIZE);
    a->malloc_hst(a);
    b->malloc_hst(b);
    c->malloc_hst(c);
    a->malloc_hip(a);
    b->malloc_hip(b);
    c->malloc_hip(c);
    memset_hst(a, 1.0f);
    memset_hst(b, 2.0f);
    a->h2d(a);
    b->h2d(b);
    dim3 dimGrid, dimBlock;
    dimBlock.x = 1024, dimBlock.y = 1, dimBlock.z = 1;
    dimGrid.x = LEN/1024, dimGrid.y = 1, dimGrid.z = 1;
    hipLaunchKernel(HIP_KERNEL_NAME(Add), dimGrid, dimBlock, 0, 0, (float*)a->dev_ptr, (float*)b->dev_ptr, (float*)c->dev_ptr, LEN);
    c->d2h(c);
    assert(((float*)c->hst_ptr)[10] == 3.0f);


}

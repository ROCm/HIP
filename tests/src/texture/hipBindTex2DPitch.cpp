/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.
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

/*HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_RUNTIME rocclr
 * TEST: %t
 * HIT_END
 */
#include "test_common.h"

#define SIZE_H 8
#define SIZE_W 12
#define TYPE_t float

texture<TYPE_t, 2, hipReadModeElementType> tex;

// texture object is a kernel argument
__global__ void texture2dCopyKernel( TYPE_t* dst) {

    int x = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int y = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;
    if ( (x< SIZE_W) && (y< SIZE_H) ){
        dst[SIZE_W*y+x] = tex2D(tex, x, y);
    }
}

int main (void)
{
    TYPE_t* B;
    TYPE_t* A;
    TYPE_t* devPtrB;
    TYPE_t* devPtrA;

    B = new TYPE_t[SIZE_H*SIZE_W];
    A = new TYPE_t[SIZE_H*SIZE_W];
    for(size_t i=1; i <= (SIZE_H*SIZE_W); i++){
        A[i-1] = i;
    }

    size_t devPitchA, tex_ofs;
    HIPCHECK(hipMallocPitch((void**)&devPtrA, &devPitchA ,SIZE_W*sizeof(TYPE_t), SIZE_H)) ;
    HIPCHECK(hipMemcpy2D(devPtrA, devPitchA, A, SIZE_W*sizeof(TYPE_t),
            SIZE_W*sizeof(TYPE_t), SIZE_H, hipMemcpyHostToDevice));

    tex.normalized = false;
    HIPCHECK(hipBindTexture2D(&tex_ofs, &tex, devPtrA, &tex.channelDesc,
                                       SIZE_W, SIZE_H, devPitchA));
    HIPCHECK(hipMalloc((void**)&devPtrB, SIZE_W*sizeof(TYPE_t)*SIZE_H)) ;

    hipLaunchKernelGGL(texture2dCopyKernel, dim3(4,4,1), dim3(32,32,1), 0, 0, devPtrB);
    hipDeviceSynchronize();
    HIPCHECK(hipMemcpy2D(B, SIZE_W*sizeof(TYPE_t), devPtrB, SIZE_W*sizeof(TYPE_t),
            SIZE_W*sizeof(TYPE_t), SIZE_H, hipMemcpyDeviceToHost));

    HipTest::checkArray(A, B, SIZE_H, SIZE_W);
    delete []A;
    delete []B;
    hipFree(devPtrA);
    hipFree(devPtrB);
    passed();
}

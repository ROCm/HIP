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

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

template <typename T>
void runTest(int width,int height,int depth, hipChannelFormatKind formatKind)
{
    unsigned int size = width * height * depth * sizeof(T);
    T* hData = (T*) malloc(size);
    memset(hData, 0, size);

    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                hData[i*width*height + j*width +k] = i*width*height + j*width + k;
            }
        }
    }
    printf("test- sizeof(T) =%d\n", sizeof(T));
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(T)*8, 0, 0, 0, formatKind);
    hipArray *arr,*arr1;

    HIPCHECK(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height, depth), hipArrayDefault));
    HIPCHECK(hipMalloc3DArray(&arr1, &channelDesc, make_hipExtent(width, height, depth), hipArrayDefault));
    hipMemcpy3DParms myparms = {0};
    myparms.srcPos = make_hipPos(0,0,0);
    myparms.dstPos = make_hipPos(0,0,0);
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T), width, height);
    myparms.dstArray = arr;
    myparms.extent = make_hipExtent(width , height, depth);
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyHostToDevice;
#else
    myparms.kind = hipMemcpyHostToDevice;
#endif
    HIPCHECK(hipMemcpy3D(&myparms));
    HIPCHECK(hipDeviceSynchronize());
    //Array to Array
    memset(&myparms,0x0, sizeof(hipMemcpy3DParms));
    myparms.srcPos = make_hipPos(0,0,0);
    myparms.dstPos = make_hipPos(0,0,0);
    myparms.srcArray = arr;
    myparms.dstArray = arr1;
    myparms.extent = make_hipExtent(width, height, depth);
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToDevice;
#else
    myparms.kind = hipMemcpyDeviceToDevice;
#endif
    HIPCHECK(hipMemcpy3D(&myparms));
    HIPCHECK(hipDeviceSynchronize());

    T *hOutputData = (T*) malloc(size);
    memset(hOutputData, 0,  size);
    //Device to host
    memset(&myparms,0x0, sizeof(hipMemcpy3DParms));
    myparms.srcPos = make_hipPos(0,0,0);
    myparms.dstPos = make_hipPos(0,0,0);
    myparms.dstPtr = make_hipPitchedPtr(hOutputData, width * sizeof(T), width, height);
    myparms.srcArray = arr1;
    myparms.extent = make_hipExtent(width, height, depth);
#ifdef __HIP_PLATFORM_NVCC__
    myparms.kind = cudaMemcpyDeviceToHost;
#else
    myparms.kind = hipMemcpyDeviceToHost;
#endif
    HIPCHECK(hipMemcpy3D(&myparms));
    HIPCHECK(hipDeviceSynchronize());

    // Check result
    HipTest::checkArray(hData,hOutputData,width,height,depth);
    hipFreeArray(arr);
    hipFreeArray(arr1);
    free(hData);
    free(hOutputData);
}

int main(int argc, char **argv)
{
    for(int i=1;i<25;i++)
    {
        runTest<float>(i,i,i, hipChannelFormatKindFloat);
        runTest<int>(i+1,i,i, hipChannelFormatKindSigned);
        runTest<char>(i,i+1,i, hipChannelFormatKindSigned);
    }
    passed();
}

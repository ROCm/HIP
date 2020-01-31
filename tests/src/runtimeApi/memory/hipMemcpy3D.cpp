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
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"

template <typename T>
void runTest(int width,int height,int depth)
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
    // Allocate array and copy image data
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc(sizeof(T)*8, 0, 0, 0, hipChannelFormatKindSigned);
    hipArray *arr;

    HIPCHECK(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height, depth), hipArrayCubemap));
    hipMemcpy3DParms myparms = {0};
    myparms.srcPos = make_hipPos(0,0,0);
    myparms.dstPos = make_hipPos(0,0,0);
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T), width, height);
    myparms.dstArray = arr;
    myparms.extent = make_hipExtent(width * sizeof(T), height, depth);
    myparms.kind = hipMemcpyHostToDevice;
    HIPCHECK(hipMemcpy3D(&myparms));
    HIPCHECK(hipDeviceSynchronize());
    HipTest::checkArray(hData,(T*)arr->data,width,height,depth);
    hipFreeArray(arr);
    free(hData);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    for(int i=1;i<25;i++)
    {
        runTest<float>(i,i,i);
        runTest<int>(i+1,i,i);
        runTest<char>(i,i+1,i);
    }
    passed();
}

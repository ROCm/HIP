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
    
    hipArray *arr=NULL,*arr1=NULL;
   
    HIP_ARRAY3D_DESCRIPTOR desc = {0};
    desc.Format = HIP_AD_FORMAT_FLOAT;
    desc.Depth = 1;
    desc.Flags = hipArrayDefault;
    desc.NumChannels = 4;
    desc.Width = 1;
    desc.Height = 1;

    HIPCHECK(hipArray3DCreate(&arr, &desc)); 
    HIPCHECK(hipArray3DCreate(&arr1, &desc)); 

    HIP_MEMCPY3D myparms = {0};
    myparms.srcXInBytes = myparms.srcY = myparms.srcZ = 0;
    myparms.srcMemoryType = hipMemoryTypeHost;
    myparms.srcHost = (void *)hData;
    myparms.srcPitch = width * sizeof(T);
    myparms.srcHeight = height;   

    myparms.dstXInBytes = myparms.dstY = myparms.dstZ = 0;
    myparms.dstMemoryType = hipMemoryTypeArray;
    myparms.dstArray = arr;
    myparms.dstArray->isDrv = true;
    myparms.WidthInBytes = width;
    myparms.Height = height;
    myparms.Depth = depth;

    HIPCHECK(hipDrvMemcpy3D(&myparms));
    HIPCHECK(hipDeviceSynchronize());
     
    //Array to Array
    memset(&myparms,0x0, sizeof(myparms));
    myparms.srcXInBytes = myparms.srcY = myparms.srcZ = 0;
    myparms.srcMemoryType = hipMemoryTypeArray;
    myparms.srcArray = arr;
    myparms.srcArray->isDrv = true;
    myparms.dstXInBytes = myparms.dstY = myparms.dstZ = 0;
    myparms.dstMemoryType = hipMemoryTypeArray;
    myparms.dstArray = arr1;
    myparms.dstArray->isDrv = true;
    myparms.WidthInBytes = width;
    myparms.Height = height;
    myparms.Depth = depth;
    
    HIPCHECK(hipDrvMemcpy3D(&myparms));
    HIPCHECK(hipDeviceSynchronize());

    T *hOutputData = (T*) malloc(size);
    memset(hOutputData, 0, size);
    //Device to host
    memset(&myparms,0x0, sizeof(myparms));
    myparms.srcXInBytes = myparms.srcY = myparms.srcZ = 0;
    myparms.srcMemoryType = hipMemoryTypeArray;
    myparms.srcArray = arr1;
    myparms.srcArray->isDrv = true;
    
    myparms.dstXInBytes = myparms.dstY = myparms.dstZ = 0;
    myparms.dstMemoryType = hipMemoryTypeHost;
    myparms.dstHost = (void *)hOutputData;
    myparms.dstPitch = width * sizeof(T);
    myparms.dstHeight = height;  
    myparms.WidthInBytes = width;
    myparms.Height = height;
    myparms.Depth = depth;
    
    HIPCHECK(hipDrvMemcpy3D(&myparms));
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
    for(int i=1;i<25;++i)
    {
        runTest<float>(i,i,i);
        runTest<int>(i+1,i,i);
        runTest<char>(i,i+1,i);
    }
    passed();
}

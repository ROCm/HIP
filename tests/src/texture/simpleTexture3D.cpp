/*
Copyright (c) 2019-present Advanced Micro Devices, Inc. All rights reserved.

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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * TEST: %t
 * HIT_END
 */
#include "test_common.h"

//typedef char T;
const char *sampleName = "simpleTexture3D";

// Texture reference for 3D texture
texture<float, hipTextureType3D, hipReadModeElementType> texf;

texture<int, hipTextureType3D, hipReadModeElementType> texi;

texture<char, hipTextureType3D, hipReadModeElementType> texc;

template <typename T>
__global__ void simpleKernel3DArray(T* outputData, 
                                    int width,
                                    int height,int depth)
{
    for (int i = 0; i < depth; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                if(std::is_same<T, float>::value)
                    outputData[i*width*height + j*width + k] = tex3D(texf, k, j, i);
                else if(std::is_same<T, int>::value)
                    outputData[i*width*height + j*width + k] = tex3D(texi, k, j, i);
                else if(std::is_same<T, char>::value)
                    outputData[i*width*height + j*width + k] = tex3D(texc, k, j, i);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for tex3D
////////////////////////////////////////////////////////////////////////////////
template <typename T>
void runTest(int width,int height,int depth,texture<T, hipTextureType3D, hipReadModeElementType> *tex)
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

    // Allocate array and copy image data
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc<T>();
    hipArray *arr;

    HIPCHECK(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height, depth), hipArrayDefault));
    hipMemcpy3DParms myparms = {0};
    myparms.srcPos = make_hipPos(0,0,0);
    myparms.dstPos = make_hipPos(0,0,0);
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T), width, height);
    myparms.dstArray = arr;
    myparms.extent = make_hipExtent(width, height, depth);
    myparms.kind = hipMemcpyHostToDevice;
    HIPCHECK(hipMemcpy3D(&myparms));

    // set texture parameters
    tex->addressMode[0] = hipAddressModeWrap;
    tex->addressMode[1] = hipAddressModeWrap;
    tex->filterMode = hipFilterModePoint;
    tex->normalized = false;

    // Bind the array to the texture
    HIPCHECK(hipBindTextureToArray(*tex, arr, channelDesc));

    // Allocate device memory for result
    T* dData = NULL;
    hipMalloc((void **) &dData, size);

    hipLaunchKernelGGL(simpleKernel3DArray, dim3(1,1,1), dim3(1,1,1), 0, 0, dData, width, height, depth);
    HIPCHECK(hipDeviceSynchronize());

    // Allocate mem for the result on host side
    T *hOutputData = (T*) malloc(size);
    memset(hOutputData, 0,  size);

    // copy result from device to host
    HIPCHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));
    HipTest::checkArray(hData,hOutputData,width,height,depth); 

    hipFree(dData);
    hipFreeArray(arr);
    free(hData);
    free(hOutputData);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);
    for(int i=1;i<25;i++)
    {
        runTest<float>(i,i,i,&texf);
        runTest<int>(i+1,i,i,&texi);
        runTest<char>(i,i+1,i,&texc);
    }
    passed();
}


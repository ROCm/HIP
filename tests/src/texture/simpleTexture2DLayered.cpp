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
 * BUILD: %t %s ../test_common.cpp
 * TEST: %t
 * HIT_END
 */
#include "test_common.h"

typedef float T;

// Texture reference for 2D Layered texture
texture<float, hipTextureType2DLayered> tex2DL;

__global__ void simpleKernelLayeredArray(T* outputData,int width,int height,int layer)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    outputData[layer*width*height + y*width + x] = tex2DLayered(tex2DL, x, y, layer);
}

////////////////////////////////////////////////////////////////////////////////
void runTest(int width,int height,int num_layers,texture<T, hipTextureType2DLayered> *tex)
{
    unsigned int size = width * height * num_layers * sizeof(T);
    T* hData = (T*) malloc(size);
    memset(hData, 0, size);

    for (unsigned int layer = 0; layer < num_layers; layer++){
       for (int i = 0; i < (int)(width * height); i++){
           hData[layer*width*height + i] = i;
       }
    }
    hipChannelFormatDesc channelDesc;
    // Allocate array and copy image data
    channelDesc = hipCreateChannelDesc(sizeof(T)*8, 0, 0, 0, hipChannelFormatKindFloat);
    hipArray *arr;

    HIPCHECK(hipMalloc3DArray(&arr, &channelDesc, make_hipExtent(width, height, num_layers), hipArrayLayered));
    hipMemcpy3DParms myparms = {0};
    myparms.srcPos = make_hipPos(0,0,0);
    myparms.dstPos = make_hipPos(0,0,0);
    myparms.srcPtr = make_hipPitchedPtr(hData, width * sizeof(T), width, height);
    myparms.dstArray = arr;
    myparms.extent = make_hipExtent(width , height, num_layers);
    //myparms.kind = hipMemcpyHostToDevice;
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
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    for (unsigned int layer = 0; layer < num_layers; layer++)
        hipLaunchKernelGGL(simpleKernelLayeredArray, dimGrid, dimBlock, 0, 0, dData, width, height, layer);
    
    HIPCHECK(hipDeviceSynchronize());
    // Allocate mem for the result on host side
    T *hOutputData = (T*) malloc(size);
    memset(hOutputData, 0,  size);

    // copy result from device to host
    HIPCHECK(hipMemcpy(hOutputData, dData, size, hipMemcpyDeviceToHost));
    HipTest::checkArray(hData,hOutputData,width,height,num_layers); 

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
    runTest(512,512,5,&tex2DL);
    passed();
}


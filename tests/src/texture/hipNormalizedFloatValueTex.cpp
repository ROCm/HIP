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
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia EXCLUDE_HIP_PLATFORM amd
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#define SIZE 10

static float getNormalizedValue(const float value,
                                const hipChannelFormatDesc& desc) {
    if ((desc.x == 8) && (desc.f == hipChannelFormatKindSigned))
        return (value / SCHAR_MAX);
    if ((desc.x == 8) && (desc.f == hipChannelFormatKindUnsigned))
        return (value / UCHAR_MAX);
    if ((desc.x == 16) && (desc.f == hipChannelFormatKindSigned))
        return (value / SHRT_MAX);
    if ((desc.x == 16) && (desc.f == hipChannelFormatKindUnsigned))
        return (value / USHRT_MAX);
    return value;
}

texture<char, hipTextureType1D, hipReadModeNormalizedFloat> texc;

texture<unsigned char, hipTextureType1D, hipReadModeNormalizedFloat> texuc;

texture<short, hipTextureType1D, hipReadModeNormalizedFloat> texs;

texture<unsigned short, hipTextureType1D, hipReadModeNormalizedFloat> texus;


template<typename T>
__global__ void normalizedValTextureTest(unsigned int numElements, float* pDst)
{
    unsigned int elementID = hipThreadIdx_x;
    if(elementID >= numElements)
        return;
    float coord =(float) elementID/numElements;
    if(std::is_same<T, char>::value)
        pDst[elementID] = tex1D(texc, coord);
    else if(std::is_same<T, unsigned char>::value)
        pDst[elementID] = tex1D(texuc, coord);
    else if(std::is_same<T, short>::value)
        pDst[elementID] = tex1D(texs, coord);
    else if(std::is_same<T, unsigned short>::value)
        pDst[elementID] = tex1D(texus, coord);
}

template<typename T>
bool textureTest(texture<T, hipTextureType1D, hipReadModeNormalizedFloat> *tex)
{
    hipChannelFormatDesc desc = hipCreateChannelDesc<T>();
    hipArray_t dData;
    HIPCHECK(hipMallocArray(&dData, &desc, SIZE, 1, hipArrayDefault));

    T hData[] = {65, 66, 67, 68, 69, 70, 71, 72, 73, 74};
    HIPCHECK(hipMemcpy2DToArray(dData, 0, 0, hData, sizeof(T)*SIZE, sizeof(T)*SIZE, 1, hipMemcpyHostToDevice));

    tex->normalized = true;
    tex->channelDesc = desc;
    HIPCHECK(hipBindTextureToArray(tex, dData, &desc));

    float *dOutputData = NULL;
    HIPCHECK(hipMalloc((void **) &dOutputData, sizeof(float)*SIZE));

    hipLaunchKernelGGL(normalizedValTextureTest<T>, dim3(1,1,1), dim3(SIZE,1,1), 0, 0, SIZE, dOutputData);

    float *hOutputData = new float[SIZE];
    HIPCHECK(hipMemcpy(hOutputData, dOutputData, (sizeof(float)*SIZE), hipMemcpyDeviceToHost));

    bool testResult = true;
    for(int i = 0; i < SIZE; i++)
    {
        float expected = getNormalizedValue(float(hData[i]), desc);
        if(expected != hOutputData[i])
        {
            printf("mismatch at index:%d output:%f expected:%f\n",i,hOutputData[i],expected);
            testResult = false;
            break;
        }
    }

    HIPCHECK(hipFreeArray(dData));
    HIPCHECK(hipFree(dOutputData));
    delete [] hOutputData;
    return testResult;
}

int main(int argc, char** argv)
{
    int device = 0;
    bool status = true;
    HIPCHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    HIPCHECK(hipGetDeviceProperties(&props, device));
    std::cout << "Device :: " << props.name << std::endl;
    #ifdef __HIP_PLATFORM_AMD__
    std::cout << "Arch - AMD GPU :: " << props.gcnArch << std::endl;
    #endif
    
    status &= textureTest<char>          (&texc);
    status &= textureTest<unsigned char> (&texuc);
    status &= textureTest<short>         (&texs);
    status &= textureTest<unsigned short>(&texus);

    if(status){
        passed();
    }
    else{
        failed("checks failed!");
    }
}

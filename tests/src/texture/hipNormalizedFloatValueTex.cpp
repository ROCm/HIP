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

#define SIZE 10
texture<float, hipTextureType1D, hipReadModeElementType> textureNormalizedVal_1D;

__global__ void normalizedValTextureTest(unsigned int numElements, float* pDst)
{
    unsigned int elementID = hipThreadIdx_x;
    if(elementID >= numElements)
	    return;
    float coord =(float) elementID/(numElements-1);
    pDst[elementID] = tex1D(textureNormalizedVal_1D, coord);
}

template<typename T>
bool textureTest(enum hipArray_Format texFormat)
{
    T hData[] = {65, 66, 67, 68, 69, 70, 71, 72,73,74};
    T *dData = NULL;
    HIPCHECK(hipMalloc((void **) &dData, sizeof(T)*SIZE));
    HIPCHECK(hipMemcpyHtoD((hipDeviceptr_t)dData, hData, sizeof(T)*SIZE));
    
    textureReference* texRef = &textureNormalizedVal_1D;
    HIPCHECK(hipTexRefSetAddressMode(texRef, 0, hipAddressModeClamp));
    HIPCHECK(hipTexRefSetAddressMode(texRef, 1, hipAddressModeClamp));
    HIPCHECK(hipTexRefSetFilterMode(texRef, hipFilterModePoint));
    HIPCHECK(hipTexRefSetFlags(texRef, HIP_TRSF_NORMALIZED_COORDINATES)); 
    HIPCHECK(hipTexRefSetFormat(texRef, texFormat, 1));
    
    HIP_ARRAY_DESCRIPTOR desc;
    desc.Width = SIZE;
    desc.Height = 1;
    desc.Format = texFormat;
    desc.NumChannels = 1;
    HIPCHECK(hipTexRefSetAddress2D(texRef, &desc, (hipDeviceptr_t)dData, sizeof(T)*SIZE));
    
    bool testResult = true;
    float *dOutputData = NULL;
    HIPCHECK(hipMalloc((void **) &dOutputData, sizeof(float)*SIZE));
 
    hipLaunchKernelGGL(HIP_KERNEL_NAME(normalizedValTextureTest), dim3(1,1,1), dim3(SIZE,1,1), 0, 0, SIZE, dOutputData);

    float *hOutputData = new float[SIZE];
    HIPCHECK(hipMemcpyDtoH(hOutputData, (hipDeviceptr_t)dOutputData, (sizeof(float)*SIZE)));
    
    for(int i = 0; i < SIZE; i++)
    {
    	if((float)hData[i]/texFormatToSize[texFormat] != hOutputData[i])
        {
	    printf("mismatch at index:%d for texType:%d output:%f\n",i,texFormat,hOutputData[i]);
            testResult = false;
	    break;
        }
    }
    hipFree(dData);
    hipFree(dOutputData);
    hipUnbindTexture(textureNormalizedVal_1D);
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
    #ifdef __HIP_PLATFORM_HCC__
    std::cout << "Arch - AMD GPU :: " << props.gcnArch << std::endl;
    #endif
    
    status &= textureTest<char>          (HIP_AD_FORMAT_SIGNED_INT8);
    status &= textureTest<unsigned char> (HIP_AD_FORMAT_UNSIGNED_INT8);
    status &= textureTest<short>         (HIP_AD_FORMAT_SIGNED_INT16);
    status &= textureTest<unsigned short>(HIP_AD_FORMAT_UNSIGNED_INT16);
    status &= textureTest<float>         (HIP_AD_FORMAT_FLOAT);
	
    if(status){
        passed();
    }
    else{
        failed("checks failed!");
    }
}

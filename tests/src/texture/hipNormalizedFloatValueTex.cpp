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
 * BUILD_CMD: normalizedVal_kernel.code %hc --genco %S/hipNormalizedVal_kernel.cpp -o normalizedVal_kernel.code
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include <unistd.h>
#define fileName "normalizedVal_kernel.code"
#define SIZE 10
texture<float, hipTextureType1D, hipReadModeElementType> textureNormalizedVal_1D;

template<typename T>
bool textureTest(hipArray_Format texFormat)
{
    T hData[] = {65, 66, 67, 68, 69, 70, 71, 72,73,74};
    T *dData = NULL;
    HIPCHECK(hipMalloc((void **) &dData, sizeof(T)*SIZE));
    HIPCHECK(hipMemcpyHtoD((hipDeviceptr_t)dData, hData, sizeof(T)*SIZE));

    hipModule_t Module;
    HIPCHECK(hipModuleLoad(&Module, fileName));
    
    hipTexRef texRef ;
    HIPCHECK(hipModuleGetTexRef(&texRef, Module, "textureNormalizedVal_1D"));
 
    HIPCHECK(hipTexRefSetAddressMode(texRef, 0, HIP_TR_ADDRESS_MODE_CLAMP));
    HIPCHECK(hipTexRefSetAddressMode(texRef, 1, HIP_TR_ADDRESS_MODE_CLAMP));
    HIPCHECK(hipTexRefSetFilterMode(texRef, HIP_TR_FILTER_MODE_POINT));
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

    struct {
        unsigned int _Ad;
        float * _Bd;
    } args;
    args._Ad = SIZE;
    args._Bd = dOutputData;

    size_t sizeTemp = sizeof(args);

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &sizeTemp, HIP_LAUNCH_PARAM_END};

    hipFunction_t Function;
    HIPCHECK(hipModuleGetFunction(&Function, Module, "normalizedValTextureTest"));
    HIPCHECK(hipModuleLaunchKernel(Function, 1,1,1, SIZE,1,1, 0, 0, NULL, (void**)&config));
    HIPCHECK(hipDeviceSynchronize());
    float *hOutputData = new float[SIZE];
    HIPCHECK(hipMemcpyDtoH(hOutputData, (hipDeviceptr_t)dOutputData, (sizeof(float)*SIZE)));
    
    for(int i = 0; i < SIZE; i++)
    {
        int size;
        switch(texFormat){
            case HIP_AD_FORMAT_UNSIGNED_INT8:
               size = UCHAR_MAX;
            break;
            case HIP_AD_FORMAT_UNSIGNED_INT16:
               size = USHRT_MAX;
            break;
            case HIP_AD_FORMAT_SIGNED_INT8:
               size = SCHAR_MAX;
            break;
            case HIP_AD_FORMAT_SIGNED_INT16:
               size = SHRT_MAX;
            break;
            case HIP_AD_FORMAT_FLOAT:
            case HIP_AD_FORMAT_UNSIGNED_INT32:
            case HIP_AD_FORMAT_SIGNED_INT32:
            case HIP_AD_FORMAT_HALF:
               size = 1;
            break;
        }
        if((float)hData[i]/size != hOutputData[i]){
           printf("mismatch at index:%d for texType:%d output:%f\n",i,texFormat,hOutputData[i]);
           testResult = false;
           break;
        }
    }
    hipFree(dData);
    hipFree(dOutputData);
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
    /*status &= textureTest<unsigned char> (HIP_AD_FORMAT_UNSIGNED_INT8);
    status &= textureTest<short>         (HIP_AD_FORMAT_SIGNED_INT16);
    status &= textureTest<unsigned short>(HIP_AD_FORMAT_UNSIGNED_INT16);
    status &= textureTest<float>         (HIP_AD_FORMAT_FLOAT);*/
	
    if(status){
        passed();
    }
    else{
        failed("checks failed!");
    }
}

/*
Copyright (c) 2015-Present Advanced Micro Devices, Inc. All rights reserved.

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
 * RUN: %t
 * HIT_END
 */
#include <stdlib.h>
#include <stdio.h>

#include <hip/hip_runtime.h>
#include "test_common.h"

#define R 8 //rows, height
#define C 8 //columns, width


texture<int, hipTextureType2D,hipReadModeElementType> tex;

texture<int, hipTextureType1D, hipReadModeElementType>tex_1D;
bool testResult = true;

void runTest(void);

int main(int argc, char** argv) {
    runTest();

    if (testResult) {
        passed();
    } else {
        exit(EXIT_FAILURE);
    }
}


void runTest()
{
int val[R][C],val1[64],i,j;
size_t offset;
void *dev_ptr1D;

for(i=0;i<R;i++)
    for(j=0;j<C;j++)
	 val[i][j]=(i+1)*(j+1);


hipChannelFormatDesc chan_test,chan_desc=hipCreateChannelDesc(32,0,0,0,hipChannelFormatKindSigned);

hipArray *hip_arr;
HIPCHECK(hipMallocArray(&hip_arr, &chan_desc,C,R,0));

HIPCHECK(hipMemcpyToArray(hip_arr,0,0, val, R*C*sizeof(int), hipMemcpyHostToDevice));

HIP_ARRAY_DESCRIPTOR arr_desc;

arr_desc.format=HIP_AD_FORMAT_FLOAT;
arr_desc.width=R;
arr_desc.height=C;

hip_arr->textureType=hipTextureType2D;

tex.addressMode[0]=hipAddressModeWrap;
tex.addressMode[1]=hipAddressModeWrap;
tex.filterMode=hipFilterModePoint;
tex.normalized=0;

HIPCHECK(hipTexRefSetFlags(&tex, hipReadModeNormalizedFloat));
if(tex.normalized == 1)
   testResult=true;
else
   {
   testResult=false;
   printf("hipTexRefSetFlags() api didn't work as expected.");
    }


HIPCHECK(hipTexRefSetFilterMode(&tex, hipFilterModeLinear));

if(tex.filterMode == 1)
   testResult=true;
else
   {
   testResult=false;
   printf("hipTexRefSetFilterMode() api didn't work as expected.");
   }


HIPCHECK(hipTexRefSetAddressMode(&tex, 0,hipAddressModeClamp));
if(tex.addressMode[0] == 1)
   testResult=true;
else
   {	
   testResult=false;
   printf("hipTexRefSetAddressMode() api didn't work as expected.");
   }

HIPCHECK(hipTexRefSetFormat(&tex, HIP_AD_FORMAT_HALF, 4));

if(tex.format == HIP_AD_FORMAT_HALF)
  testResult=true;
else
   {
   testResult=false;	
   printf("hipTexRefSetFormat() api didn't work as expected.");
  }


HIPCHECK(hipTexRefSetArray(&tex, hip_arr, 0));

HIPCHECK(hipTexRefSetAddress2D(&tex, &arr_desc, hip_arr->data, C*sizeof(int) ));

//The following code part is specifically to test hipTexRefSetAddress() api


for(i=0;i<64;i++)
   val1[i]=i;

tex_1D.addressMode[0]=hipAddressModeWrap;
tex_1D.filterMode=hipFilterModePoint;
tex_1D.normalized=0;
 
 HIPCHECK(hipMalloc(&dev_ptr1D, 64*sizeof(int)));
 
 HIPCHECK(hipMemcpy(dev_ptr1D, val1, 64*sizeof(int), hipMemcpyHostToDevice));
 offset=0;
 HIPCHECK(hipTexRefSetAddress(&offset , &tex_1D, dev_ptr1D, 64*sizeof(int)));
 
 HIPCHECK(hipFree(dev_ptr1D));
//hipFreeArray() has issue in implementations hence commented
//HIPCHECK(hipFreeArray(hip_arr));
}

/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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
 * RUN: %t
 * HIT_END
 */
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <hip/hip_runtime.h>
#include "test_common.h"

using namespace std;
#define HIP_ENABLE_PRINTF
#define R 8 //rows, height
#define C 8 //columns, width


texture<int, hipTextureType2D,hipReadModeElementType> tex;

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

string out;
int val[R][C],i,j;
size_t offset;

for(i=0;i<R;i++)
    for(j=0;j<C;j++)
	{
	 val[i][j]=(i+1)*(j+1);
	}


hipChannelFormatDesc chan_test,chan_desc=hipCreateChannelDesc(32,0,0,0,hipChannelFormatKindSigned);

hipArray *hipArray;
hipMallocArray(&hipArray, &chan_desc,C,R,0);


hipMemcpyToArray(hipArray,0,0, val, R*C*sizeof(int), hipMemcpyHostToDevice);

tex.addressMode[0]=hipAddressModeWrap;
tex.addressMode[1]=hipAddressModeWrap;
tex.filterMode=hipFilterModePoint;
tex.normalized=0;

hipBindTextureToArray(&tex, hipArray, &chan_desc);

out=hipGetErrorString(hipGetTextureAlignmentOffset(&offset,&tex));
if(strcmp(out.c_str(),"hipSuccess")==0)
  testResult=true;
else
  testResult=false;


out=hipGetErrorString(hipUnbindTexture(&tex));

hipFree(hipArray);
}

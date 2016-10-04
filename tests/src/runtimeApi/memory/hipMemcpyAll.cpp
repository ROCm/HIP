/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include<iostream>
#include<assert.h>
#include"test_common.h"

#define len 1024*1024
#define size len * sizeof(float)

template<typename T>
void hmemset(T *ptr, T value)
{
	for(int i=0;i<len;i++){
		ptr[i] = value;
	}
}

int main(){

int num;
hipGetDeviceCount(&num);
if(num < 2)
{
    printf ("warning: Not enough GPUs to run the test, exiting without running.\n");
    passed();
    return 0;
}

float *h0, *h1;
float *ph0, *ph1;
float *d0, *d1;
h0 = new float[len];
h1 = new float[len];
hmemset(h0, 1.0f);
int gpu0 = 0, gpu1 = 1;
hipSetDevice(gpu0);
hipHostMalloc((void**)&ph0, size);
hipMalloc(&d0, size);
hipSetDevice(gpu1);
hipHostMalloc((void**)&ph1, size);
hipMalloc(&d1, size);
hipSetDevice(gpu0);



hipMemcpy(h1, h0, size, hipMemcpyDefault);
hipMemcpy(ph0, h1, size, hipMemcpyDefault);
hipMemcpy(ph1, ph0, size, hipMemcpyDefault);
assert(h0[0] == ph1[0]);
hmemset(ph1, 0.0f);
hipMemcpy(h0, ph1, size, hipMemcpyDefault);
assert(h0[0] == 0.0f);




hipSetDevice(gpu0);
hmemset(ph0, 2.0f);
hipMemcpy(d0, ph0, size, hipMemcpyDefault);
hipMemcpy(h0, d0, size, hipMemcpyDefault);

assert(h0[0] == ph0[0]);
hmemset(h0, 3.0f);
hipMemcpy(d0, h0, size, hipMemcpyDefault);

hipMemcpy(ph0, d0, size, hipMemcpyDefault);

assert(h0[0] == ph0[0]);

hipSetDevice(gpu1);
hmemset(ph1, 2.0f);
hipMemcpy(d1, ph1, size, hipMemcpyDefault);

hipMemcpy(h1, d1, size, hipMemcpyDefault);

assert(h1[0] == ph1[0]);
hmemset(h1, 3.0f);
hipMemcpy(d1, h1, size, hipMemcpyDefault);

hipMemcpy(ph1, d1, size, hipMemcpyDefault);

assert(h1[0] == ph1[0]);

hipSetDevice(gpu0);
hmemset(ph0, 4.0f);
hipMemcpy(d0, ph0, size, hipMemcpyDefault);

hipMemcpy(ph0, d0, size, hipMemcpyDefault);

hipMemcpy(h0, d0, size, hipMemcpyDefault);

assert(ph0[0] == 4.0f);
assert(h0[0] == 4.0f);

hipSetDevice(gpu1);
hmemset(ph1, 5.0f);
hipMemcpy(d1, ph1, size, hipMemcpyDefault);

hipMemcpy(ph1, d1, size, hipMemcpyDefault);

hipMemcpy(h1, d1, size, hipMemcpyDefault);

assert(ph1[0] == 5.0f);
assert(h1[0] == 5.0f);

hipSetDevice(gpu0);
hipMemcpy(d0, ph1, size, hipMemcpyDefault);

hipMemcpy(d1, d0, size, hipMemcpyDefault);
passed();
}

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
 * RUN: %t --tests 0x1
 * RUN: %t --tests 0x2
 * HIT_END
 */

// TODO - bug if run both back-to-back

#include"test_common.h"
#include<malloc.h>

__global__ void Inc(hipLaunchParm lp, float *Ad){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Ad[tx] = Ad[tx] + float(1);
}


template<typename T>
void doMemCopy(size_t numElements, int offset, T *A, T *Bh, T *Bd) 
{
    A = A + offset;
    numElements -= offset;

    size_t sizeBytes = numElements * sizeof(T);

	HIPCHECK(hipHostRegister(A, sizeBytes, 0));


    // Reset
    for(size_t i=0;i<numElements;i++){
        A[i] = float(i);
        Bh[i] = 0.0f;
    }

    HIPCHECK(hipMemset(Bd, 13.0f, sizeBytes));


    // 
    HIPCHECK(hipMemcpy(Bd, A, sizeBytes,  hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(Bh, Bd, sizeBytes, hipMemcpyDeviceToHost));

    // Make sure the copy worked
    for(size_t i=0;i<numElements;i++){
        if (Bh[i] != A[i]) {
            printf ("mismatch at Bh[%zu]=%f, A[%zu]=%f\n", i, Bh[i], i, A[i]);
            failed("mismatch");
        };
    }

    HIPCHECK(hipHostUnregister(A));

}

int main(int argc, char *argv[])
{

    HipTest::parseStandardArguments(argc, argv, true);

    const size_t size = N * sizeof(float);

    if (p_tests & 0x1) { 
        float *A, **Ad;
        int num_devices;
        HIPCHECK(hipGetDeviceCount(&num_devices));
        Ad = new float*[num_devices];
        A = (float*)malloc(size);
        HIPCHECK(hipHostRegister(A, size, 0));


        for(int i=0;i<N;i++){
            A[i] = float(1);
        }


        for(int i=0;i<num_devices;i++){
            HIPCHECK(hipSetDevice(i));
            HIPCHECK(hipHostGetDevicePointer((void**)&Ad[i], A, 0));
        }

        // Reference the registered device pointer Ad from inside the kernel:
        for(int i=0;i<num_devices;i++){
            HIPCHECK(hipSetDevice(i));
            hipLaunchKernel(Inc, dim3(N/512), dim3(512), 0, 0, Ad[i]);

            HIPCHECK(hipDeviceSynchronize());
        }
        HIPASSERT(A[10] == 1.0f + float(num_devices));

        HIPCHECK(hipHostUnregister(A));

        free (A);
    }


    if (p_tests & 0x2) { 
        // Sensitize HIP bug if device does not match where the memory was registered.
        HIPCHECK(hipSetDevice(0));


        float * A = (float*)malloc(size);

        // Copy to B, this should be optimal pinned malloc copy:
        // Note we are using the host pointer here:
        float *Bh, *Bd;
        Bh = (float*)malloc(size);
        HIPCHECK(hipMalloc(&Bd, size));

        // TODO - change to 256:
#define OFFSETS_TO_TRY 1 
        assert (N>OFFSETS_TO_TRY);
        for (size_t i=0; i<OFFSETS_TO_TRY; i++) {
            doMemCopy(N, i, A, Bh, Bd);
        }



        free (A);

    }



	passed();
}

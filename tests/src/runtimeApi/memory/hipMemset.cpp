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
// Simple test for memset.
// Also serves as a template for other tests.

/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * //Small copy
 * TEST: %t -N 10    --memsetval 0x42 --memsetD32val 0x101 --memsetD16val 0x10 --memsetD8val 0x1
 * // Oddball size
 * TEST: %t -N 10013 --memsetval 0x5a --memsetD32val 0xDEADBEEF --memsetD16val 0xDEAD --memsetD8val 0xDE
 * // Big copy
 * TEST: %t -N 256M  --memsetval 0xa6 --memsetD32val 0xCAFEBABE --memsetD16val 0xCAFE --memsetD8val 0xCA
 * HIT_END
 */

#include "hip/hip_runtime.h"
#include "test_common.h"

bool testhipMemset(int memsetval,int p_gpuDevice)
{
    size_t Nbytes = N*sizeof(char);
    printf ("testhipMemset N=%zu  memsetval=%2x device=%d\n", N, memsetval, p_gpuDevice);
    char *A_d;
    char *A_h;
    bool testResult = true;

    HIPCHECK ( hipMalloc(&A_d, Nbytes) );
    A_h = (char*)malloc(Nbytes);
    HIPCHECK ( hipMemset(A_d, memsetval, Nbytes) );
    HIPCHECK ( hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetval) {
            testResult = false;
            printf("mismatch at index:%d computed:%02x, memsetval:%02x\n", i, (int)A_h[i], (int)memsetval);
            break;
        }
    }
    HIPCHECK(hipFree(A_d));
    free(A_h);
    return testResult;
}

bool testhipMemsetD32(int memsetD32val,int p_gpuDevice)
{
    size_t Nbytes = N*sizeof(int);
    printf ("testhipMemsetD32 N=%zu  memsetD32val=%8x device=%d\n", N, memsetD32val, p_gpuDevice);
    int *A_d;
    int *A_h;
    bool testResult = true;

    HIPCHECK ( hipMalloc(&A_d, Nbytes) );
    A_h = (int*)malloc(Nbytes);
    HIPCHECK ( hipMemsetD32((hipDeviceptr_t)A_d, memsetD32val, N) );
    HIPCHECK ( hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetD32val) {
            testResult = false;            printf("mismatch at index:%d computed:%08x, memsetD32val:%08x\n", i, A_h[i], memsetD32val);
            break;
        }
    }
    HIPCHECK(hipFree(A_d));
    free(A_h);
    return testResult;
}

bool testhipMemsetD16(short memsetD16val,int p_gpuDevice)
{
    size_t Nbytes = N*sizeof(int);
    printf ("testhipMemsetD16 N=%zu  memsetD16val=%4x device=%d\n", N, memsetD16val, p_gpuDevice);
    short *A_d;
    short *A_h;
    bool testResult = true;

    HIPCHECK ( hipMalloc(&A_d, Nbytes) );
    A_h = (short*)malloc(Nbytes);
    HIPCHECK ( hipMemsetD16((hipDeviceptr_t)A_d, memsetD16val, N) );
    HIPCHECK ( hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetD16val) {
            testResult = false;            printf("mismatch at index:%d computed:%08x, memsetD16val:%08x\n", i, A_h[i], memsetD32val);
            break;
        }
    }
    HIPCHECK(hipFree(A_d));
    free(A_h);
    return testResult;
}

bool testhipMemsetD8(char memsetD8val,int p_gpuDevice)
{
    size_t Nbytes = N*sizeof(int);
    printf ("testhipMemsetD8 N=%zu  memsetD8val=%4x device=%d\n", N, memsetD8val, p_gpuDevice);
    char *A_d;
    char *A_h;
    bool testResult = true;

    HIPCHECK ( hipMalloc(&A_d, Nbytes) );
    A_h = (char*)malloc(Nbytes);
    HIPCHECK ( hipMemsetD8((hipDeviceptr_t)A_d, memsetD8val, N) );
    HIPCHECK ( hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetD8val) {
            testResult = false;            printf("mismatch at index:%d computed:%08x, memsetD8val:%08x\n", i, A_h[i], memsetD8val);
            break;
        }
    }
    HIPCHECK(hipFree(A_d));
    free(A_h);
    return testResult;
}

bool testhipMemsetAsync(int memsetval,int p_gpuDevice)
{
    size_t Nbytes = N*sizeof(int);
    printf ("testhipMemsetAsync N=%zu  memsetval=%2x device=%d\n", N, memsetval, p_gpuDevice);
    char *A_d;
    char *A_h;
    bool testResult = true;

    HIPCHECK ( hipMalloc((void**)&A_d, Nbytes) );
    A_h = (char*)malloc(Nbytes);
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK ( hipMemsetAsync(A_d, memsetval, Nbytes, stream ));
    HIPCHECK ( hipStreamSynchronize(stream));
    HIPCHECK ( hipMemcpy(A_h, (void*)A_d, Nbytes, hipMemcpyDeviceToHost));

    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetval) {
            testResult = false;
            printf("mismatch at index:%d computed:%02x, memsetval:%02x\n", i, (int)A_h[i], (int)memsetval);
            break;
        }
    }
    HIPCHECK(hipFree((void*)A_d));
    HIPCHECK(hipStreamDestroy(stream));
    free(A_h);
    return testResult;
}

bool testhipMemsetD32Async(int memsetD32val,int p_gpuDevice)
{
    size_t Nbytes = N*sizeof(int);
    printf ("testhipMemsetD32Async N=%zu  memsetval=%8x device=%d\n", N, memsetD32val, p_gpuDevice);
    int *A_d;
    int *A_h;
    bool testResult = true;

    HIPCHECK ( hipMalloc((void**)&A_d, Nbytes) );
    A_h = (int*)malloc(Nbytes);
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK ( hipMemsetD32Async((hipDeviceptr_t)A_d, memsetD32val, N, stream ));
    HIPCHECK ( hipStreamSynchronize(stream));
    HIPCHECK ( hipMemcpy(A_h, (void*)A_d, Nbytes, hipMemcpyDeviceToHost));

    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetD32val) {
            testResult = false;
            printf("mismatch at index:%d computed:%02x, memsetD32val:%02x\n", i, A_h[i], memsetD32val);
            break;
        }
    }
    HIPCHECK(hipFree((void*)A_d));
    HIPCHECK(hipStreamDestroy(stream));
    free(A_h);
    return testResult;
}

bool testhipMemsetD16Async(short memsetD16val,int p_gpuDevice)
{
    size_t Nbytes = N*sizeof(int);
    printf ("testhipMemsetD16Async N=%zu  memsetval=%8x device=%d\n", N, memsetD16val, p_gpuDevice);
    short *A_d;
    short *A_h;
    bool testResult = true;

    HIPCHECK ( hipMalloc((void**)&A_d, Nbytes) );
    A_h = (short*)malloc(Nbytes);
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK ( hipMemsetD16Async((hipDeviceptr_t)A_d, memsetD16val, N, stream ));
    HIPCHECK ( hipStreamSynchronize(stream));
    HIPCHECK ( hipMemcpy(A_h, (void*)A_d, Nbytes, hipMemcpyDeviceToHost));

    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetD16val) {
            testResult = false;
            printf("mismatch at index:%d computed:%02x, memsetD16val:%02x\n", i, A_h[i], memsetD16val);
            break;
        }
    }
    HIPCHECK(hipFree((void*)A_d));
    HIPCHECK(hipStreamDestroy(stream));
    free(A_h);
    return testResult;
}

bool testhipMemsetD8Async(char memsetD8val,int p_gpuDevice)
{
    size_t Nbytes = N*sizeof(int);
    printf ("testhipMemsetD8Async N=%zu  memsetD8val=%2x device=%d\n", N, memsetD8val, p_gpuDevice);
    char *A_d;
    char *A_h;
    bool testResult = true;

    HIPCHECK ( hipMalloc((void**)&A_d, Nbytes) );
    A_h = (char*)malloc(Nbytes);
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    HIPCHECK ( hipMemsetD8Async((hipDeviceptr_t)A_d, memsetD8val, N, stream ));
    HIPCHECK ( hipStreamSynchronize(stream));
    HIPCHECK ( hipMemcpy(A_h, (void*)A_d, Nbytes, hipMemcpyDeviceToHost));

    for (int i=0; i<N; i++) {
        if (A_h[i] != memsetD8val) {
            testResult = false;
            printf("mismatch at index:%d computed:%02x, memsetD8val:%02x\n", i, A_h[i], memsetD8val);
            break;
        }
    }
    HIPCHECK(hipFree((void*)A_d));
    HIPCHECK(hipStreamDestroy(stream));
    free(A_h);
    return testResult;
}

int main(int argc, char *argv[])
{
    HipTest::parseStandardArguments(argc, argv, true);
    bool testResult = true;
    HIPCHECK(hipSetDevice(p_gpuDevice));
    testResult &= testhipMemset(memsetval, p_gpuDevice);
    testResult &= testhipMemsetAsync(memsetval, p_gpuDevice);
    testResult &= testhipMemsetD32(memsetD32val, p_gpuDevice);
    testResult &= testhipMemsetD32Async(memsetD32val, p_gpuDevice);
    testResult &= testhipMemsetD16(memsetD16val, p_gpuDevice);
    testResult &= testhipMemsetD16Async(memsetD16val, p_gpuDevice);
    testResult &= testhipMemsetD8(memsetD8val, p_gpuDevice);
    testResult &= testhipMemsetD8Async(memsetD8val, p_gpuDevice);
    if (testResult) passed();
    failed("Output Mismatch\n"); 
}

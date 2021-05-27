/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/device_functions.h>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#define NUM_TESTS 65

#define HI_INT 0xfacefeed
#define LO_INT 0xdeadbeef

__global__ void funnelshift_kernel(unsigned int* l_out, unsigned int* lc_out,
        unsigned int* r_out, unsigned int* rc_out) {

    for (int i = 0; i < NUM_TESTS; i++) {
        l_out[i] = __funnelshift_l(LO_INT, HI_INT, i);
        lc_out[i] = __funnelshift_lc(LO_INT, HI_INT, i);
        r_out[i] = __funnelshift_r(LO_INT, HI_INT, i);
        rc_out[i] = __funnelshift_rc(LO_INT, HI_INT, i);
    }
}

static unsigned int cpu_funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift)
{
    // Concatenate hi:lo
    uint64_t val = hi;
    val <<= 32;
    val |= lo;
    // left shift by intput & 31
    val <<= (shift & 31);
    // pull out upper 32 bits and return them
    val >>= 32;
    return val & 0xffffffff;
}

static unsigned int cpu_funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    // Concatenate hi:lo
    uint64_t val = hi;
    val <<= 32;
    val |= lo;
    // left shift by min(input,32)
    if (shift > 32)
        shift = 32;
    val <<= shift;
    // pull out upper 32 bits and return them
    val >>= 32;
    return val & 0xffffffff;
}

static unsigned int cpu_funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)
{
    // Concatenate hi:lo
    uint64_t val = hi;
    val <<= 32;
    val |= lo;
    // right shift by intput & 31
    val >>= (shift & 31);
    // return lower 32 bits
    return val & 0xffffffff;
}

static unsigned int cpu_funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift)
{
    // Concatenate hi:lo
    uint64_t val = hi;
    val <<= 32;
    val |= lo;
    // left shift by min(input, 32)
    if (shift > 32)
        shift = 32;
    val >>= shift;
    // return lower 32 bits
    return val & 0xffffffff;
}

using namespace std;

int main() {
    unsigned int *host_l_output;
    unsigned int *host_lc_output;
    unsigned int *host_r_output;
    unsigned int *host_rc_output;

    unsigned int *device_l_output;
    unsigned int *device_lc_output;
    unsigned int *device_r_output;
    unsigned int *device_rc_output;

    unsigned int *golden_l;
    unsigned int *golden_lc;
    unsigned int *golden_r;
    unsigned int *golden_rc;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    cout << "hip Device prop succeeded " << endl;


    int i;
    int errors;

    host_l_output = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    host_lc_output = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    host_r_output = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    host_rc_output = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));

    golden_l = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    golden_lc = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    golden_r = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    golden_rc = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));

    for (int i = 0; i < NUM_TESTS; i++) {
        golden_l[i] = cpu_funnelshift_l(LO_INT, HI_INT, i);
        golden_lc[i] = cpu_funnelshift_lc(LO_INT, HI_INT, i);
        golden_r[i] = cpu_funnelshift_r(LO_INT, HI_INT, i);
        golden_rc[i] = cpu_funnelshift_rc(LO_INT, HI_INT, i);
    }

    HIP_ASSERT(hipMalloc((void**)&device_l_output, NUM_TESTS * sizeof(unsigned int)));
    HIP_ASSERT(hipMalloc((void**)&device_lc_output, NUM_TESTS * sizeof(unsigned int)));
    HIP_ASSERT(hipMalloc((void**)&device_r_output, NUM_TESTS * sizeof(unsigned int)));
    HIP_ASSERT(hipMalloc((void**)&device_rc_output, NUM_TESTS * sizeof(unsigned int)));

    hipLaunchKernelGGL(funnelshift_kernel, dim3(1), dim3(1), 0, 0,
            device_l_output, device_lc_output, device_r_output,
            device_rc_output);

    HIP_ASSERT(hipMemcpy(host_l_output, device_l_output, NUM_TESTS * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(host_lc_output, device_lc_output, NUM_TESTS * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(host_r_output, device_r_output, NUM_TESTS * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(host_rc_output, device_rc_output, NUM_TESTS * sizeof(unsigned int), hipMemcpyDeviceToHost));

    // verify the results
    errors = 0;
    printf("HI val: 0x%x\n", HI_INT);
    printf("LO val: 0x%x\n", LO_INT);

    for (i = 0; i < NUM_TESTS; i++) {
        printf("gpu_funnelshift_l(%d) = 0x%x, cpu_funnelshift_l(%d) = 0x%x\n",
                i, host_l_output[i], i, golden_l[i]);
        if (host_l_output[i] != golden_l[i]) {
            errors++;
            printf("\tERROR!\n");
        }
    }
    if (errors != 0) {
        cout << "FAILED: funnelshift_l" << endl;
        return -1;
    } else {
        cout << "funnelshift_l checked!" << endl;
    }

    errors = 0;
    for (i = 0; i < NUM_TESTS; i++) {
        printf("gpu_funnelshift_lc(%d) = 0x%x, cpu_funnelshift_lc(%d) = 0x%x\n",
                i, host_lc_output[i], i, golden_lc[i]);
        if (host_lc_output[i] != golden_lc[i]) {
            errors++;
            printf("\tERROR!\n");
        }
    }
    if (errors != 0) {
        cout << "FAILED: funnelshift_lc" << endl;
        return -1;
    } else {
        cout << "funnelshift_lc checked!" << endl;
    }

    errors = 0;
    for (i = 0; i < NUM_TESTS; i++) {
        printf("gpu_funnelshift_r(%d) = 0x%x, cpu_funnelshift_r(%d) = 0x%x\n",
                i, host_r_output[i], i, golden_r[i]);
        if (host_r_output[i] != golden_r[i]) {
            errors++;
            printf("\tERROR!\n");
        }
    }
    if (errors != 0) {
        cout << "FAILED: funnelshift_r" << endl;
        return -1;
    } else {
        cout << "funnelshift_r checked!" << endl;
    }

    errors = 0;
    for (i = 0; i < NUM_TESTS; i++) {
        printf("gpu_funnelshift_rc(%d) = 0x%x, cpu_funnelshift_rc(%d) = 0x%x\n",
                i, host_rc_output[i], i, golden_rc[i]);
        if (host_rc_output[i] != golden_rc[i]) {
            errors++;
            printf("\tERROR!\n");
        }
    }
    if (errors != 0) {
        cout << "FAILED: funnelshift_rc" << endl;
        return -1;
    } else {
        cout << "funnelshift_rc checked!" << endl;
    }
    errors = 0;

    cout << "funnelshift tests PASSED!" << endl;

    HIP_ASSERT(hipFree(device_l_output));
    HIP_ASSERT(hipFree(device_lc_output));
    HIP_ASSERT(hipFree(device_r_output));
    HIP_ASSERT(hipFree(device_rc_output));

    free(host_l_output);
    free(host_lc_output);
    free(host_r_output);
    free(host_rc_output);

    return errors;
}

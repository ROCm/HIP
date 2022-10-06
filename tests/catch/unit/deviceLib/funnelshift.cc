/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>
#include <hip/device_functions.h>


#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>


#define NUM_TESTS 65

#define HI_INT 0xfacefeed
#define LO_INT 0xdeadbeef

__global__ void funnelshift_kernel(unsigned int* l_out, unsigned int* lc_out, unsigned int* r_out,
                                   unsigned int* rc_out) {
  for (int i = 0; i < NUM_TESTS; i++) {
    l_out[i] = __funnelshift_l(LO_INT, HI_INT, i);
    lc_out[i] = __funnelshift_lc(LO_INT, HI_INT, i);
    r_out[i] = __funnelshift_r(LO_INT, HI_INT, i);
    rc_out[i] = __funnelshift_rc(LO_INT, HI_INT, i);
  }
}

static unsigned int cpu_funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift) {
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

static unsigned int cpu_funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift) {
  // Concatenate hi:lo
  uint64_t val = hi;
  val <<= 32;
  val |= lo;
  // left shift by min(input,32)
  if (shift > 32) shift = 32;
  val <<= shift;
  // pull out upper 32 bits and return them
  val >>= 32;
  return val & 0xffffffff;
}

static unsigned int cpu_funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift) {
  // Concatenate hi:lo
  uint64_t val = hi;
  val <<= 32;
  val |= lo;
  // right shift by intput & 31
  val >>= (shift & 31);
  // return lower 32 bits
  return val & 0xffffffff;
}

static unsigned int cpu_funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift) {
  // Concatenate hi:lo
  uint64_t val = hi;
  val <<= 32;
  val |= lo;
  // left shift by min(input, 32)
  if (shift > 32) shift = 32;
  val >>= shift;
  // return lower 32 bits
  return val & 0xffffffff;
}

TEST_CASE("Unit_funnelshift") {
  unsigned int* host_l_output;
  unsigned int* host_lc_output;
  unsigned int* host_r_output;
  unsigned int* host_rc_output;

  unsigned int* device_l_output;
  unsigned int* device_lc_output;
  unsigned int* device_r_output;
  unsigned int* device_rc_output;

  unsigned int* golden_l;
  unsigned int* golden_lc;
  unsigned int* golden_r;
  unsigned int* golden_rc;

  hipDeviceProp_t devProp;
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  INFO("System minor : " << devProp.minor);
  INFO("System major : " << devProp.major);
  INFO("agent prop name : " << devProp.name);

  INFO("hip Device prop succeeded");


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

  HIP_CHECK(hipMalloc((void**)&device_l_output, NUM_TESTS * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&device_lc_output, NUM_TESTS * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&device_r_output, NUM_TESTS * sizeof(unsigned int)));
  HIP_CHECK(hipMalloc((void**)&device_rc_output, NUM_TESTS * sizeof(unsigned int)));

  hipLaunchKernelGGL(funnelshift_kernel, dim3(1), dim3(1), 0, 0, device_l_output, device_lc_output,
                     device_r_output, device_rc_output);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(host_l_output, device_l_output, NUM_TESTS * sizeof(unsigned int),
                       hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(host_lc_output, device_lc_output, NUM_TESTS * sizeof(unsigned int),
                       hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(host_r_output, device_r_output, NUM_TESTS * sizeof(unsigned int),
                       hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(host_rc_output, device_rc_output, NUM_TESTS * sizeof(unsigned int),
                       hipMemcpyDeviceToHost));

  // verify the results
  errors = 0;
  printf("HI val: 0x%x\n", HI_INT);
  printf("LO val: 0x%x\n", LO_INT);

  for (i = 0; i < NUM_TESTS; i++) {
    if (host_l_output[i] != golden_l[i]) {
      errors++;
      INFO("Mismatch: " << host_l_output[i] << " - " << golden_l[i]);
      break;
    }
  }


  for (i = 0; i < NUM_TESTS && errors == 0; i++) {
    if (host_lc_output[i] != golden_lc[i]) {
      errors++;
      INFO("Mismatch: " << host_lc_output[i] << " - " << golden_lc[i]);
      break;
    }
  }

  errors = 0;
  for (i = 0; i < NUM_TESTS && errors == 0; i++) {
    if (host_r_output[i] != golden_r[i]) {
      errors++;
      INFO("Mismatch: " << host_r_output[i] << " - " << golden_r[i]);
      break;
    }
  }

  for (i = 0; i < NUM_TESTS && errors == 0; i++) {
    if (host_rc_output[i] != golden_rc[i]) {
      errors++;
      INFO("Mismatch: " << host_rc_output[i] << " - " << golden_rc[i]);
      break;
    }
  }

  HIP_CHECK(hipFree(device_l_output));
  HIP_CHECK(hipFree(device_lc_output));
  HIP_CHECK(hipFree(device_r_output));
  HIP_CHECK(hipFree(device_rc_output));

  free(host_l_output);
  free(host_lc_output);
  free(host_r_output);
  free(host_rc_output);

  REQUIRE(errors == 0);
}
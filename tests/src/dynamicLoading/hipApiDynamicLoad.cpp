/*
Copyright (c) 2020-present Advanced Micro Devices, Inc. All rights reserved.

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

/* Test is to load hip runtime using dlopen and get function pointer
 * using dlsym for hip apis using dlsym()
 * */

/* HIT_START
 * BUILD_CMD: bit_extract_kernel.code %hc --genco %S/bit_extract_kernel.cpp -o bit_extract_kernel.code EXCLUDE_HIP_PLATFORM nvidia
 * BUILD_CMD: %t %hc %S/%s -I%S/.. -o %T/%t -ldl EXCLUDE_HIP_PLATFORM nvidia EXCLUDE_HIP_LIB_TYPE static
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include <stdio.h>
#include <iostream>
#include <dlfcn.h>
#include <fstream>
#include <vector>

#define fileName "bit_extract_kernel.code"

#define LEN 64
#define SIZE LEN * sizeof(float)

int main(int argc, char* argv[]) {
  uint32_t *A_d, *C_d;
  uint32_t *A_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(uint32_t);

  void* handle = dlopen("libamdhip64.so", RTLD_LAZY);
  if (!handle) {
      std::cout << dlerror() << "\n";
      failed("hip runtime failed to load from dlopen\n");
  }

  std::cout << "hip runtime loaded using dlopen\n";

  void* sym_hipGetDevice = dlsym(handle, "hipGetDevice");
  void* sym_hipMalloc = dlsym(handle, "hipMalloc");
  void* sym_hipMemcpyHtoD = dlsym(handle, "hipMemcpyHtoD");
  void* sym_hipMemcpyDtoH = dlsym(handle, "hipMemcpyDtoH");
  void* sym_hipModuleLoad = dlsym(handle, "hipModuleLoad");
  void* sym_hipGetDeviceProperties = dlsym(handle, "hipGetDeviceProperties");
  void* sym_hipModuleGetFunction = dlsym(handle, "hipModuleGetFunction");
  void* sym_hipModuleLaunchKernel = dlsym(handle, "hipModuleLaunchKernel");

  dlclose(handle);
  hipError_t (*dyn_hipGetDevice)(hipDevice_t*, int) = reinterpret_cast
             <hipError_t (*)(hipDevice_t*, int)>(sym_hipGetDevice);

  hipError_t (*dyn_hipMalloc)(void**, uint32_t) = reinterpret_cast
             <hipError_t (*)(void**, uint32_t)>(sym_hipMalloc);

  hipError_t (*dyn_hipMemcpyHtoD)(hipDeviceptr_t, void*, size_t) = reinterpret_cast
             <hipError_t (*)(hipDeviceptr_t, void*, size_t)>(sym_hipMemcpyHtoD);

  hipError_t (*dyn_hipMemcpyDtoH)(void*, hipDeviceptr_t, size_t) = reinterpret_cast
             <hipError_t (*)(void*, hipDeviceptr_t, size_t)>(sym_hipMemcpyDtoH);

  hipError_t (*dyn_hipModuleLoad)(hipModule_t*, const char*) = reinterpret_cast
             <hipError_t (*)(hipModule_t*, const char*)>(sym_hipModuleLoad);

  hipError_t (*dyn_hipGetDeviceProperties)(hipDeviceProp_t*, int) = reinterpret_cast
             <hipError_t (*)(hipDeviceProp_t*, int)>(sym_hipGetDeviceProperties);

  hipError_t (*dyn_hipModuleGetFunction)(hipFunction_t*, hipModule_t, const char*) =
             reinterpret_cast<hipError_t (*)(hipFunction_t*, hipModule_t, const char*)>
             (sym_hipModuleGetFunction);

  hipError_t (*dyn_hipModuleLaunchKernel)(hipFunction_t, unsigned int, unsigned int,
             unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
             hipStream_t, void**, void**) = reinterpret_cast<hipError_t(*)
             (hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int,
             unsigned int, unsigned int, unsigned int, hipStream_t, void**, void**)>
             (sym_hipModuleLaunchKernel);

  hipDevice_t device;
  HIPCHECK(dyn_hipGetDevice(&device, 0));

  hipDeviceProp_t props;
  HIPCHECK(dyn_hipGetDeviceProperties(&props, device));
  printf("info: running on device #%d %s\n", device, props.name);
  printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
  A_h = reinterpret_cast<uint32_t*>(malloc(Nbytes));
  HIPASSERT(A_h != NULL);
  C_h = reinterpret_cast<uint32_t*>(malloc(Nbytes));
  HIPASSERT(C_h != NULL);

  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
  }

  printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
  HIPCHECK(dyn_hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));
  HIPCHECK(dyn_hipMalloc(reinterpret_cast<void**>(&C_d), Nbytes));

  printf("info: copy Host2Device\n");
  HIPCHECK(dyn_hipMemcpyHtoD((hipDeviceptr_t)(A_d), A_h, Nbytes));

  printf("info: launch 'bit_extract_kernel' \n");

  struct {
    void* _Cd;
    void* _Ad;
    size_t _N;
  } args;
  args._Cd = reinterpret_cast<void**> (C_d);
  args._Ad = reinterpret_cast<void**> (A_d);
  args._N = (size_t) N;
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  hipModule_t Module;
  HIPCHECK(dyn_hipModuleLoad(&Module, fileName));

  hipFunction_t Function;
  HIPCHECK(dyn_hipModuleGetFunction(&Function, Module, "bit_extract_kernel"));

  HIPCHECK(dyn_hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL,
                                     reinterpret_cast<void**>(&config)));

  printf("info: copy Device2Host\n");
  HIPCHECK(dyn_hipMemcpyDtoH(C_h, (hipDeviceptr_t)(C_d), Nbytes));

  printf("info: check result\n");
  for (size_t i = 0; i < N; i++) {
    unsigned Agold = ((A_h[i] & 0xf00) >> 8);
    if (C_h[i] != Agold) {
      fprintf(stderr, "mismatch detected.\n");
      printf("%zu: %08x =? %08x (Ain=%08x)\n", i, C_h[i], Agold, A_h[i]);
      failed("Test failed\n");
    }
  }
  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));
  free(A_h);
  free(C_h);
  passed();
}

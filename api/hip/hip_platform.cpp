/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>

#include "hip_internal.hpp"
#include "platform/program.hpp"
#include "platform/runtime.hpp"

constexpr unsigned __cudaFatMAGIC  = 0x1ee55a01;
constexpr unsigned __cudaFatMAGIC2 = 0x466243b1;
constexpr unsigned __cudaFatMAGIC3 = 0xba55ed50;

struct __CudaFatBinaryWrapper {
  unsigned int magic;
  unsigned int version;
  void*        binary;
  void*        dummy1;
};

struct __CudaFatBinaryHeader {
  unsigned int           magic;
  unsigned short         version;
  unsigned short         headerSize;
  unsigned long long int fatSize;
};

struct __CudaPartHeader{
  unsigned short         type;
  unsigned short         dummy1;
  unsigned int           headerSize;
  unsigned long long int partSize;
  unsigned long long int dummy2;
  unsigned int           dummy3;
  unsigned int           subarch;
};

extern "C" hipModule_t __cudaRegisterFatBinary(void* bundle)
{
  if (!amd::Runtime::initialized()) { // FIXME: fix initialization
    hipInit(0);
  }

  amd::Program* program = new amd::Program(*g_context);
  if (!program) return nullptr;

  struct __CudaFatBinaryWrapper* fbwrapper = (struct __CudaFatBinaryWrapper*)bundle;
  if (fbwrapper->magic != __cudaFatMAGIC2 || fbwrapper->version != 1) {
    return nullptr;
  }
  struct __CudaFatBinaryHeader* fbheader = (struct __CudaFatBinaryHeader*)fbwrapper->binary;
  if (fbheader->magic != __cudaFatMAGIC3 || fbheader->version != 1) {
    return nullptr;
  }
  struct __CudaPartHeader* pheader = (struct __CudaPartHeader*)(
      (uintptr_t)fbheader + fbheader->headerSize);
  struct __CudaPartHeader* end = (struct __CudaPartHeader*)(
      (uintptr_t)pheader + fbheader->fatSize);

  while (pheader < end) {
    if (true/*pheader->subarch == match a device in the context*/) {
      void *image = (void*)((uintptr_t)pheader + pheader->headerSize);
      size_t size = pheader->partSize;
      if (CL_SUCCESS != program->addDeviceProgram(*g_context->devices()[0], image, size) ||
        CL_SUCCESS != program->build(g_context->devices(), nullptr, nullptr, nullptr)) {
        return nullptr;
      }
      break;
    }
    pheader = (struct __CudaPartHeader*)(
        (uintptr_t)pheader + pheader->headerSize + pheader->partSize);
  }

  return reinterpret_cast<hipModule_t>(as_cl(program));
}

std::map<const void*, hipFunction_t> g_functions;


extern "C" void __cudaRegisterFunction(
  hipModule_t  module,
  const void*  hostFunction,
  char*        deviceFunction,
  const char*  deviceName,
  unsigned int threadLimit,
  uint3*       tid,
  uint3*       bid,
  dim3*        blockDim,
  dim3*        gridDim,
  int*         wSize)
{
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(module));

  const amd::Symbol* symbol = program->findSymbol(deviceName);
  if (!symbol) return;

  amd::Kernel* kernel = new amd::Kernel(*program, *symbol, deviceName);
  if (!kernel) return;

  // FIXME: not thread safe
  g_functions.insert(std::make_pair(hostFunction, reinterpret_cast<hipFunction_t>(as_cl(kernel))));
}

extern "C" void __cudaRegisterVar(
  hipModule_t module,
  char*       hostVar,
  char*       deviceVar,
  const char* deviceName,
  int         ext,
  int         size,
  int         constant,
  int         global)
{
}

extern "C" void __cudaUnregisterFatBinary(
  hipModule_t module
)
{
}

dim3 g_gridDim; // FIXME: place in execution stack
dim3 g_blockDim; // FIXME: place in execution stack
size_t g_sharedMem; // FIXME: place in execution stack
hipStream_t g_stream; // FIXME: place in execution stack

extern "C" hipError_t cudaConfigureCall(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem,
  hipStream_t stream)
{
  // FIXME: should push and new entry on the execution stack

  g_gridDim = gridDim;
  g_blockDim = blockDim;
  g_sharedMem = sharedMem;
  g_stream = stream;

  return hipSuccess;
}

char* g_arguments[1024]; // FIXME: needs to grow

extern "C" hipError_t cudaSetupArgument(
  const void *arg,
  size_t size,
  size_t offset)
{
  // FIXME: should modify the top of the execution stack

  ::memcpy(g_arguments + offset, arg, size);
  return hipSuccess;
}

extern "C" hipError_t cudaLaunch(const void *hostFunction)
{
  std::map<const void*, hipFunction_t>::iterator it;
  if ((it = g_functions.find(hostFunction)) == g_functions.end())
    return hipErrorUnknown;

  // FIXME: should pop an entry from the execution stack

  void *extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, g_arguments,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, 0 /* FIXME: not needed, but should be correct*/,
      HIP_LAUNCH_PARAM_END
    };

  return hipModuleLaunchKernel(it->second,
    g_gridDim.x, g_gridDim.y, g_gridDim.z,
    g_blockDim.x, g_blockDim.y, g_blockDim.z,
    g_sharedMem, g_stream, nullptr, extra);
}

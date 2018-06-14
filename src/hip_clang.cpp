/*
Copyright (c) 2018 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <unordered_map>
#include <string>

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"

constexpr unsigned __hipFatMAGIC2 = 0x48495046; // "HIPF"

#define CLANG_OFFLOAD_BUNDLER_MAGIC "__CLANG_OFFLOAD_BUNDLE__"
#define AMDGCN_AMDHSA_TRIPLE "hip-amdgcn-amd-amdhsa"

struct __ClangOffloadBundleDesc {
  uint64_t offset;
  uint64_t size;
  uint64_t tripleSize;
  const char triple[1];
};

struct __ClangOffloadBundleHeader {
  const char magic[sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1];
  uint64_t numBundles;
  __ClangOffloadBundleDesc desc[1];
};

struct __CudaFatBinaryWrapper {
  unsigned int                magic;
  unsigned int                version;
  __ClangOffloadBundleHeader* binary;
  void*                       unused;
};


extern "C" std::vector<hipModule_t>*
__hipRegisterFatBinary(const void* data)
{
  HIP_INIT();

  const __CudaFatBinaryWrapper* fbwrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(data);
  if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
    return nullptr;
  }

  const __ClangOffloadBundleHeader* header = fbwrapper->binary;
  std::string magic(reinterpret_cast<const char*>(header), sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC)) {
    return nullptr;
  }

  auto modules = new std::vector<hipModule_t>{g_deviceCnt};
  if (!modules) {
    return nullptr;
  }

  const __ClangOffloadBundleDesc* desc = &header->desc[0];
  for (uint64_t i = 0; i < header->numBundles; ++i,
       desc = reinterpret_cast<const __ClangOffloadBundleDesc*>(
           reinterpret_cast<uintptr_t>(&desc->triple[0]) + desc->tripleSize)) {

    std::string triple{&desc->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};
    if (triple.compare(AMDGCN_AMDHSA_TRIPLE))
      continue;

    std::string target{&desc->triple[sizeof(AMDGCN_AMDHSA_TRIPLE)],
      desc->tripleSize - sizeof(AMDGCN_AMDHSA_TRIPLE)};

    for (int deviceId = 0; deviceId < g_deviceCnt; ++deviceId) {
      hsa_agent_t agent = g_allAgents[deviceId + 1];

      char name[64] = {};
      hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
      if (target.compare(name)) {
         continue;
      }

      ihipModule_t* module = new ihipModule_t;
      if (!module) {
        continue;
      }

      hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                &module->executable);

      std::string image{reinterpret_cast<const char*>(
          reinterpret_cast<uintptr_t>(header) + desc->offset), desc->size};
      module->executable = hip_impl::load_executable(image, module->executable, agent);

      if (module->executable.handle) {
        modules->at(deviceId) = module;
      }
    }
  }

  return modules;
}

std::map<const void*, std::vector<hipFunction_t>> g_functions;

extern "C" void __hipRegisterFunction(
  std::vector<hipModule_t>* modules,
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
  std::vector<hipFunction_t> functions{g_deviceCnt};

  for (int deviceId = 0; deviceId < g_deviceCnt; ++deviceId) {
    hipFunction_t function;
    if (hipSuccess == hipModuleGetFunction(&function, modules->at(deviceId), deviceName)) {
      functions[deviceId] = function;
    }
  }

  g_functions.insert(std::make_pair(hostFunction, std::move(functions)));
}

extern "C" void __hipRegisterVar(
  std::vector<hipModule_t>* modules,
  char*       hostVar,
  char*       deviceVar,
  const char* deviceName,
  int         ext,
  int         size,
  int         constant,
  int         global)
{
}

extern "C" void __hipUnregisterFatBinary(std::vector<hipModule_t>* modules)
{
  std::for_each(modules->begin(), modules->end(), [](hipModule_t module){ delete module; });
  delete modules;
}

hipError_t hipConfigureCall(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem,
  hipStream_t stream)
{
  auto ctx = ihipGetTlsDefaultCtx();
  LockedAccessor_CtxCrit_t crit(ctx->criticalData());

  crit->_execStack.push(ihipExec_t{gridDim, blockDim, sharedMem, stream});
  return hipSuccess;
}

hipError_t hipSetupArgument(
  const void *arg,
  size_t size,
  size_t offset)
{
  auto ctx = ihipGetTlsDefaultCtx();
  LockedAccessor_CtxCrit_t crit(ctx->criticalData());
  auto& arguments = crit->_execStack.top()._arguments;

  if (arguments.size() < offset + size) {
    arguments.resize(offset + size);
  }

  ::memcpy(&arguments[offset], arg, size);
  return hipSuccess;
}

hipError_t hipLaunchByPtr(const void *hostFunction)
{
  ihipExec_t exec;
  {
    auto ctx = ihipGetTlsDefaultCtx();
    LockedAccessor_CtxCrit_t crit(ctx->criticalData());
    exec = std::move(crit->_execStack.top());
    crit->_execStack.pop();
  }

  int deviceId;
  if (exec._hStream) {
    deviceId = exec._hStream->getDevice()->_deviceId;
  }
  else if (ihipGetTlsDefaultCtx() && ihipGetTlsDefaultCtx()->getDevice()) {
    deviceId = ihipGetTlsDefaultCtx()->getDevice()->_deviceId;
  }
  else {
    deviceId = 0;
  }

  decltype(g_functions)::iterator it;
  if ((it = g_functions.find(hostFunction)) == g_functions.end())
    return hipErrorUnknown;

  size_t size = exec._arguments.size();
  void *extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &exec._arguments[0],
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
      HIP_LAUNCH_PARAM_END
    };

  return hipModuleLaunchKernel(it->second[deviceId],
    exec._gridDim.x, exec._gridDim.y, exec._gridDim.z,
    exec._blockDim.x, exec._blockDim.y, exec._blockDim.z,
    exec._sharedMem, exec._hStream, nullptr, extra);
}


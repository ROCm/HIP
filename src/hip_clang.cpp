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
#include <fstream>

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "hip_fatbin.h"
#include "trace_helper.h"
#include "program_state.inl"

#ifdef __GNUC__
#pragma GCC visibility push (default)
#endif

extern "C" std::vector<hipModule_t>*
__hipRegisterFatBinary(const void* data)
{
  hip_impl::hip_init();

  tprintf(DB_FB, "Enter __hipRegisterFatBinary(%p)\n", data);
  const __CudaFatBinaryWrapper* fbwrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(data);
  if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
    return nullptr;
  }

  const __ClangOffloadBundleHeader* header = fbwrapper->binary;
  std::string magic(reinterpret_cast<const char*>(header), sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC)) {
    return nullptr;
  }

  auto modules = new std::vector<hipModule_t>(g_deviceCnt);
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
    tprintf(DB_FB, "Found bundle for %s\n", target.c_str());

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
      if (HIP_DUMP_CODE_OBJECT)
        __hipDumpCodeObject(image);
      module->executable = hip_impl::get_program_state().load_executable_no_copy(
        reinterpret_cast<const char*>(header) + desc->offset, desc->size,
        module->executable, agent);

      if (module->executable.handle) {
         hip_impl::program_state_impl::read_kernarg_metadata(image, module->kernargs);
         modules->at(deviceId) = module;

         tprintf(DB_FB, "Loaded code object for %s, args size=%ld\n", name, module->kernargs.size());
      } else {
        fprintf(stderr, "Failed to load code object for %s\n", name);
        abort();
      }
    }
  }

  for (int deviceId = 0; deviceId < g_deviceCnt; ++deviceId) {
    hsa_agent_t agent = g_allAgents[deviceId + 1];

    char name[64] = {};
    hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
    if (!(*modules)[deviceId]) {
      fprintf(stderr, "No device code bundle for %s\n", name);
      abort();
    }
  }

  tprintf(DB_FB, "__hipRegisterFatBinary succeeds and returns %p\n", modules);
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
  HIP_INIT_API(NONE, modules, hostFunction, deviceFunction, deviceName);
  std::vector<hipFunction_t> functions(g_deviceCnt);

  assert(modules && modules->size() >= g_deviceCnt);
  for (int deviceId = 0; deviceId < g_deviceCnt; ++deviceId) {
    hipFunction_t function;
    hsa_agent_t agent = g_allAgents[deviceId + 1];
    if ((hipSuccess == hipModuleGetFunctionEx(&function, modules->at(deviceId), deviceName, &agent) ||
        // With code-object-v3, we need to match the kernel descriptor symbol name
        (hipSuccess == hipModuleGetFunctionEx(
                           &function, modules->at(deviceId),
                           (std::string(deviceName) + std::string(".kd")).c_str(),
                           &agent
                       ))) && function != nullptr) {
      functions[deviceId] = function;
    }
    else {
      tprintf(DB_FB, "__hipRegisterFunction cannot find kernel %s for"
          " device %d\n", deviceName, deviceId);
    }
  }

  g_functions.insert(std::make_pair(hostFunction, std::move(functions)));
}

static inline const char* hsa_strerror(hsa_status_t status) {
  const char* str = nullptr;
  if (hsa_status_string(status, &str) == HSA_STATUS_SUCCESS) {
    return str;
  }
  return "Unknown error";
}

struct RegisteredVar {
public:
  RegisteredVar(): size_(0), devicePtr_(nullptr) {}
  ~RegisteredVar() {}

  static inline const char* hsa_strerror(hsa_status_t status) {
  const char* str = nullptr;
  if (hsa_status_string(status, &str) == HSA_STATUS_SUCCESS) {
    return str;
  }
  return "Unknown error";
}

hipDeviceptr_t getdeviceptr() const { return devicePtr_; };
  size_t getvarsize() const { return size_; };

  size_t size_;               // Size of the variable
  hipDeviceptr_t devicePtr_;  //Device Memory Address of the variable.
};

struct DeviceVar {
  void* shadowVptr;
  std::string hostVar;
  size_t size;
  std::vector<hipModule_t>* modules;
  std::vector<RegisteredVar> rvars;
  bool dyn_undef;
};

std::unordered_multimap<std::string, DeviceVar > g_vars;

//The logic follows PlatformState::getGlobalVar in ROCclr RT
static DeviceVar* findVar(std::string hostVar, int deviceId, hipModule_t hmod) {
  DeviceVar* dvar = nullptr;
  if (hmod != nullptr) {
    // If module is provided, then get the var only from that module
    auto var_range = g_vars.equal_range(hostVar);
    for (auto it = var_range.first; it != var_range.second; ++it) {
      if ((*it->second.modules)[deviceId] == hmod) {
        dvar = &(it->second);
        break;
      }
    }
  } else {
    // If var count is < 2, return the var
    if (g_vars.count(hostVar) < 2) {
      auto it = g_vars.find(hostVar);
      dvar = ((it == g_vars.end()) ? nullptr : &(it->second));
    } else {
      // If var count is > 2, return the original var,
      // if original var count != 1, return g_vars.end()/Invalid
      size_t orig_global_count = 0;
      auto var_range = g_vars.equal_range(hostVar);
      for (auto it = var_range.first; it != var_range.second; ++it) {
        // when dyn_undef is set, it is a shadow var
        if (it->second.dyn_undef == false) {
          ++orig_global_count;
          dvar = &(it->second);
        }
      }
      dvar = ((orig_global_count == 1) ? dvar : nullptr);
    }
  }
  return dvar;
}

hipError_t ihipGetGlobalVar(hipDeviceptr_t* dev_ptr, size_t* size_ptr,
                             const char* hostVar, hipModule_t hmod) {
  GET_TLS();
  auto ctx = ihipGetTlsDefaultCtx();

  if (!ctx) return hipErrorInvalidValue;

  auto device = ctx->getDevice();

  if (!device) return hipErrorInvalidValue;

  ihipDevice_t* currentDevice = ihipGetDevice(device->_deviceId);

  if (!currentDevice) return hipErrorInvalidValue;

  int deviceId = device->_deviceId;

  DeviceVar* dvar = findVar(std::string(hostVar), deviceId, hmod);
  if (dvar == nullptr) return hipErrorInvalidValue;

  if (dvar->rvars[deviceId].getdeviceptr() == nullptr) return hipErrorInvalidValue;

  *size_ptr = dvar->rvars[deviceId].getvarsize();
  *dev_ptr = dvar->rvars[deviceId].getdeviceptr();
  return hipSuccess;
}

static bool createGlobalVarObj(const hsa_executable_t& hsaExecutable, const hsa_agent_t& hasAgent,
                               const char* global_name, void** device_pptr, size_t* bytes) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  hsa_symbol_kind_t sym_type;
  hsa_executable_symbol_t global_symbol;
  std::string buildLog;

  /* Find HSA Symbol by name */
  status = hsa_executable_get_symbol_by_name(hsaExecutable, global_name, &hasAgent,
                                             &global_symbol);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog += "Error: Failed to find the Symbol by Name: ";
    buildLog += hsa_strerror(status);
    tprintf(DB_FB, "createGlobalVarObj: %s\n", buildLog.c_str());
    return false;
  }

  /* Find HSA Symbol Type */
  status = hsa_executable_symbol_get_info(global_symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
                                          &sym_type);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog += "Error: Failed to find the Symbol Type : ";
    buildLog += hsa_strerror(status);
    tprintf(DB_FB, "createGlobalVarObj: %s\n", buildLog.c_str());
    return false;
  }

  /* Make sure symbol type is VARIABLE */
  if (sym_type != HSA_SYMBOL_KIND_VARIABLE) {
    buildLog += "Error: Symbol is not of type VARIABLE : ";
    buildLog += hsa_strerror(status);
    tprintf(DB_FB, "createGlobalVarObj: %s\n", buildLog.c_str());
    return false;
  }

  /* Retrieve the size of the variable */
  status = hsa_executable_symbol_get_info(global_symbol, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, bytes);

  if (status != HSA_STATUS_SUCCESS) {
    buildLog += "Error: Failed to retrieve the Symbol Size : ";
    buildLog += hsa_strerror(status);
    tprintf(DB_FB, "createGlobalVarObj: %s\n", buildLog.c_str());
    return false;
  }

  /* Find HSA Symbol Address */
  status = hsa_executable_symbol_get_info(global_symbol,
                                          HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, device_pptr);
  if (status != HSA_STATUS_SUCCESS) {
    buildLog += "Error: Failed to find the Symbol Address : ";
    buildLog += hsa_strerror(status);
    tprintf(DB_FB, "createGlobalVarObj: %s\n", buildLog.c_str());
    return false;
  } else {
    tprintf(DB_FB, "createGlobalVarObj: var %s : device=%p, size=%zu\n", global_name, *device_pptr, *bytes);
  }

  return true;
}

// Registers a device-side global variable.
// For each global variable in device code, there is a corresponding shadow
// global variable in host code. The shadow host variable is used to keep
// track of the value of the device side global variable between kernel
// executions.
// The basic logic is taken from ROCclr RT, but there is much difference.
extern "C" void __hipRegisterVar(
  std::vector<hipModule_t>* modules,   // The device modules containing code object
  char*       var,       // The shadow variable in host code
  char*       hostVar,   // Variable name in host code
  const char* deviceVar, // Variable name in device code
  int         ext,       // Whether this variable is external
  int         size,      // Size of the variable
  int         constant,  // Whether this variable is constant
  int         global)    // Unknown, always 0
{
    HIP_INIT_API(__hipRegisterVar, modules, var, hostVar, deviceVar, ext, size, constant, global);

    DeviceVar dvar{var, std::string{ hostVar }, static_cast<size_t>(size), modules,
           std::vector<RegisteredVar>{ g_deviceCnt }, false };

    for (int deviceId = 0; deviceId < g_deviceCnt; deviceId++) {
        auto device = ihipGetDevice(deviceId);
        if(!device) {
           continue;
        }
        hsa_executable_t& executable = (*modules)[deviceId]->executable;
        hsa_agent_t& agent = g_allAgents[deviceId + 1];
        size_t bytes = 0;
        hipDeviceptr_t devicePtr = nullptr;

        bool success = createGlobalVarObj(executable, agent, hostVar, &devicePtr, &bytes);
        if(!success) {
           return;
        }
        dvar.rvars[deviceId].devicePtr_ = devicePtr;
        dvar.rvars[deviceId].size_ = bytes;

        hc::AmPointerInfo ptrInfo(nullptr, devicePtr, devicePtr, bytes, device->_acc, true, false);
        hc::am_memtracker_add(devicePtr, ptrInfo);

    #if USE_APP_PTR_FOR_CTX
        hc::am_memtracker_update(devicePtr, device->_deviceId, 0u, ihipGetTlsDefaultCtx());
    #else
        hc::am_memtracker_update(devicePtr, device->_deviceId, 0u);
    #endif
    }
    g_vars.insert(std::make_pair(std::string(hostVar), dvar));
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
  GET_TLS();
  auto ctx = ihipGetTlsDefaultCtx();
  LockedAccessor_CtxCrit_t crit(ctx->criticalData());

  crit->_execStack.push(ihipExec_t{gridDim, blockDim, sharedMem, stream});
  return hipSuccess;
}


extern "C" hipError_t __hipPushCallConfiguration(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem,
  hipStream_t stream)
{
  GET_TLS();
  auto ctx = ihipGetTlsDefaultCtx();
  LockedAccessor_CtxCrit_t crit(ctx->criticalData());

  crit->_execStack.push(ihipExec_t{gridDim, blockDim, sharedMem, stream});
  return hipSuccess;
}

extern "C" hipError_t __hipPopCallConfiguration(
  dim3 *gridDim,
  dim3 *blockDim,
  size_t *sharedMem,
  hipStream_t *stream)
{
  GET_TLS();
  auto ctx = ihipGetTlsDefaultCtx();
  LockedAccessor_CtxCrit_t crit(ctx->criticalData());

  ihipExec_t exec;
  exec = std::move(crit->_execStack.top());
  crit->_execStack.pop();

  *gridDim = exec._gridDim;
  *blockDim = exec._blockDim;
  *sharedMem = exec._sharedMem;
  *stream = exec._hStream;

  return hipSuccess;
}

int getCurrentDeviceId()
{
  GET_TLS();

  int deviceId = 0;
  auto ctx = ihipGetTlsDefaultCtx();

  if(!ctx) return deviceId;

  LockedAccessor_CtxCrit_t crit(ctx->criticalData());

  if(crit->_execStack.size() != 0)
  {
    auto &exec = crit->_execStack.top();

    if (exec._hStream) {
      deviceId = exec._hStream->getDevice()->_deviceId;
    } else if (ctx->getDevice()) {
      deviceId = ctx->getDevice()->_deviceId;
    }
  } else if (ctx->getDevice()) {
    deviceId = ctx->getDevice()->_deviceId;
  }
  return deviceId;
}

hipFunction_t ihipGetDeviceFunction(const void *hostFunction)
{
  int deviceId = getCurrentDeviceId();
  auto it = g_functions.find(hostFunction);
  if (it == g_functions.end() || !it->second[deviceId]) {
    return nullptr;
  }
  return it->second[deviceId];
}

hipError_t hipSetupArgument(
  const void *arg,
  size_t size,
  size_t offset)
{
  HIP_INIT_API(hipSetupArgument, arg, size, offset);
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
  HIP_INIT_API(hipLaunchByPtr, hostFunction);
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

  hipError_t e = hipSuccess;
  decltype(g_functions)::iterator it;
  if ((it = g_functions.find(hostFunction)) == g_functions.end() ||
      !it->second[deviceId]) {
    e = hipErrorUnknown;
    fprintf(stderr, "hipLaunchByPtr cannot find kernel with stub address %p"
        " for device %d!\n", hostFunction, deviceId);
    abort();
  } else {
    size_t size = exec._arguments.size();
    void *extra[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &exec._arguments[0],
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
        HIP_LAUNCH_PARAM_END
      };

    e = hipModuleLaunchKernel(it->second[deviceId],
      exec._gridDim.x, exec._gridDim.y, exec._gridDim.z,
      exec._blockDim.x, exec._blockDim.y, exec._blockDim.z,
      exec._sharedMem, exec._hStream, nullptr, extra);
  }

  return ihipLogStatus(e);
}
#ifdef __GNUC__
#pragma GCC visibility pop
#endif

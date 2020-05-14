/* Copyright (c) 2015-present Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <hip/hip_runtime.h>
#include <hip/hcc_detail/texture_types.h>
#include "hip_internal.hpp"
#include "platform/program.hpp"
#include "platform/runtime.hpp"

#include <unordered_map>
#include "elfio.hpp"

constexpr unsigned __hipFatMAGIC2 = 0x48495046; // "HIPF"

thread_local std::stack<ihipExec_t> execStack_;
PlatformState* PlatformState::platform_; // Initiaized as nullptr by default

struct __CudaFatBinaryWrapper {
  unsigned int magic;
  unsigned int version;
  void*        binary;
  void*        dummy1;
};

#define CLANG_OFFLOAD_BUNDLER_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"
#define HIP_AMDGCN_AMDHSA_TRIPLE "hip-amdgcn-amd-amdhsa"
#define HCC_AMDGCN_AMDHSA_TRIPLE "hcc-amdgcn-amd-amdhsa-"

struct __ClangOffloadBundleDesc {
  uint64_t offset;
  uint64_t size;
  uint64_t tripleSize;
  const char triple[1];
};

struct __ClangOffloadBundleHeader {
  const char magic[sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC_STR) - 1];
  uint64_t numBundles;
  __ClangOffloadBundleDesc desc[1];
};

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
    hipModule_t hmod, const char* name);

hipError_t ihipCreateGlobalVarObj(const char* name, hipModule_t hmod, amd::Memory** amd_mem_obj,
                                  hipDeviceptr_t* dptr, size_t* bytes);

static bool isCompatibleCodeObject(const std::string& codeobj_target_id,
                                   const char* device_name) {
  // Workaround for device name mismatch.
  // Device name may contain feature strings delimited by '+', e.g.
  // gfx900+xnack. Currently HIP-Clang does not include feature strings
  // in code object target id in fat binary. Therefore drop the feature
  // strings from device name before comparing it with code object target id.
  std::string short_name(device_name);
  auto feature_loc = short_name.find('+');
  if (feature_loc != std::string::npos) {
    short_name.erase(feature_loc);
  }
  return codeobj_target_id == short_name;
}

// Extracts code objects from fat binary in data for device names given in devices.
// Returns true if code objects are extracted successfully.
hipError_t __hipExtractCodeObjectFromFatBinary(const void* data,
                                         const std::vector<const char*>& devices,
                                         std::vector<std::pair<const void*, size_t>>& code_objs)
{
  std::string magic((const char*)data, sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC_STR) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC_STR)) {
    return hipErrorInvalidKernelFile;
  }

  code_objs.resize(devices.size());
  const auto obheader = reinterpret_cast<const __ClangOffloadBundleHeader*>(data);
  const auto* desc = &obheader->desc[0];
  unsigned num_code_objs = 0;
  for (uint64_t i = 0; i < obheader->numBundles; ++i,
       desc = reinterpret_cast<const __ClangOffloadBundleDesc*>(
           reinterpret_cast<uintptr_t>(&desc->triple[0]) + desc->tripleSize)) {

    std::size_t offset = 0;
    if (!std::strncmp(desc->triple, HIP_AMDGCN_AMDHSA_TRIPLE,
        sizeof(HIP_AMDGCN_AMDHSA_TRIPLE) - 1)) {
      offset = sizeof(HIP_AMDGCN_AMDHSA_TRIPLE); //For code objects created by CLang
    } else if (!std::strncmp(desc->triple, HCC_AMDGCN_AMDHSA_TRIPLE,
               sizeof(HCC_AMDGCN_AMDHSA_TRIPLE) - 1)) {
      offset = sizeof(HCC_AMDGCN_AMDHSA_TRIPLE); //For code objects created by Hcc
    } else {
      continue;
    }
    std::string target(desc->triple + offset, desc->tripleSize - offset);

    const void *image = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(obheader) + desc->offset);
    size_t size = desc->size;

    for (size_t dev = 0; dev < devices.size(); ++dev) {
      const char* name = devices[dev];

      if (!isCompatibleCodeObject(target, name)) {
          continue;
      }
      code_objs[dev] = std::make_pair(image, size);
      num_code_objs++;
    }
  }
  if (num_code_objs == devices.size()) {
    return hipSuccess;
  } else {
    DevLogError("hipErrorNoBinaryForGpu: Coudn't find binary for current devices!");
    return hipErrorNoBinaryForGpu;
  }
}

extern "C" std::vector<std::pair<hipModule_t, bool>>* __hipRegisterFatBinary(const void* data)
{
  const __CudaFatBinaryWrapper* fbwrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(data);
  if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
    DevLogPrintfError("Cannot Register fat binary. FatMagic: %u version: %u ",
                      fbwrapper->magic, fbwrapper->version);
    return nullptr;
  }

  return PlatformState::instance().addFatBinary(fbwrapper->binary);
}

void PlatformState::digestFatBinary(const void* data, std::vector<std::pair<hipModule_t, bool>>& programs)
{
  if (programs.size() > 0) {
    return;
  }

  std::vector<std::pair<const void*, size_t>> code_objs;
  std::vector<const char*> devices;
  for (size_t dev = 0; dev < g_devices.size(); ++dev) {
    devices.push_back(g_devices[dev]->devices()[0]->info().name_);
  }

  if (hipSuccess != __hipExtractCodeObjectFromFatBinary((char*)data, devices, code_objs)) {
    return;
  }

  programs.resize(g_devices.size());

  for (size_t dev = 0; dev < g_devices.size(); ++dev) {
    amd::Context* ctx = g_devices[dev]->asContext();
    amd::Program* program = new amd::Program(*ctx);
    if (program == nullptr) {
      return;
    }
    if (CL_SUCCESS == program->addDeviceProgram(
            *ctx->devices()[0], code_objs[dev].first, code_objs[dev].second, false)) {
      programs.at(dev) = std::make_pair(reinterpret_cast<hipModule_t>(as_cl(program)) , false);
    }
  }
}

void PlatformState::init()
{
  amd::ScopedLock lock(lock_);

  if(initialized_ || g_devices.empty()) {
    return;
  }
  initialized_ = true;

  for (auto& it : modules_) {
    digestFatBinary(it.first, it.second);
  }
  for (auto& it : functions_) {
    it.second.functions.resize(g_devices.size());
  }
  for (auto& it : vars_) {
    it.second.rvars.resize(g_devices.size());
  }
}

bool PlatformState::unregisterFunc(hipModule_t hmod) {
  amd::ScopedLock lock(lock_);
  auto mod_it = module_map_.find(hmod);
  if (mod_it != module_map_.cend()) {
    PlatformState::Module* mod_ptr = mod_it->second;
    if(mod_ptr != nullptr) {
      for (auto func_it = mod_ptr->functions_.begin(); func_it != mod_ptr->functions_.end(); ++func_it) {
        PlatformState::DeviceFunction &devFunc = func_it->second;
        for (size_t dev = 0; dev < g_devices.size(); ++dev) {
          if (devFunc.functions[dev] != 0) {
            hip::Function* f = reinterpret_cast<hip::Function*>(devFunc.functions[dev]);
            delete f;
          }
        }
        delete devFunc.modules;
      }
      delete mod_ptr;
    }
    module_map_.erase(mod_it);
  }
  return true;
}

std::vector< std::pair<hipModule_t, bool> >* PlatformState::unregisterVar(hipModule_t hmod) {
  amd::ScopedLock lock(lock_);
  std::vector< std::pair<hipModule_t, bool> >* rmodules = nullptr;
  auto it = vars_.begin();
  while (it != vars_.end()) {
    DeviceVar& dvar = it->second;
    if ((*dvar.modules)[0].first == hmod) {
      rmodules = dvar.modules;
      if (dvar.shadowAllocated) {
        texture<float, hipTextureType1D, hipReadModeElementType>* tex_hptr
          = reinterpret_cast<texture<float, hipTextureType1D, hipReadModeElementType> *>(dvar.shadowVptr);
        delete tex_hptr;
      }
      for (size_t dev = 0; dev < g_devices.size(); ++dev) {
        if (dvar.rvars[dev].getdeviceptr()) {
          amd::MemObjMap::RemoveMemObj(dvar.rvars[dev].getdeviceptr());
        }
      }
      vars_.erase(it++);
    } else {
      ++it;
    }
  }
  return rmodules;
}

PlatformState::DeviceVar* PlatformState::findVar(std::string hostVar, int deviceId, hipModule_t hmod) {
  DeviceVar* dvar = nullptr;
  if (hmod != nullptr) {
    // If module is provided, then get the var only from that module
    auto var_range = vars_.equal_range(hostVar);
    for (auto it = var_range.first; it != var_range.second; ++it) {
      if ((*it->second.modules)[deviceId].first == hmod) {
        dvar = &(it->second);
        break;
      }
    }
  } else {
    // If var count is < 2, return the var
    if (vars_.count(hostVar) < 2) {
      auto it = vars_.find(hostVar);
      dvar = ((it == vars_.end()) ? nullptr : &(it->second));
    } else {
      // If var count is > 2, return the original var,
      // if original var count != 1, return vars_.end()/Invalid
      size_t orig_global_count = 0;
      auto var_range = vars_.equal_range(hostVar);
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

bool PlatformState::findSymbol(const void *hostVar,
                               hipModule_t &hmod, std::string &symbolName) {
  auto it = symbols_.find(hostVar);
  if (it != symbols_.end()) {
    hmod = it->second.first;
    symbolName = it->second.second;
    return true;
  }
  DevLogPrintfError("Could not find the Symbol: %s \n", symbolName.c_str());
  return false;
}

void PlatformState::registerVarSym(const void* hostVar, hipModule_t hmod, const char* symbolName) {
  amd::ScopedLock lock(lock_);
  symbols_.insert(std::make_pair(hostVar, std::make_pair(hmod, std::string(symbolName))));
}

void PlatformState::registerVar(const char* hostvar,
                                const DeviceVar& rvar) {
  amd::ScopedLock lock(lock_);
  vars_.insert(std::make_pair(std::string(hostvar), rvar));
}

void PlatformState::registerFunction(const void* hostFunction,
                                     const DeviceFunction& func) {
  amd::ScopedLock lock(lock_);
  functions_.insert(std::make_pair(hostFunction, func));
}

bool ihipGetFuncAttributes(const char* func_name, amd::Program* program, hipFuncAttributes* func_attr) {
  device::Program* dev_program
    = program->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  const auto it = dev_program->kernels().find(std::string(func_name));
  if (it == dev_program->kernels().cend()) {
    DevLogPrintfError("Could not find the function %s \n", func_name);
    return false;
  }

  const device::Kernel::WorkGroupInfo* wginfo = it->second->workGroupInfo();
  func_attr->localSizeBytes = wginfo->localMemSize_;
  func_attr->sharedSizeBytes = wginfo->size_;
  func_attr->maxThreadsPerBlock = wginfo->wavefrontSize_;
  func_attr->numRegs = wginfo->usedVGPRs_;

  return true;
}

bool PlatformState::getShadowVarInfo(std::string var_name, hipModule_t hmod,
                                     void** var_addr, size_t* var_size) {
  DeviceVar* dvar = findVar(var_name, ihipGetDevice(), hmod);
  if (dvar != nullptr) {
    *var_addr = dvar->shadowVptr;
    *var_size = dvar->size;
    return true;
  } else {
    DevLogPrintfError("Cannot find Var name: %s in module: 0x%x \n", var_name.c_str(), hmod);
    return false;
  }
}

bool CL_CALLBACK getSvarInfo(cl_program program, std::string var_name, void** var_addr,
                             size_t* var_size) {
  return PlatformState::instance().getShadowVarInfo(var_name, reinterpret_cast<hipModule_t>(program),
                                                    var_addr, var_size);
}

bool PlatformState::registerModFuncs(std::vector<std::string>& func_names, hipModule_t* module) {
  amd::ScopedLock lock(lock_);
  PlatformState::Module* mod_ptr = new PlatformState::Module(*module);

  for (auto it = func_names.begin(); it != func_names.end(); ++it) {
    auto modules = new std::vector<std::pair<hipModule_t, bool> >(g_devices.size());
    for (size_t dev = 0; dev < g_devices.size(); ++dev) {
      modules->at(dev) = std::make_pair(*module, true);
    }

    PlatformState::DeviceFunction dfunc{*it, modules,
      std::vector<hipFunction_t>(g_devices.size(), 0)};
    mod_ptr->functions_.insert(std::make_pair(*it, dfunc));
  }

  module_map_.insert(std::make_pair(*module, mod_ptr));
  return true;
}

bool PlatformState::findModFunc(hipFunction_t* hfunc, hipModule_t hmod, const char* name) {
  amd::ScopedLock lock(lock_);

  auto mod_it = module_map_.find(hmod);
  if (mod_it != module_map_.cend()) {
    assert(mod_it->second != nullptr);
    auto func_it = mod_it->second->functions_.find(name);
    if (func_it != mod_it->second->functions_.cend()) {
      PlatformState::DeviceFunction& devFunc = func_it->second;
      if (devFunc.functions[ihipGetDevice()] == 0) {
        if(!createFunc(&devFunc.functions[ihipGetDevice()], hmod, name)) {
          DevLogPrintfError("Could not create a function: %s at module: 0x%x \n", name, hmod);
          return false;
        }
      }
      *hfunc = devFunc.functions[ihipGetDevice()];
      return true;
    }
  }
  DevLogPrintfError("Cannot find module: 0x%x in PlatformState Module Map \n", hmod);
  return false;
}

bool PlatformState::createFunc(hipFunction_t* hfunc, hipModule_t hmod, const char* name) {
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));

  const amd::Symbol* symbol = program->findSymbol(name);
  if (!symbol) {
    DevLogPrintfError("Cannot find Symbol with name: %s \n", name);
    return false;
  }

  amd::Kernel* kernel = new amd::Kernel(*program, *symbol, name);
  if (!kernel) {
    DevLogPrintfError("Could not create a new kernel with name: %s \n", name);
    return false;
  }

  hip::Function* f = new hip::Function(kernel);
  if (!f) {
    DevLogPrintfError("Could not create a new function with name: %s \n", name);
    return false;
  }

  *hfunc = f->asHipFunction();

  return true;
}


hipFunction_t PlatformState::getFunc(const void* hostFunction, int deviceId) {
  amd::ScopedLock lock(lock_);
  const auto it = functions_.find(hostFunction);
  if (it != functions_.cend()) {
    PlatformState::DeviceFunction& devFunc = it->second;
    if (devFunc.functions[deviceId] == 0) {
      hipModule_t module = (*devFunc.modules)[deviceId].first;
      if (!(*devFunc.modules)[deviceId].second) {
        amd::Program* program = as_amd(reinterpret_cast<cl_program>(module));
        program->setVarInfoCallBack(&getSvarInfo);
        if (CL_SUCCESS != program->build(g_devices[deviceId]->devices(), nullptr, nullptr, nullptr)) {
          DevLogPrintfError("Build error for module: 0x%x at device: %u \n", module, deviceId);
          return nullptr;
        }
        (*devFunc.modules)[deviceId].second = true;
      }
      hipFunction_t function = nullptr;
      if (createFunc(&function, module, devFunc.deviceName.c_str()) &&
          function != nullptr) {
        devFunc.functions[deviceId] = function;
      }
      else {
   //     tprintf(DB_FB, "__hipRegisterFunction cannot find kernel %s for"
   //         " device %d\n", deviceName, deviceId);
      }
    }
    return devFunc.functions[deviceId];
  }
  DevLogPrintfError("Cannot find function: 0x%x in PlatformState \n", hostFunction);
  return nullptr;
}

bool PlatformState::getFuncAttr(const void* hostFunction,
                                hipFuncAttributes* func_attr) {
  if (func_attr == nullptr) {
    return false;
  }

  const auto it = functions_.find(hostFunction);
  if (it == functions_.cend()) {
    DevLogPrintfError("Cannot find hostFunction 0x%x \n", hostFunction);
    return false;
  }

  PlatformState::DeviceFunction& devFunc = it->second;
  int deviceId = ihipGetDevice();

  /* If module has not been initialized yet, build the kernel now*/
  if (!(*devFunc.modules)[deviceId].second) {
    if (nullptr == PlatformState::instance().getFunc(hostFunction, deviceId)) {
      DevLogPrintfError("Cannot get hostFunction: 0x%x for deviceId:%d \n", hostFunction, deviceId);
      return false;
    }
  }

  amd::Program* program = as_amd(reinterpret_cast<cl_program>((*devFunc.modules)[deviceId].first));
  if (!ihipGetFuncAttributes(devFunc.deviceName.c_str(), program, func_attr)) {
    DevLogPrintfError("Cannot get Func attributes for function: %s \n",
                      devFunc.deviceName.c_str());
    return false;
  }
  return true;
}

bool PlatformState::getTexRef(const char* hostVar, hipModule_t hmod, textureReference** texRef) {
  amd::ScopedLock lock(lock_);
  DeviceVar* dvar = findVar(std::string(hostVar), ihipGetDevice(), hmod);
  if (dvar == nullptr) {
    DevLogPrintfError("Cannot find var:%s for creating texture reference at module: 0x%x \n",
                      hostVar, hmod);
    return false;
  }

  switch (dvar->kind) {
  case PlatformState::DVK_Variable:
    // TODO: Need to define a target-specific symbol info to indicate the device
    // variable kind, i.e. regular variable, texture or surface.
    // Before that, have to assume the specified variable is a texture or
    // surface reference variable.
    dvar->kind = DVK_Texture;
    // FALL THROUGH
  case PlatformState::DVK_Texture:
    break;
  default:
    // If it's already used as non-texture variable, bail out.
    return false;
  }

  if (!dvar->shadowVptr) {
    dvar->shadowVptr = new texture<char>{};
    dvar->shadowAllocated = true;
  }
  *texRef = reinterpret_cast<textureReference *>(dvar->shadowVptr);
  registerVarSym(dvar->shadowVptr, hmod, hostVar);

  return true;
}

bool PlatformState::getGlobalVar(const char* hostVar, int deviceId, hipModule_t hmod,
                                 hipDeviceptr_t* dev_ptr, size_t* size_ptr) {
  amd::ScopedLock lock(lock_);
  DeviceVar* dvar = findVar(std::string(hostVar), deviceId, hmod);
  if (dvar != nullptr) {
    if (dvar->rvars[deviceId].getdeviceptr() == nullptr) {
      size_t sym_size = 0;
      hipDeviceptr_t device_ptr = nullptr;
      amd::Memory* amd_mem_obj = nullptr;

      if (!(*dvar->modules)[deviceId].second) {
        amd::Program* program = as_amd(reinterpret_cast<cl_program>((*dvar->modules)[deviceId].first));
        program->setVarInfoCallBack(&getSvarInfo);
        if (CL_SUCCESS != program->build(g_devices[deviceId]->devices(), nullptr, nullptr, nullptr)) {
          DevLogPrintfError("Build Failure for module: 0x%x \n", hmod);
          return false;
        }
        (*dvar->modules)[deviceId].second = true;
      }
      if((hipSuccess == ihipCreateGlobalVarObj(dvar->hostVar.c_str(), (*dvar->modules)[deviceId].first,
                                               &amd_mem_obj, &device_ptr, &sym_size))
           && (device_ptr != nullptr)) {
        dvar->rvars[deviceId].size_ = sym_size;
        dvar->rvars[deviceId].devicePtr_ = device_ptr;
        dvar->rvars[deviceId].amd_mem_obj_ = amd_mem_obj;
        amd::MemObjMap::AddMemObj(device_ptr, amd_mem_obj);
      } else {
        LogError("__hipRegisterVar cannot find kernel for device \n");
      }
    }
    *size_ptr = dvar->rvars[deviceId].getvarsize();
    *dev_ptr = dvar->rvars[deviceId].getdeviceptr();
    return true;
  } else {
    DevLogPrintfError("Could not find global var: %s at module:0x%x \n", hostVar, hmod);
    return false;
  }
}

bool PlatformState::getGlobalVarFromSymbol(const void* hostVar, int deviceId,
                                           hipDeviceptr_t* dev_ptr,
                                           size_t* size_ptr) {
  hipModule_t hmod;
  std::string symbolName;
  if (!PlatformState::instance().findSymbol(hostVar, hmod, symbolName)) {
    return false;
  }
  return PlatformState::instance().getGlobalVar(symbolName.c_str(),
                                                ihipGetDevice(), hmod,
                                                dev_ptr, size_ptr);
}

void PlatformState::setupArgument(const void *arg, size_t size, size_t offset) {
  auto& arguments = execStack_.top().arguments_;

  if (arguments.size() < offset + size) {
    arguments.resize(offset + size);
  }

  ::memcpy(&arguments[offset], arg, size);
}

void PlatformState::configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                  hipStream_t stream) {
  execStack_.push(ihipExec_t{gridDim, blockDim, sharedMem, stream});
}

void PlatformState::popExec(ihipExec_t& exec) {
  exec = std::move(execStack_.top());
  execStack_.pop();
}

namespace {
const int HIP_ENABLE_DEFERRED_LOADING{[] () {
  char *var = getenv("HIP_ENABLE_DEFERRED_LOADING");
  return var ? atoi(var) : 1;
}()};
} /* namespace */

extern "C" void __hipRegisterFunction(
  std::vector<std::pair<hipModule_t,bool> >* modules,
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
  PlatformState::DeviceFunction func{ std::string{deviceName}, modules, std::vector<hipFunction_t>{g_devices.size()}};
  PlatformState::instance().registerFunction(hostFunction, func);
  if (!HIP_ENABLE_DEFERRED_LOADING) {
    HIP_INIT();
    for (size_t i = 0; i < g_devices.size(); ++i) {
      PlatformState::instance().getFunc(hostFunction, i);
    }
  }
}

// Registers a device-side global variable.
// For each global variable in device code, there is a corresponding shadow
// global variable in host code. The shadow host variable is used to keep
// track of the value of the device side global variable between kernel
// executions.
extern "C" void __hipRegisterVar(
  std::vector<std::pair<hipModule_t,bool> >* modules,   // The device modules containing code object
  void*       var,       // The shadow variable in host code
  char*       hostVar,   // Variable name in host code
  char*       deviceVar, // Variable name in device code
  int         ext,       // Whether this variable is external
  size_t      size,      // Size of the variable
  int         constant,  // Whether this variable is constant
  int         global)    // Unknown, always 0
{
    PlatformState::DeviceVar dvar{PlatformState::DVK_Variable,
                                  var,
                                  std::string{hostVar},
                                  size,
                                  modules,
                                  std::vector<PlatformState::RegisteredVar>{g_devices.size()},
                                  false,
                                  /*type*/ 0,
                                  /*norm*/ 0};

    PlatformState::instance().registerVar(hostVar, dvar);
    PlatformState::instance().registerVarSym(var, nullptr, deviceVar);
}

extern "C" void __hipRegisterSurface(std::vector<std::pair<hipModule_t, bool>>*
                                         modules,      // The device modules containing code object
                                     void* var,        // The shadow variable in host code
                                     char* hostVar,    // Variable name in host code
                                     char* deviceVar,  // Variable name in device code
                                     int type, int ext) {
    PlatformState::DeviceVar dvar{PlatformState::DVK_Surface,
                                  var,
                                  std::string{hostVar},
                                  sizeof(surfaceReference),  // Copy whole surfaceReference
                                  modules,
                                  std::vector<PlatformState::RegisteredVar>{g_devices.size()},
                                  false,
                                  type,
                                  /*norm*/ 0};
    PlatformState::instance().registerVar(hostVar, dvar);
    PlatformState::instance().registerVarSym(var, nullptr, deviceVar);
}

extern "C" void __hipRegisterTexture(std::vector<std::pair<hipModule_t, bool>>*
                                         modules,      // The device modules containing code object
                                     void* var,        // The shadow variable in host code
                                     char* hostVar,    // Variable name in host code
                                     char* deviceVar,  // Variable name in device code
                                     int type, int norm, int ext) {
    PlatformState::DeviceVar dvar{PlatformState::DVK_Texture,
                                  var,
                                  std::string{hostVar},
                                  sizeof(textureReference),  // Copy whole textureReference so far.
                                  modules,
                                  std::vector<PlatformState::RegisteredVar>{g_devices.size()},
                                  false,
                                  type,
                                  norm};
    PlatformState::instance().registerVar(hostVar, dvar);
    PlatformState::instance().registerVarSym(var, nullptr, deviceVar);
}

extern "C" void __hipUnregisterFatBinary(std::vector< std::pair<hipModule_t, bool> >* modules)
{
  HIP_INIT();

  std::for_each(modules->begin(), modules->end(), [](std::pair<hipModule_t, bool> module){
    if (module.first != nullptr) {
      as_amd(reinterpret_cast<cl_program>(module.first))->release();
    }
  });
  if (modules->size() > 0) {
    PlatformState::instance().unregisterVar((*modules)[0].first);
  }
  PlatformState::instance().removeFatBinary(modules);
}

extern "C" hipError_t hipConfigureCall(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem,
  hipStream_t stream)
{
  HIP_INIT_API(hipConfigureCall, gridDim, blockDim, sharedMem, stream);

  PlatformState::instance().configureCall(gridDim, blockDim, sharedMem, stream);

  HIP_RETURN(hipSuccess);
}

extern "C" hipError_t __hipPushCallConfiguration(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem,
  hipStream_t stream)
{
  HIP_INIT_API(__hipPushCallConfiguration, gridDim, blockDim, sharedMem, stream);

  PlatformState::instance().configureCall(gridDim, blockDim, sharedMem, stream);

  HIP_RETURN(hipSuccess);
}

extern "C" hipError_t __hipPopCallConfiguration(dim3 *gridDim,
                                                dim3 *blockDim,
                                                size_t *sharedMem,
                                                hipStream_t *stream) {
  HIP_INIT_API(__hipPopCallConfiguration, gridDim, blockDim, sharedMem, stream);

  ihipExec_t exec;
  PlatformState::instance().popExec(exec);
  *gridDim = exec.gridDim_;
  *blockDim = exec.blockDim_;
  *sharedMem = exec.sharedMem_;
  *stream = exec.hStream_;

  HIP_RETURN(hipSuccess);
}

extern "C" hipError_t hipSetupArgument(
  const void *arg,
  size_t size,
  size_t offset)
{
  HIP_INIT_API(hipSetupArgument, arg, size, offset);

  PlatformState::instance().setupArgument(arg, size, offset);

  HIP_RETURN(hipSuccess);
}

extern "C" hipError_t hipLaunchByPtr(const void *hostFunction)
{
  HIP_INIT_API(hipLaunchByPtr, hostFunction);

  ihipExec_t exec;
  PlatformState::instance().popExec(exec);

  hip::Stream* stream = reinterpret_cast<hip::Stream*>(exec.hStream_);
  int deviceId = (stream != nullptr)? stream->DeviceId() : ihipGetDevice();
  if (deviceId == -1) {
    DevLogPrintfError("Wrong DeviceId: %d \n", deviceId);
    HIP_RETURN(hipErrorNoDevice);
  }
  hipFunction_t func = PlatformState::instance().getFunc(hostFunction, deviceId);
  if (func == nullptr) {
    DevLogPrintfError("Could not retrieve hostFunction: 0x%x \n", hostFunction);
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  size_t size = exec.arguments_.size();
  void *extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &exec.arguments_[0],
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
      HIP_LAUNCH_PARAM_END
    };

  HIP_RETURN(hipModuleLaunchKernel(func,
    exec.gridDim_.x, exec.gridDim_.y, exec.gridDim_.z,
    exec.blockDim_.x, exec.blockDim_.y, exec.blockDim_.z,
    exec.sharedMem_, exec.hStream_, nullptr, extra));
}

hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
  HIP_INIT_API(hipGetSymbolAddress, devPtr, symbol);

  hipModule_t hmod;
  std::string symbolName;
  if (!PlatformState::instance().findSymbol(symbol, hmod, symbolName)) {
    DevLogPrintfError("Cannot find symbol: %s \n", symbolName.c_str());
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  size_t size = 0;
  if(!PlatformState::instance().getGlobalVar(symbolName.c_str(), ihipGetDevice(), hmod,
                                             devPtr, &size)) {
    DevLogPrintfError("Cannot find global variable device ptr for symbol: %s at device: %d \n",
                      symbolName.c_str(), ihipGetDevice());
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipGetSymbolSize(size_t* sizePtr, const void* symbol) {
  HIP_INIT_API(hipGetSymbolSize, sizePtr, symbol);

  hipModule_t hmod;
  std::string symbolName;
  if (!PlatformState::instance().findSymbol(symbol, hmod, symbolName)) {
    DevLogPrintfError("Cannot find symbol: %s \n", symbolName.c_str());
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  hipDeviceptr_t devPtr = nullptr;
  if (!PlatformState::instance().getGlobalVar(symbolName.c_str(), ihipGetDevice(), hmod,
                                              &devPtr, sizePtr)) {
    DevLogPrintfError("Cannot find global variable device ptr for symbol: %s at device: %d \n",
                      symbolName.c_str(), ihipGetDevice());
    HIP_RETURN(hipErrorInvalidSymbol);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t ihipCreateGlobalVarObj(const char* name, hipModule_t hmod, amd::Memory** amd_mem_obj,
                                  hipDeviceptr_t* dptr, size_t* bytes)
{
  HIP_INIT();

  amd::Program* program = nullptr;
  device::Program* dev_program = nullptr;

  /* Get Device Program pointer*/
  program = as_amd(reinterpret_cast<cl_program>(hmod));
  dev_program = program->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  if (dev_program == nullptr) {
    DevLogPrintfError("Cannot get Device Function for module: 0x%x \n", hmod);
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  /* Find the global Symbols */
  if (!dev_program->createGlobalVarObj(amd_mem_obj, dptr, bytes, name)) {
    DevLogPrintfError("Cannot create Global Var obj for symbol: %s \n", name);
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  HIP_RETURN(hipSuccess);
}


namespace hip_impl {
hipError_t ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, int* numGrids,
    const amd::Device& device, hipFunction_t func, int  blockSize,
    size_t dynamicSMemSize, bool bCalcPotentialBlkSz)
{
  hip::Function* function = hip::Function::asFunction(func);
  const amd::Kernel& kernel = *function->function_;

  const device::Kernel::WorkGroupInfo* wrkGrpInfo = kernel.getDeviceKernel(device)->workGroupInfo();
  if (blockSize == 0) {
    if (bCalcPotentialBlkSz == false){
      return hipErrorInvalidValue;
    }
    else {
      blockSize = device.info().maxWorkGroupSize_; // maxwavefrontperblock
    }
  }

  // Make sure the requested block size is smaller than max supported
  if (blockSize > int(device.info().maxWorkGroupSize_)) {
    numBlocks = 0;
    numGrids = 0;
    return hipSuccess;
  }

  // Find threads accupancy per CU => simd_per_cu * GPR usage
  constexpr size_t MaxWavesPerSimd = 8;  // Limited by SPI 32 per CU, hence 8 per SIMD
  size_t VgprWaves = MaxWavesPerSimd;
  if (wrkGrpInfo->usedVGPRs_ > 0) {
    VgprWaves = wrkGrpInfo->availableVGPRs_ / amd::alignUp(wrkGrpInfo->usedVGPRs_, 4);
  }

  size_t GprWaves = VgprWaves;
  if (wrkGrpInfo->usedSGPRs_ > 0) {
    const size_t maxSGPRs = (device.info().gfxipVersion_ < 800) ? 512 : 800;
    size_t SgprWaves = maxSGPRs / amd::alignUp(wrkGrpInfo->usedSGPRs_, 16);
    GprWaves = std::min(VgprWaves, SgprWaves);
  }

  size_t alu_accupancy = device.info().simdPerCU_ * std::min(MaxWavesPerSimd, GprWaves);
  alu_accupancy *= wrkGrpInfo->wavefrontSize_;
  // Calculate blocks occupancy per CU
  *numBlocks = alu_accupancy / amd::alignUp(blockSize, wrkGrpInfo->wavefrontSize_);

  size_t total_used_lds = wrkGrpInfo->usedLDSSize_ + dynamicSMemSize;
  if (total_used_lds != 0) {
    // Calculate LDS occupancy per CU. lds_per_cu / (static_lsd + dynamic_lds)
    int lds_occupancy = static_cast<int>(device.info().localMemSize_ / total_used_lds);
    *numBlocks = std::min(*numBlocks, lds_occupancy);
  }

  if (bCalcPotentialBlkSz) {
    *numGrids = *numBlocks * device.info().numRTCUs_;
  }

  return hipSuccess;
}
}

extern "C" {
hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             const void* f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit)
{
  HIP_INIT_API(hipOccupancyMaxPotentialBlockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  if ((gridSize == nullptr) || (blockSize == nullptr)) {
    return HIP_RETURN(hipErrorInvalidValue);
  }
  hipFunction_t func = PlatformState::instance().getFunc(f, ihipGetDevice());
  if (func == nullptr) {
    return HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int num_grids = 0;
  int num_blocks = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &num_grids, device, func, 0, dynSharedMemPerBlk,true);
  if (ret == hipSuccess) {
    *blockSize = num_blocks;
    *gridSize = num_grids;
  }
  HIP_RETURN(ret);
}

hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit)
{
  HIP_INIT_API(hipModuleOccupancyMaxPotentialBlockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  if ((gridSize == nullptr) || (blockSize == nullptr)) {
    return HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int num_grids = 0;
  int num_blocks = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &num_grids, device, f, 0, dynSharedMemPerBlk,true);
  if (ret == hipSuccess) {
    *blockSize = num_blocks;
    *gridSize = num_grids;
  }
  HIP_RETURN(ret);
}

hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit, unsigned int flags)
{
  HIP_INIT_API(hipModuleOccupancyMaxPotentialBlockSizeWithFlags, f, dynSharedMemPerBlk, blockSizeLimit, flags);
  if ((gridSize == nullptr) || (blockSize == nullptr)) {
    return HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int num_grids = 0;
  int num_blocks = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &num_grids, device, f, 0, dynSharedMemPerBlk,true);
  if (ret == hipSuccess) {
    *blockSize = num_blocks;
    *gridSize = num_grids;
  }
  HIP_RETURN(ret);
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, 
                                             hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk)
{
  HIP_INIT_API(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, f, blockSize, dynSharedMemPerBlk);
  if (numBlocks == nullptr) {
    return HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, nullptr, device, f, blockSize, dynSharedMemPerBlk, false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                              hipFunction_t f, int blockSize, 
                                                              size_t dynSharedMemPerBlk, unsigned int flags)
{
  HIP_INIT_API(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, f, blockSize, dynSharedMemPerBlk, flags);
  if (numBlocks == nullptr) {
    return HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, nullptr, device, f, blockSize, dynSharedMemPerBlk, false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                        const void* f, int blockSize, size_t dynamicSMemSize)
{
  HIP_INIT_API(hipOccupancyMaxActiveBlocksPerMultiprocessor, f, blockSize, dynamicSMemSize);
  if (numBlocks == nullptr) {
    return HIP_RETURN(hipErrorInvalidValue);
  }

  hipFunction_t func = PlatformState::instance().getFunc(f, ihipGetDevice());
  if (func == nullptr) {
    return HIP_RETURN(hipErrorInvalidValue);
  }

  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, nullptr, device, func, blockSize, dynamicSMemSize, false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                                 const void* f,
                                                                 int  blockSize, size_t dynamicSMemSize, unsigned int flags)
{
  HIP_INIT_API(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, f, blockSize, dynamicSMemSize, flags);
  if (numBlocks == nullptr) {
    return HIP_RETURN(hipErrorInvalidValue);
  }

  hipFunction_t func = PlatformState::instance().getFunc(f, ihipGetDevice());
  if (func == nullptr) {
    return HIP_RETURN(hipErrorInvalidValue);
  }

  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, nullptr, device, func, blockSize, dynamicSMemSize, false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}
}


#if defined(ATI_OS_LINUX)

namespace hip_impl {

struct dl_phdr_info {
  ELFIO::Elf64_Addr        dlpi_addr;
  const char       *dlpi_name;
  const ELFIO::Elf64_Phdr *dlpi_phdr;
  ELFIO::Elf64_Half        dlpi_phnum;
};

extern "C" int dl_iterate_phdr(
  int (*callback) (struct dl_phdr_info *info, size_t size, void *data), void *data
);

struct Symbol {
  std::string name;
  ELFIO::Elf64_Addr value = 0;
  ELFIO::Elf_Xword size = 0;
  ELFIO::Elf_Half sect_idx = 0;
  uint8_t bind = 0;
  uint8_t type = 0;
  uint8_t other = 0;
};

inline Symbol read_symbol(const ELFIO::symbol_section_accessor& section, unsigned int idx) {
  assert(idx < section.get_symbols_num());

  Symbol r;
  section.get_symbol(idx, r.name, r.value, r.size, r.bind, r.type, r.sect_idx, r.other);

  return r;
}

template <typename P>
inline ELFIO::section* find_section_if(ELFIO::elfio& reader, P p) {
    const auto it = find_if(reader.sections.begin(), reader.sections.end(), std::move(p));

    return it != reader.sections.end() ? *it : nullptr;
}

std::vector<std::pair<uintptr_t, std::string>> function_names_for(const ELFIO::elfio& reader,
                                                                  ELFIO::section* symtab) {
  std::vector<std::pair<uintptr_t, std::string>> r;
  ELFIO::symbol_section_accessor symbols{reader, symtab};

  for (auto i = 0u; i != symbols.get_symbols_num(); ++i) {
    auto tmp = read_symbol(symbols, i);

    if (tmp.type == STT_FUNC && tmp.sect_idx != SHN_UNDEF && !tmp.name.empty()) {
      r.emplace_back(tmp.value, tmp.name);
    }
  }

  return r;
}

const std::vector<std::pair<uintptr_t, std::string>>& function_names_for_process() {
  static constexpr const char self[] = "/proc/self/exe";

  static std::vector<std::pair<uintptr_t, std::string>> r;
  static std::once_flag f;

  std::call_once(f, []() {
    ELFIO::elfio reader;

    if (reader.load(self)) {
      const auto it = find_section_if(
          reader, [](const ELFIO::section* x) { return x->get_type() == SHT_SYMTAB; });

      if (it) r = function_names_for(reader, it);
    }
  });

  return r;
}


const std::unordered_map<uintptr_t, std::string>& function_names()
{
  static std::unordered_map<uintptr_t, std::string> r{
    function_names_for_process().cbegin(),
    function_names_for_process().cend()};
  static std::once_flag f;

  std::call_once(f, []() {
    dl_iterate_phdr([](dl_phdr_info* info, size_t, void*) {
      ELFIO::elfio reader;

      if (reader.load(info->dlpi_name)) {
        const auto it = find_section_if(
            reader, [](const ELFIO::section* x) { return x->get_type() == SHT_SYMTAB; });

        if (it) {
          auto n = function_names_for(reader, it);

          for (auto&& f : n) f.first += info->dlpi_addr;

          r.insert(make_move_iterator(n.begin()), make_move_iterator(n.end()));
        }
      }
      return 0;
    },
    nullptr);
  });

  return r;
}

std::vector<char> bundles_for_process() {
  static constexpr const char self[] = "/proc/self/exe";
  static constexpr const char kernel_section[] = ".kernel";
  std::vector<char> r;

  ELFIO::elfio reader;

  if (reader.load(self)) {
    auto it = find_section_if(
        reader, [](const ELFIO::section* x) { return x->get_name() == kernel_section; });

    if (it) r.insert(r.end(), it->get_data(), it->get_data() + it->get_size());
  }

  return r;
}

const std::vector<hipModule_t>& modules() {
    static std::vector<hipModule_t> r;
    static std::once_flag f;

    std::call_once(f, []() {
      static std::vector<std::vector<char>> bundles{bundles_for_process()};

      dl_iterate_phdr(
          [](dl_phdr_info* info, std::size_t, void*) {
        ELFIO::elfio tmp;
        if (tmp.load(info->dlpi_name)) {
          const auto it = find_section_if(
              tmp, [](const ELFIO::section* x) { return x->get_name() == ".kernel"; });

          if (it) bundles.emplace_back(it->get_data(), it->get_data() + it->get_size());
        }
        return 0;
      },
      nullptr);

      for (auto&& bundle : bundles) {
        if (bundle.empty()) {
          continue;
        }
        std::string magic(&bundle[0], sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC_STR) - 1);
        if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC_STR))
          continue;

        const auto obheader = reinterpret_cast<const __ClangOffloadBundleHeader*>(&bundle[0]);
        const auto* desc = &obheader->desc[0];
        for (uint64_t i = 0; i < obheader->numBundles; ++i,
             desc = reinterpret_cast<const __ClangOffloadBundleDesc*>(
                 reinterpret_cast<uintptr_t>(&desc->triple[0]) + desc->tripleSize)) {

          std::string triple(desc->triple, sizeof(HCC_AMDGCN_AMDHSA_TRIPLE) - 1);
          if (triple.compare(HCC_AMDGCN_AMDHSA_TRIPLE))
            continue;

          std::string target(desc->triple + sizeof(HCC_AMDGCN_AMDHSA_TRIPLE),
                             desc->tripleSize - sizeof(HCC_AMDGCN_AMDHSA_TRIPLE));

          if (isCompatibleCodeObject(target, hip::getCurrentDevice()->devices()[0]->info().name_)) {
            hipModule_t module;
            if (hipSuccess == hipModuleLoadData(&module, reinterpret_cast<const void*>(
                reinterpret_cast<uintptr_t>(obheader) + desc->offset)))
              r.push_back(module);
              break;
          }
        }
      }
    });

    return r;
}

const std::unordered_map<uintptr_t, hipFunction_t>& functions()
{
  static std::unordered_map<uintptr_t, hipFunction_t> r;
  static std::once_flag f;

  std::call_once(f, []() {
    for (auto&& function : function_names()) {
      for (auto&& module : modules()) {
        hipFunction_t f;
        if (hipSuccess == hipModuleGetFunction(&f, module, function.second.c_str())) {
          r[function.first] = f;
        }
      }
    }
  });

  return r;
}


void hipLaunchKernelGGLImpl(
  uintptr_t function_address,
  const dim3& numBlocks,
  const dim3& dimBlocks,
  uint32_t sharedMemBytes,
  hipStream_t stream,
  void** kernarg)
{
  HIP_INIT();

  const auto it = functions().find(function_address);
  if (it == functions().cend())
    assert(0);

  hipModuleLaunchKernel(it->second,
    numBlocks.x, numBlocks.y, numBlocks.z,
    dimBlocks.x, dimBlocks.y, dimBlocks.z,
    sharedMemBytes, stream, nullptr, kernarg);
}

void hipLaunchCooperativeKernelGGLImpl(
  uintptr_t function_address,
  const dim3& numBlocks,
  const dim3& dimBlocks,
  uint32_t sharedMemBytes,
  hipStream_t stream,
  void** kernarg)
{
  HIP_INIT();

  hipLaunchCooperativeKernel(reinterpret_cast<void*>(function_address),
    numBlocks, dimBlocks, kernarg, sharedMemBytes, stream);
}

}

#endif // defined(ATI_OS_LINUX)

extern "C" hipError_t hipLaunchKernel(const void *hostFunction,
                                      dim3 gridDim,
                                      dim3 blockDim,
                                      void** args,
                                      size_t sharedMemBytes,
                                      hipStream_t stream)
{
  HIP_INIT_API(hipLaunchKernel, hostFunction, gridDim, blockDim, args, sharedMemBytes,
               stream);

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  int deviceId = (s != nullptr)? s->DeviceId() : ihipGetDevice();
  if (deviceId == -1) {
    DevLogPrintfError("Wrong Device Id: %d \n", deviceId);
    HIP_RETURN(hipErrorNoDevice);
  }
  hipFunction_t func = PlatformState::instance().getFunc(hostFunction, deviceId);
  if (func == nullptr) {
#ifdef ATI_OS_LINUX
    const auto it = hip_impl::functions().find(reinterpret_cast<uintptr_t>(hostFunction));
    if (it == hip_impl::functions().cend()) {
      DevLogPrintfError("Cannot find function: 0x%x \n", hostFunction);
      HIP_RETURN(hipErrorInvalidDeviceFunction);
    }
    func = it->second;
#else
    HIP_RETURN(hipErrorInvalidDeviceFunction);
#endif
  }

  HIP_RETURN(hipModuleLaunchKernel(func, gridDim.x, gridDim.y, gridDim.z,
                                    blockDim.x, blockDim.y, blockDim.z,
                                    sharedMemBytes, stream, args, nullptr));
}

// conversion routines between float and half precision
static inline std::uint32_t f32_as_u32(float f) { union { float f; std::uint32_t u; } v; v.f = f; return v.u; }
static inline float u32_as_f32(std::uint32_t u) { union { float f; std::uint32_t u; } v; v.u = u; return v.f; }
static inline int clamp_int(int i, int l, int h) { return std::min(std::max(i, l), h); }

// half float, the f16 is in the low 16 bits of the input argument
static inline float __convert_half_to_float(std::uint32_t a) noexcept {
  std::uint32_t u = ((a << 13) + 0x70000000U) & 0x8fffe000U;
  std::uint32_t v = f32_as_u32(u32_as_f32(u) * u32_as_f32(0x77800000U)/*0x1.0p+112f*/) + 0x38000000U;
  u = (a & 0x7fff) != 0 ? v : u;
  return u32_as_f32(u) * u32_as_f32(0x07800000U)/*0x1.0p-112f*/;
}

// float half with nearest even rounding
// The lower 16 bits of the result is the bit pattern for the f16
static inline std::uint32_t __convert_float_to_half(float a) noexcept {
  std::uint32_t u = f32_as_u32(a);
  int e = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
  std::uint32_t m = ((u >> 11) & 0xffe) | ((u & 0xfff) != 0);
  std::uint32_t i = 0x7c00 | (m != 0 ? 0x0200 : 0);
  std::uint32_t n = ((std::uint32_t)e << 12) | m;
  std::uint32_t s = (u >> 16) & 0x8000;
  int b = clamp_int(1-e, 0, 13);
  std::uint32_t d = (0x1000 | m) >> b;
  d |= (d << b) != (0x1000 | m);
  std::uint32_t v = e < 1 ? d : n;
  v = (v >> 2) + (((v & 0x7) == 3) | ((v & 0x7) > 5));
  v = e > 30 ? 0x7c00 : v;
  v = e == 143 ? i : v;
  return s | v;
}

extern "C" float __gnu_h2f_ieee(unsigned short h){
  return __convert_half_to_float((std::uint32_t) h);
}

extern "C" unsigned short __gnu_f2h_ieee(float f){
  return (unsigned short)__convert_float_to_half(f);
}

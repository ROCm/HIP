#include "hip_code_object.hpp"

#include <cstring>

#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "platform/program.hpp"
#include <elf/elf.hpp>

namespace hip {

uint64_t CodeObject::ElfSize(const void *emi) {
  return amd::Elf::getElfSize(emi);
}

bool CodeObject::isCompatibleCodeObject(const std::string& codeobj_target_id,
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

// This will be moved to COMGR eventually
hipError_t CodeObject::ExtractCodeObjectFromFile(amd::Os::FileDesc fdesc, size_t fsize,
                       const std::vector<const char*>& device_names,
                       std::vector<std::pair<const void*, size_t>>& code_objs) {

  hipError_t hip_error = hipSuccess;

  if (fdesc < 0) {
    return hipErrorFileNotFound;
  }

  // Map the file to memory, with offset 0.
  const void* image = nullptr;
  if (!amd::Os::MemoryMapFileDesc(fdesc, fsize, 0, &image)) {
    return hipErrorInvalidValue;
  }

  // retrieve code_objs{binary_image, binary_size} for devices
  hip_error = extractCodeObjectFromFatBinary(image, device_names, code_objs);

  // Unmap the file memory after extracting code object.
  if (!amd::Os::MemoryUnmapFile(image, fsize)) {
    return hipErrorInvalidValue;
  }

  return hip_error;
}

// This will be moved to COMGR eventually
hipError_t CodeObject::ExtractCodeObjectFromMemory(const void* data,
                       const std::vector<const char*>& device_names,
                       std::vector<std::pair<const void*, size_t>>& code_objs,
                       std::string& uri) {

  // Get the URI from memory
  if (!amd::Os::GetURIFromMemory(data, 0, uri)) {
    return hipErrorInvalidValue;
  }

  return extractCodeObjectFromFatBinary(data, device_names, code_objs);
}

// This will be moved to COMGR eventually
hipError_t CodeObject::extractCodeObjectFromFatBinary(const void* data,
                       const std::vector<const char*>& device_names,
                       std::vector<std::pair<const void*, size_t>>& code_objs) {
  std::string magic((const char*)data, sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC_STR) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC_STR)) {
    return hipErrorInvalidKernelFile;
  }

  code_objs.resize(device_names.size());
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

    for (size_t dev = 0; dev < device_names.size(); ++dev) {
      const char* name = device_names[dev];

      if (!isCompatibleCodeObject(target, name)) {
          continue;
      }
      code_objs[dev] = std::make_pair(image, size);
      num_code_objs++;
    }
  }
  if (num_code_objs == device_names.size()) {
    return hipSuccess;
  } else {
    guarantee(false && "hipErrorNoBinaryForGpu: Coudn't find binary for current devices!");
    return hipErrorNoBinaryForGpu;
  }
}

hipError_t DynCO::loadCodeObject(const char* fname, const void* image) {

  amd::ScopedLock lock(dclock_);

  // Number of devices = 1 in dynamic code object
  fb_info_ = new FatBinaryInfo(fname, image);
  std::vector<hip::Device*> devices = { g_devices[ihipGetDevice()] };
  IHIP_RETURN_ONFAIL(fb_info_->ExtractFatBinary(devices));

  // No Lazy loading for DynCO
  IHIP_RETURN_ONFAIL(fb_info_->BuildProgram(ihipGetDevice()));

  // Define Global variables
  IHIP_RETURN_ONFAIL(populateDynGlobalVars());

  // Define Global functions
  IHIP_RETURN_ONFAIL(populateDynGlobalFuncs());

  return hipSuccess;
}

//Dynamic Code Object
DynCO::~DynCO() {
  amd::ScopedLock lock(dclock_);

  for (auto& elem : vars_) {
    delete elem.second;
  }
  vars_.clear();

  for (auto& elem : functions_) {
    delete elem.second;
  }
  functions_.clear();

  delete fb_info_;
}

hipError_t DynCO::getDeviceVar(DeviceVar** dvar, std::string var_name, int device_id) {
  amd::ScopedLock lock(dclock_);

  auto it = vars_.find(var_name);
  if (it == vars_.end()) {
    DevLogPrintfError("Cannot find the Var: %s ", var_name.c_str());
    return hipErrorNotFound;
  }

  it->second->getDeviceVar(dvar, device_id, module());
  return hipSuccess;
}

hipError_t DynCO::getDynFunc(hipFunction_t* hfunc, std::string func_name) {
  amd::ScopedLock lock(dclock_);

  auto it = functions_.find(func_name);
  if (it == functions_.end()) {
    DevLogPrintfError("Cannot find the function: %s ", func_name.c_str());
    return hipErrorNotFound;
  }

  /* See if this could be solved */
  return it->second->getDynFunc(hfunc, module());
}

hipError_t DynCO::populateDynGlobalVars() {
  amd::ScopedLock lock(dclock_);

  std::vector<std::string> var_names;
  std::vector<std::string> undef_var_names;

  //For Dynamic Modules there is only one hipFatBinaryDevInfo_
  device::Program* dev_program
    = fb_info_->GetProgram(ihipGetDevice())->getDeviceProgram
                          (*hip::getCurrentDevice()->devices()[0]);

  if (!dev_program->getGlobalVarFromCodeObj(&var_names)) {
    DevLogPrintfError("Could not get Global vars from Code Obj for Module: 0x%x \n", module());
    return hipErrorSharedObjectSymbolNotFound;
  }

  if (!dev_program->getUndefinedVarFromCodeObj(&undef_var_names)) {
    DevLogPrintfError("Could not get undefined Variables for Module: 0x%x \n", module());
    return hipErrorSharedObjectSymbolNotFound;
  }

  for (auto& elem : var_names) {
    vars_.insert(std::make_pair(elem, new Var(elem, Var::DeviceVarKind::DVK_Variable, 0, 0, 0, nullptr)));
  }

  for (auto& elem : undef_var_names) {
    vars_.insert(std::make_pair(elem, new Var(elem, Var::DeviceVarKind::DVK_Texture, 0, 0, 0, nullptr)));
  }

  return hipSuccess;
}

hipError_t DynCO::populateDynGlobalFuncs() {
  amd::ScopedLock lock(dclock_);

  std::vector<std::string> func_names;
  device::Program* dev_program
    = fb_info_->GetProgram(ihipGetDevice())->getDeviceProgram(
                           *hip::getCurrentDevice()->devices()[0]);

  // Get all the global func names from COMGR
  if (!dev_program->getGlobalFuncFromCodeObj(&func_names)) {
    DevLogPrintfError("Could not get Global Funcs from Code Obj for Module: 0x%x \n", module());
    return hipErrorSharedObjectSymbolNotFound;
  }

  for (auto& elem : func_names) {
    functions_.insert(std::make_pair(elem, new Function(elem)));
  }

  return hipSuccess;
}

//Static Code Object
StatCO::StatCO() {
}

StatCO::~StatCO() {
  amd::ScopedLock lock(sclock_);

  for (auto& elem : functions_) {
    delete elem.second;
  }
  functions_.clear();

  for (auto& elem : vars_) {
    delete elem.second;
  }
  vars_.clear();
}

hipError_t StatCO::digestFatBinary(const void* data, FatBinaryInfo*& programs) {
  amd::ScopedLock lock(sclock_);

  if (programs != nullptr) {
    return hipSuccess;
  }

  // Create a new fat binary object and extract the fat binary for all devices.
  programs = new FatBinaryInfo(nullptr, data);
  IHIP_RETURN_ONFAIL(programs->ExtractFatBinary(g_devices));

  return hipSuccess;
}

FatBinaryInfo** StatCO::addFatBinary(const void* data, bool initialized) {
  amd::ScopedLock lock(sclock_);

  if (initialized) {
    digestFatBinary(data, modules_[data]);
  }

  return &modules_[data];
}

hipError_t StatCO::removeFatBinary(FatBinaryInfo** module) {
  amd::ScopedLock lock(sclock_);

  auto vit = vars_.begin();
  while (vit != vars_.end()) {
    if (vit->second->moduleInfo() == module) {
      delete vit->second;
      vit = vars_.erase(vit);
    } else {
      ++vit;
    }
  }

  auto fit = functions_.begin();
  while (fit != functions_.end()) {
    if (fit->second->moduleInfo() == module) {
      delete fit->second;
      fit = functions_.erase(fit);
    } else {
      ++fit;
    }
  }

  auto mit = modules_.begin();
  while (mit != modules_.end()) {
    if (&mit->second == module) {
      delete mit->second;
      mit = modules_.erase(mit);
    } else {
      ++mit;
    }
  }

  return hipSuccess;
}

hipError_t StatCO::registerStatFunction(const void* hostFunction, Function* func) {
  amd::ScopedLock lock(sclock_);

  if (functions_.find(hostFunction) != functions_.end()) {
    DevLogPrintfError("hostFunctionPtr: 0x%x already exists", hostFunction);
  }
  functions_.insert(std::make_pair(hostFunction, func));

  return hipSuccess;
}

hipError_t StatCO::getStatFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId) {
  amd::ScopedLock lock(sclock_);

  const auto it = functions_.find(hostFunction);
  if (it == functions_.end()) {
    return hipErrorInvalidSymbol;
  }

  return it->second->getStatFunc(hfunc, deviceId);
}

hipError_t StatCO::getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction, int deviceId) {
  amd::ScopedLock lock(sclock_);

  const auto it = functions_.find(hostFunction);
  if (it == functions_.end()) {
    return hipErrorInvalidSymbol;
  }

  return it->second->getStatFuncAttr(func_attr, deviceId);
}

hipError_t StatCO::registerStatGlobalVar(const void* hostVar, Var* var) {
  amd::ScopedLock lock(sclock_);

  if (vars_.find(hostVar) != vars_.end()) {
    return hipErrorInvalidSymbol;
  }

  vars_.insert(std::make_pair(hostVar, var));
  return hipSuccess;
}

hipError_t StatCO::getStatGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                                    size_t* size_ptr) {
  amd::ScopedLock lock(sclock_);

  const auto it = vars_.find(hostVar);
  if (it == vars_.end()) {
    return hipErrorInvalidSymbol;
  }

  DeviceVar* dvar = nullptr;
  IHIP_RETURN_ONFAIL(it->second->getStatDeviceVar(&dvar, deviceId));

  *dev_ptr = dvar->device_ptr();
  *size_ptr = dvar->size();
  return hipSuccess;
}

hipError_t StatCO::getStatGlobalVarByName(std::string hostVar, int deviceId, hipModule_t hmod,
                                          hipDeviceptr_t* dev_ptr, size_t* size_ptr) {
  amd::ScopedLock lock(sclock_);

  for (auto& elem : vars_) {
    if ((elem.second->name() == hostVar)
        && (elem.second->module(deviceId) == hmod)) {
      *dev_ptr = elem.second->device_ptr(deviceId);
      *size_ptr = elem.second->device_size(deviceId);
      return hipSuccess;
    }
  }

  return hipErrorNotFound;
}
}; //namespace: hip

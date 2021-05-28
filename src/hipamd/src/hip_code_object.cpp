/*
Copyright (c) 2015-2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "hip_code_object.hpp"
#include "amd_hsa_elf.hpp"

#include <cstring>

#include <hip/amd_detail/amd_driver_types.h>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "platform/program.hpp"
#include <elf/elf.hpp>

hipError_t ihipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                      amd::HostQueue& queue, bool isAsync = false);
hipError_t ihipFree(void* ptr);
//forward declaration of methods required for managed variables
hipError_t ihipMallocManaged(void** ptr, size_t size, unsigned int align = 0);
namespace {
size_t constexpr strLiteralLength(char const* str) {
  return *str ? 1 + strLiteralLength(str + 1) : 0;
}
constexpr char const* CLANG_OFFLOAD_BUNDLER_MAGIC_STR = "__CLANG_OFFLOAD_BUNDLE__";
constexpr char const* OFFLOAD_KIND_HIP = "hip";
constexpr char const* OFFLOAD_KIND_HIPV4 = "hipv4";
constexpr char const* OFFLOAD_KIND_HCC = "hcc";
constexpr char const* AMDGCN_TARGET_TRIPLE = "amdgcn-amd-amdhsa-";

// ClangOFFLOADBundle info.
static constexpr size_t bundle_magic_string_size =
    strLiteralLength(CLANG_OFFLOAD_BUNDLER_MAGIC_STR);

// Clang Offload bundler description & Header.
struct __ClangOffloadBundleInfo {
  uint64_t offset;
  uint64_t size;
  uint64_t bundleEntryIdSize;
  const char bundleEntryId[1];
};

struct __ClangOffloadBundleHeader {
  const char magic[bundle_magic_string_size - 1];
  uint64_t numOfCodeObjects;
  __ClangOffloadBundleInfo desc[1];
};
}  // namespace

namespace hip {

uint64_t CodeObject::ElfSize(const void *emi) {
  return amd::Elf::getElfSize(emi);
}

static bool getProcName(uint32_t EFlags, std::string& proc_name, bool& xnackSupported,
                        bool& sramEccSupported) {
  switch (EFlags & EF_AMDGPU_MACH) {
    case EF_AMDGPU_MACH_AMDGCN_GFX700:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx700";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX701:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx701";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX702:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx702";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX703:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx703";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX704:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx704";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX705:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx705";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX801:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx801";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX802:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx802";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX803:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx803";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX805:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx805";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX810:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx810";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX900:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx900";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX902:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx902";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX904:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx904";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX906:
      xnackSupported = true;
      sramEccSupported = true;
      proc_name = "gfx906";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX908:
      xnackSupported = true;
      sramEccSupported = true;
      proc_name = "gfx908";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX909:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx909";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX90A:
      xnackSupported = true;
      sramEccSupported = true;
      proc_name = "gfx90a";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX90C:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx90c";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1010:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx1010";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1011:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx1011";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1012:
      xnackSupported = true;
      sramEccSupported = false;
      proc_name = "gfx1012";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1030:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1030";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1031:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1031";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1032:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1032";
      break;
    case EF_AMDGPU_MACH_AMDGCN_GFX1033:
      xnackSupported = false;
      sramEccSupported = false;
      proc_name = "gfx1033";
      break;
    default:
      return false;
  }
  return true;
}

static bool getTripleTargetIDFromCodeObject(const void* code_object, std::string& target_id,
                                            unsigned& co_version) {
  if (!code_object) return false;
  const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(code_object);
  if (ehdr->e_machine != EM_AMDGPU) return false;
  if (ehdr->e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) return false;

  bool isXnackSupported{false}, isSramEccSupported{false};

  std::string proc_name;
  if (!getProcName(ehdr->e_flags, proc_name, isXnackSupported, isSramEccSupported)) return false;
  target_id = std::string(AMDGCN_TARGET_TRIPLE) + '-' + proc_name;

  switch (ehdr->e_ident[EI_ABIVERSION]) {
    case ELFABIVERSION_AMDGPU_HSA_V2: {
      co_version = 2;
      return false;
    }

    case ELFABIVERSION_AMDGPU_HSA_V3: {
      co_version = 3;
      if (isSramEccSupported) {
        if (ehdr->e_flags & EF_AMDGPU_FEATURE_SRAMECC_V3)
          target_id += ":sramecc+";
        else
          target_id += ":sramecc-";
      }
      if (isXnackSupported) {
        if (ehdr->e_flags & EF_AMDGPU_FEATURE_XNACK_V3)
          target_id += ":xnack+";
        else
          target_id += ":xnack-";
      }
      break;
    }

    case ELFABIVERSION_AMDGPU_HSA_V4: {
      co_version = 4;
      unsigned co_sram_value = (ehdr->e_flags) & EF_AMDGPU_FEATURE_SRAMECC_V4;
      if (co_sram_value == EF_AMDGPU_FEATURE_SRAMECC_OFF_V4)
        target_id += ":sramecc-";
      else if (co_sram_value == EF_AMDGPU_FEATURE_SRAMECC_ON_V4)
        target_id += ":sramecc+";

      unsigned co_xnack_value = (ehdr->e_flags) & EF_AMDGPU_FEATURE_XNACK_V4;
      if (co_xnack_value == EF_AMDGPU_FEATURE_XNACK_OFF_V4)
        target_id += ":xnack-";
      else if (co_xnack_value == EF_AMDGPU_FEATURE_XNACK_ON_V4)
        target_id += ":xnack+";
      break;
    }

    default: {
      return false;
    }
  }
  return true;
}

// Consumes the string 'consume_' from the starting of the given input
// eg: input = amdgcn-amd-amdhsa--gfx908 and consume_ is amdgcn-amd-amdhsa--
// input will become gfx908.
static bool consume(std::string& input, std::string consume_) {
  if (input.substr(0, consume_.size()) != consume_) {
    return false;
  }
  input = input.substr(consume_.size());
  return true;
}

// Trim String till character, will be used to get gpuname
// example: input is gfx908:sram-ecc+ and trim char is :
// input will become sram-ecc+.
static std::string trimName(std::string& input, char trim) {
  auto pos_ = input.find(trim);
  auto res = input;
  if (pos_ == std::string::npos) {
    input = "";
  } else {
    res = input.substr(0, pos_);
    input = input.substr(pos_);
  }
  return res;
}

static char getFeatureValue(std::string& input, std::string feature) {
  char res = ' ';
  if (consume(input, std::move(feature))) {
    res = input[0];
    input = input.substr(1);
  }
  return res;
}

static bool getTargetIDValue(std::string& input, std::string& processor, char& sramecc_value,
                             char& xnack_value) {
  processor = trimName(input, ':');
  sramecc_value = getFeatureValue(input, std::string(":sramecc"));
  if (sramecc_value != ' ' && sramecc_value != '+' && sramecc_value != '-') return false;
  xnack_value = getFeatureValue(input, std::string(":xnack"));
  if (xnack_value != ' ' && xnack_value != '+' && xnack_value != '-') return false;
  return true;
}

static bool getTripleTargetID(std::string bundled_co_entry_id, const void* code_object,
                              std::string& co_triple_target_id, unsigned& co_version) {
  std::string offload_kind = trimName(bundled_co_entry_id, '-');
  if (offload_kind != OFFLOAD_KIND_HIPV4 && offload_kind != OFFLOAD_KIND_HIP &&
      offload_kind != OFFLOAD_KIND_HCC)
    return false;

  if (offload_kind != OFFLOAD_KIND_HIPV4)
    return getTripleTargetIDFromCodeObject(code_object, co_triple_target_id, co_version);

  // For code object V4 onwards the bundled code object entry ID correctly
  // specifies the target tripple.
  co_version = 4;
  co_triple_target_id = bundled_co_entry_id.substr(1);
  return true;
}

static bool isCodeObjectCompatibleWithDevice(std::string co_triple_target_id,
                                             std::string agent_triple_target_id) {
  // Primitive Check
  if (co_triple_target_id == agent_triple_target_id) return true;

  // Parse code object triple target id
  if (!consume(co_triple_target_id, std::string(AMDGCN_TARGET_TRIPLE) + '-')) {
    return false;
  }

  std::string co_processor;
  char co_sram_ecc, co_xnack;
  if (!getTargetIDValue(co_triple_target_id, co_processor, co_sram_ecc, co_xnack)) {
    return false;
  }

  if (!co_triple_target_id.empty()) return false;

  // Parse agent isa triple target id
  if (!consume(agent_triple_target_id, std::string(AMDGCN_TARGET_TRIPLE) + '-')) {
    return false;
  }

  std::string agent_isa_processor;
  char isa_sram_ecc, isa_xnack;
  if (!getTargetIDValue(agent_triple_target_id, agent_isa_processor, isa_sram_ecc, isa_xnack)) {
    return false;
  }

  if (!agent_triple_target_id.empty()) return false;

  // Check for compatibility
  if (agent_isa_processor != co_processor) return false;
  if (co_sram_ecc != ' ') {
    if (co_sram_ecc != isa_sram_ecc) return false;
  }
  if (co_xnack != ' ') {
    if (co_xnack != isa_xnack) return false;
  }

  return true;
}

// This will be moved to COMGR eventually
hipError_t CodeObject::ExtractCodeObjectFromFile(amd::Os::FileDesc fdesc, size_t fsize,
                       const void ** image, const std::vector<std::string>& device_names,
                       std::vector<std::pair<const void*, size_t>>& code_objs) {

  hipError_t hip_error = hipSuccess;

  if (fdesc < 0) {
    return hipErrorFileNotFound;
  }

  // Map the file to memory, with offset 0.
  //file will be unmapped in ModuleUnload
  //const void* image = nullptr;
  if (!amd::Os::MemoryMapFileDesc(fdesc, fsize, 0, image)) {
    return hipErrorInvalidValue;
  }

  // retrieve code_objs{binary_image, binary_size} for devices
  hip_error = extractCodeObjectFromFatBinary(*image, device_names, code_objs);

  return hip_error;
}

// This will be moved to COMGR eventually
hipError_t CodeObject::ExtractCodeObjectFromMemory(const void* data,
                       const std::vector<std::string>& device_names,
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
                       const std::vector<std::string>& agent_triple_target_ids,
                       std::vector<std::pair<const void*, size_t>>& code_objs) {
  std::string magic((const char*)data, bundle_magic_string_size);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC_STR)) {
    return hipErrorInvalidKernelFile;
  }

  // Initialize Code objects
  code_objs.reserve(agent_triple_target_ids.size());
  for (size_t i = 0; i < agent_triple_target_ids.size(); i++) {
    code_objs.push_back(std::make_pair(nullptr, 0));
  }

  const auto obheader = reinterpret_cast<const __ClangOffloadBundleHeader*>(data);
  const auto* desc = &obheader->desc[0];
  size_t num_code_objs = code_objs.size();
  for (uint64_t i = 0; i < obheader->numOfCodeObjects; ++i,
                desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
                    reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) +
                    desc->bundleEntryIdSize)) {
    const void* image =
        reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(obheader) + desc->offset);
    const size_t image_size = desc->size;

    if (num_code_objs == 0) break;
    std::string bundleEntryId{desc->bundleEntryId, desc->bundleEntryIdSize};

    unsigned co_version = 0;
    std::string co_triple_target_id;
    if (!getTripleTargetID(bundleEntryId, image, co_triple_target_id, co_version)) continue;

    for (size_t dev = 0; dev < agent_triple_target_ids.size(); ++dev) {
      if (code_objs[dev].first) continue;
      if (isCodeObjectCompatibleWithDevice(co_triple_target_id, agent_triple_target_ids[dev])) {
        code_objs[dev] = std::make_pair(image, image_size);
        --num_code_objs;
      }
    }
  }
  if (num_code_objs == 0) {
    return hipSuccess;
  } else {
    LogPrintfError("%s",
                   "hipErrorNoBinaryForGpu: Unable to find code object for all current devices!");
    LogPrintfError("%s", "  Devices:");
    for (size_t i = 0; i < agent_triple_target_ids.size(); i++) {
      LogPrintfError("    %s - [%s]", agent_triple_target_ids[i].c_str(),
                     ((code_objs[i].first) ? "Found" : "Not Found"));
    }
    const auto obheader = reinterpret_cast<const __ClangOffloadBundleHeader*>(data);
    const auto* desc = &obheader->desc[0];
    LogPrintfError("%s", "  Bundled Code Objects:");
    for (uint64_t i = 0; i < obheader->numOfCodeObjects; ++i,
                  desc = reinterpret_cast<const __ClangOffloadBundleInfo*>(
                      reinterpret_cast<uintptr_t>(&desc->bundleEntryId[0]) +
                      desc->bundleEntryIdSize)) {
      std::string bundleEntryId{desc->bundleEntryId, desc->bundleEntryIdSize};
      const void* image =
          reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(obheader) + desc->offset);

      unsigned co_version = 0;
      std::string co_triple_target_id;
      bool valid_co = getTripleTargetID(bundleEntryId, image, co_triple_target_id, co_version);

      if (valid_co) {
        LogPrintfError("    %s - [code object v%u is %s]", bundleEntryId.c_str(), co_version,
                       co_triple_target_id.c_str());
      } else {
        LogPrintfError("    %s - [Unsupported]", bundleEntryId.c_str());
      }
    }

    guarantee(false, "hipErrorNoBinaryForGpu: Unable to find code object for all current devices!");
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
    if(elem.second->getVarKind() == Var::DVK_Managed) {
      ihipFree(elem.second->getManagedVarPtr());
    }
    delete elem.second;
  }
  vars_.clear();

  for (auto& elem : functions_) {
    delete elem.second;
  }
  functions_.clear();

  delete fb_info_;
}

hipError_t DynCO::getDeviceVar(DeviceVar** dvar, std::string var_name) {
  amd::ScopedLock lock(dclock_);

  CheckDeviceIdMatch();

  auto it = vars_.find(var_name);
  if (it == vars_.end()) {
    LogPrintfError("Cannot find the Var: %s ", var_name.c_str());
    return hipErrorNotFound;
  }

  it->second->getDeviceVar(dvar, device_id_, module());
  return hipSuccess;
}

hipError_t DynCO::getDynFunc(hipFunction_t* hfunc, std::string func_name) {
  amd::ScopedLock lock(dclock_);

  CheckDeviceIdMatch();

  if(hfunc == nullptr) {
    return hipErrorInvalidValue;
  }

  auto it = functions_.find(func_name);
  if (it == functions_.end()) {
    LogPrintfError("Cannot find the function: %s ", func_name.c_str());
    return hipErrorNotFound;
  }

  /* See if this could be solved */
  return it->second->getDynFunc(hfunc, module());
}

hipError_t DynCO::initDynManagedVars(const std::string& managedVar) {
  amd::ScopedLock lock(dclock_);
  DeviceVar* dvar;
  void* pointer = nullptr;
  hipError_t status = hipSuccess;
  // To get size of the managed variable
  status = getDeviceVar(&dvar, managedVar + ".managed");
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to get .managed device variable:%s",
            status, managedVar.c_str());
    return status;
  }
  // Allocate managed memory for these symbols
  status = ihipMallocManaged(&pointer, dvar->size());
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to allocate managed memory", status);
    guarantee(false, "Error during allocation of managed memory!");
  }
  // update as manager variable and set managed memory pointer and size
  auto it = vars_.find(managedVar);
  it->second->setManagedVarInfo(pointer, dvar->size());

  // copy initial value to the managed variable to the managed memory allocated
  amd::HostQueue* queue = hip::getNullStream();
  if (queue != nullptr) {
    status = ihipMemcpy(pointer, reinterpret_cast<address>(dvar->device_ptr()), dvar->size(),
                        hipMemcpyDeviceToDevice, *queue);
    if (status != hipSuccess) {
      ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to copy device ptr:%s", status,
              managedVar.c_str());
      return status;
    }
  } else {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Host Queue is NULL");
    return hipErrorInvalidResourceHandle;
  }

  // Get deivce ptr to initialize with managed memory pointer
  status = getDeviceVar(&dvar, managedVar);
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to get managed device variable:%s",
            status, managedVar.c_str());
    return status;
  }
  // copy managed memory pointer to the managed device variable
  status = ihipMemcpy(reinterpret_cast<address>(dvar->device_ptr()), &pointer, dvar->size(),
                      hipMemcpyHostToDevice, *queue);
  if (status != hipSuccess) {
    ClPrint(amd::LOG_ERROR, amd::LOG_API, "Status %d, failed to copy device ptr:%s", status,
            managedVar.c_str());
    return status;
  }
  return status;
}

hipError_t DynCO::populateDynGlobalVars() {
  amd::ScopedLock lock(dclock_);
  std::vector<std::string> var_names;
  std::vector<std::string> undef_var_names;
  std::string managedVarExt = ".managed";
  // For Dynamic Modules there is only one hipFatBinaryDevInfo_
  device::Program* dev_program = fb_info_->GetProgram(ihipGetDevice())
                                     ->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  if (!dev_program->getGlobalVarFromCodeObj(&var_names)) {
    LogPrintfError("Could not get Global vars from Code Obj for Module: 0x%x \n", module());
    return hipErrorSharedObjectSymbolNotFound;
  }

  for (auto& elem : var_names) {
    vars_.insert(
        std::make_pair(elem, new Var(elem, Var::DeviceVarKind::DVK_Variable, 0, 0, 0, nullptr)));
  }

  for (auto& elem : var_names) {
    if (elem.find(managedVarExt) != std::string::npos) {
      std::string managedVar = elem;
      managedVar.erase(managedVar.length() - managedVarExt.length(), managedVarExt.length());
      initDynManagedVars(managedVar);
    }
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
    LogPrintfError("Could not get Global Funcs from Code Obj for Module: 0x%x \n", module());
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

  auto it = managedVars_.begin();
  while (it != managedVars_.end()) {
    if ((*it)->moduleInfo() == module) {
      ihipFree((*it)->getManagedVarPtr());
      delete *it;
      managedVars_.erase(it);
    } else {
      ++it;
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

hipError_t StatCO::registerStatManagedVar(Var* var) {
  managedVars_.emplace_back(var);
  return hipSuccess;
}

hipError_t StatCO::initStatManagedVarDevicePtr(int deviceId) {
  amd::ScopedLock lock(sclock_);

  if (managedVarsDevicePtrInitalized_.find(deviceId) == managedVarsDevicePtrInitalized_.end() ||
      !managedVarsDevicePtrInitalized_[deviceId]) {
    for (auto var : managedVars_) {
      DeviceVar* dvar = nullptr;
      IHIP_RETURN_ONFAIL(var->getStatDeviceVar(&dvar, deviceId));

      amd::HostQueue* queue = hip::getNullStream();
      if(queue != nullptr) {
        ihipMemcpy(reinterpret_cast<address>(dvar->device_ptr()), var->getManagedVarPtr(),
                  dvar->size(), hipMemcpyHostToDevice, *queue);
      } else {
        ClPrint(amd::LOG_ERROR, amd::LOG_API, "Host Queue is NULL");
        return hipErrorInvalidResourceHandle;
      }
    }
    managedVarsDevicePtrInitalized_[deviceId] = true;
  }
  return hipSuccess;
}
}; //namespace: hip

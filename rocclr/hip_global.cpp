#include "hip_global.hpp"

#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_code_object.hpp"
#include "platform/program.hpp"

namespace hip {

//Device Vars
DeviceVar::DeviceVar(std::string name, hipModule_t hmod) : shadowVptr(nullptr), name_(name),
                                                           amd_mem_obj_(nullptr), device_ptr_(nullptr),
                                                           size_(0) {
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));
  device::Program* dev_program = program->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);
  if (dev_program == nullptr) {
    DevLogPrintfError("Cannot get Device Function for module: 0x%x \n", hmod);
    guarantee(false);
  }

  if(!dev_program->createGlobalVarObj(&amd_mem_obj_, &device_ptr_, &size_, name.c_str())) {
    DevLogPrintfError("Cannot create Global Var obj for symbol: %s \n", name.c_str());
    guarantee(false);
  }

  // Handle size 0 symbols
  if (size_ != 0) {
    if (amd_mem_obj_ == nullptr || device_ptr_ == nullptr) {
      DevLogPrintfError("Cannot get memory for creating device Var: %s", name.c_str());
      guarantee(false);
    }
    amd::MemObjMap::AddMemObj(device_ptr_, amd_mem_obj_);
  }
}

DeviceVar::~DeviceVar() {
  if (amd_mem_obj_ != nullptr) {
    amd::MemObjMap::RemoveMemObj(device_ptr_);
    amd_mem_obj_->release();
  }

  if (shadowVptr != nullptr) {
    textureReference* texRef = reinterpret_cast<textureReference*>(shadowVptr);
    delete texRef;
    shadowVptr = nullptr;
  }

  device_ptr_ = nullptr;
  size_ = 0;
}

//Device Functions
DeviceFunc::DeviceFunc(std::string name, hipModule_t hmod) : dflock_("function lock"),
                       name_(name), kernel_(nullptr) {
  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));

  const amd::Symbol *symbol = program->findSymbol(name.c_str());
  if (symbol == nullptr) {
    DevLogPrintfError("Cannot find Symbol with name: %s \n", name.c_str());
    guarantee(false);
  }

  kernel_ = new amd::Kernel(*program, *symbol, name);
  if (kernel_ == nullptr) {
    DevLogPrintfError("Cannot create kernel with name: %s \n", name.c_str());
    guarantee(false);
  }
}

DeviceFunc::~DeviceFunc() {
  if (kernel_ != nullptr) {
    kernel_->release();
  }
}

//Abstract functions
Function::Function(std::string name, FatBinaryInfoType* modules)
                   : name_(name), modules_(modules) {
  dFunc_.resize(g_devices.size());
}

Function::~Function() {
  for (auto& elem : dFunc_) {
    delete elem;
  }
  name_ = "";
  modules_ = nullptr;
}

hipError_t Function::getDynFunc(hipFunction_t* hfunc, hipModule_t hmod) {
  guarantee(dFunc_.size() == g_devices.size());
  if (dFunc_[ihipGetDevice()] == nullptr) {
    dFunc_[ihipGetDevice()] = new DeviceFunc(name_, hmod);
  }
  *hfunc = dFunc_[ihipGetDevice()]->asHipFunction();

  return hipSuccess;
}

hipError_t Function::getStatFunc(hipFunction_t* hfunc, int deviceId) {
  guarantee(modules_ != nullptr);
  guarantee(deviceId >= 0);
  guarantee(static_cast<size_t>(deviceId) < modules_->size());

  hipModule_t module = (*modules_)[deviceId].first;
  FatBinaryMetaInfo* fb_meta = (*modules_)[deviceId].second;

  if (!fb_meta->built()) {
    IHIP_RETURN_ONFAIL(CodeObject::add_program(deviceId, module, fb_meta->binary_ptr(),
                                               fb_meta->binary_size()));
    IHIP_RETURN_ONFAIL(CodeObject::build_module(module, g_devices[deviceId]->devices()));
    fb_meta->set_built();
  }

  if (dFunc_[deviceId] == nullptr) {
    dFunc_[deviceId] = new DeviceFunc(name_, (*modules_)[deviceId].first);
  }
  *hfunc = dFunc_[deviceId]->asHipFunction();

  return hipSuccess;
}

hipError_t Function::getStatFuncAttr(hipFuncAttributes* func_attr, int deviceId) {
  guarantee(modules_ != nullptr);
  guarantee(deviceId >= 0);
  guarantee(static_cast<size_t>(deviceId) < modules_->size());

  hipModule_t module = (*modules_)[deviceId].first;
  FatBinaryMetaInfo* fb_meta = (*modules_)[deviceId].second;

  if (!fb_meta->built()) {
    IHIP_RETURN_ONFAIL(CodeObject::add_program(deviceId, module, fb_meta->binary_ptr(),
                                               fb_meta->binary_size()));
    IHIP_RETURN_ONFAIL(CodeObject::build_module(module, g_devices[deviceId]->devices()));
    fb_meta->set_built();
  }

  if (dFunc_[deviceId] == nullptr) {
    dFunc_[deviceId] = new DeviceFunc(name_, (*modules_)[deviceId].first);
  }

  const std::vector<amd::Device*>& devices = amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false);

  amd::Kernel* kernel = dFunc_[deviceId]->kernel();
  const device::Kernel::WorkGroupInfo* wginfo = kernel->getDeviceKernel(*devices[deviceId])->workGroupInfo();
  func_attr->sharedSizeBytes = static_cast<int>(wginfo->localMemSize_);
  func_attr->binaryVersion = static_cast<int>(kernel->signature().version());
  func_attr->cacheModeCA = 0;
  func_attr->constSizeBytes = 0;
  func_attr->localSizeBytes = wginfo->privateMemSize_;
  func_attr->maxDynamicSharedSizeBytes = static_cast<int>(wginfo->availableLDSSize_
                                                          - wginfo->localMemSize_);

  func_attr->maxThreadsPerBlock = static_cast<int>(wginfo->size_);
  func_attr->numRegs = static_cast<int>(wginfo->usedVGPRs_);
  func_attr->preferredShmemCarveout = 0;
  func_attr->ptxVersion = 30;


  return hipSuccess;
}

//Abstract Vars
Var::Var(std::string name, DeviceVarKind dVarKind, size_t size, int type, int norm,
         FatBinaryInfoType* modules) : name_(name), dVarKind_(dVarKind), size_(size),
         type_(type), norm_(norm), modules_(modules) {
  dVar_.resize(g_devices.size());
}

Var::~Var() {
  for (auto& elem : dVar_) {
    delete elem;
  }
  modules_ = nullptr;
}

hipError_t Var::getDeviceVar(DeviceVar** dvar, int deviceId, hipModule_t hmod) {
  guarantee(deviceId >= 0);
  guarantee(static_cast<size_t>(deviceId) < g_devices.size());
  guarantee(dVar_.size() == g_devices.size());

  if (dVar_[deviceId] == nullptr) {
    dVar_[deviceId] = new DeviceVar(name_, hmod);
  }

  *dvar = dVar_[deviceId];
  return hipSuccess;
}

hipError_t Var::getStatDeviceVar(DeviceVar** dvar, int deviceId) {
  guarantee(deviceId >= 0);
  guarantee(static_cast<size_t>(deviceId) < g_devices.size());

  hipModule_t module = (*modules_)[deviceId].first;
  FatBinaryMetaInfo* fb_meta = (*modules_)[deviceId].second;

  if (!fb_meta->built()) {
    IHIP_RETURN_ONFAIL(CodeObject::add_program(deviceId, module, fb_meta->binary_ptr(),
                                               fb_meta->binary_size()));
    IHIP_RETURN_ONFAIL(CodeObject::build_module(module, g_devices[deviceId]->devices()));
    fb_meta->set_built();
  }

  if (dVar_[deviceId] == nullptr) {
    dVar_[deviceId] = new DeviceVar(name_, (*modules_)[deviceId].first);
  }

  *dvar = dVar_[deviceId];
  return hipSuccess;
}

}; //namespace: hip

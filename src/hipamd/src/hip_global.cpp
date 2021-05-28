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
    LogPrintfError("Cannot get Device Program for module: 0x%x \n", hmod);
    guarantee(false, "Cannot get Device Program");
  }

  if(!dev_program->createGlobalVarObj(&amd_mem_obj_, &device_ptr_, &size_, name.c_str())) {
    LogPrintfError("Cannot create Global Var obj for symbol: %s \n", name.c_str());
    guarantee(false, "Cannot create GlobalVar Obj");
  }

  // Handle size 0 symbols
  if (size_ != 0) {
    if (amd_mem_obj_ == nullptr || device_ptr_ == nullptr) {
      LogPrintfError("Cannot get memory for creating device Var: %s", name.c_str());
      guarantee(false, "Cannot get memory for creating device var");
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
    LogPrintfError("Cannot find Symbol with name: %s \n", name.c_str());
    guarantee(false, "Cannot find Symbol");
  }

  kernel_ = new amd::Kernel(*program, *symbol, name);
  if (kernel_ == nullptr) {
    LogPrintfError("Cannot create kernel with name: %s \n", name.c_str());
    guarantee(false, "Cannot Create kernel");
  }
}

DeviceFunc::~DeviceFunc() {
  if (kernel_ != nullptr) {
    kernel_->release();
  }
}

//Abstract functions
Function::Function(std::string name, FatBinaryInfo** modules)
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
  guarantee((dFunc_.size() == g_devices.size()), "dFunc Size mismatch");
  if (dFunc_[ihipGetDevice()] == nullptr) {
    dFunc_[ihipGetDevice()] = new DeviceFunc(name_, hmod);
  }
  *hfunc = dFunc_[ihipGetDevice()]->asHipFunction();

  return hipSuccess;
}

hipError_t Function::getStatFunc(hipFunction_t* hfunc, int deviceId) {
  guarantee(modules_ != nullptr, "Module not initialized");

  hipModule_t hmod = nullptr;
  IHIP_RETURN_ONFAIL((*modules_)->BuildProgram(deviceId));
  IHIP_RETURN_ONFAIL((*modules_)->GetModule(deviceId, &hmod));

  if (dFunc_[deviceId] == nullptr) {
    dFunc_[deviceId] = new DeviceFunc(name_, hmod);
  }
  *hfunc = dFunc_[deviceId]->asHipFunction();

  return hipSuccess;
}

hipError_t Function::getStatFuncAttr(hipFuncAttributes* func_attr, int deviceId) {
  guarantee((modules_ != nullptr), "Module not initialized");

  hipModule_t hmod = nullptr;
  IHIP_RETURN_ONFAIL((*modules_)->BuildProgram(deviceId));
  IHIP_RETURN_ONFAIL((*modules_)->GetModule(deviceId, &hmod));

  if (dFunc_[deviceId] == nullptr) {
    dFunc_[deviceId] = new DeviceFunc(name_, hmod);
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
         FatBinaryInfo** modules) : name_(name), dVarKind_(dVarKind), size_(size),
         type_(type), norm_(norm), modules_(modules) {
  dVar_.resize(g_devices.size());
}

Var::Var(std::string name, DeviceVarKind dVarKind, void *pointer, size_t size,
         unsigned align, FatBinaryInfo** modules) : name_(name), dVarKind_(dVarKind),
         size_(size), modules_(modules), managedVarPtr_(pointer), align_(align) {
  dVar_.resize(g_devices.size());
}

Var::~Var() {
  for (auto& elem : dVar_) {
    delete elem;
  }
  modules_ = nullptr;
}

hipError_t Var::getDeviceVar(DeviceVar** dvar, int deviceId, hipModule_t hmod) {
  guarantee((deviceId >= 0), "Invalid DeviceId, less than zero");
  guarantee((static_cast<size_t>(deviceId) < g_devices.size()),
            "Invalid DeviceId, greater than no of code objects");
  guarantee((dVar_.size() == g_devices.size()),
             "Device Var not initialized to size");

  if (dVar_[deviceId] == nullptr) {
    dVar_[deviceId] = new DeviceVar(name_, hmod);
  }

  *dvar = dVar_[deviceId];
  return hipSuccess;
}

hipError_t Var::getStatDeviceVar(DeviceVar** dvar, int deviceId) {
  guarantee((deviceId >= 0) , "Invalid DeviceId, less than zero");
  guarantee((static_cast<size_t>(deviceId) < g_devices.size()),
            "Invalid DeviceId, greater than no of code objects");
  if (dVar_[deviceId] == nullptr) {
    hipModule_t hmod = nullptr;
    IHIP_RETURN_ONFAIL((*modules_)->BuildProgram(deviceId));
    IHIP_RETURN_ONFAIL((*modules_)->GetModule(deviceId, &hmod));
    dVar_[deviceId] = new DeviceVar(name_, hmod);
  }
  *dvar = dVar_[deviceId];
  return hipSuccess;
}

}; //namespace: hip

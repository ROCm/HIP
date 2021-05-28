#ifndef HIP_GLOBAL_HPP
#define HIP_GLOBAL_HPP

#include <vector>
#include <string>

#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_fatbin.hpp"
#include "platform/program.hpp"

namespace hip {

//Forward Declaration
class CodeObject;

//Device Structures
class DeviceVar {
public:
  DeviceVar(std::string name, hipModule_t hmod);
  ~DeviceVar();

  //Accessors for device ptr and size, populated during constructor.
  hipDeviceptr_t device_ptr() const { return device_ptr_; }
  size_t size() const { return size_; }
  std::string name() const { return name_; }
  void* shadowVptr;

private:
  std::string name_;           //Name of the var
  amd::Memory* amd_mem_obj_;   //amd_mem_obj abstraction
  hipDeviceptr_t device_ptr_;  //Device Pointer
  size_t size_;                //Size of the var
};

class DeviceFunc {
public:
  DeviceFunc(std::string name, hipModule_t hmod);
  ~DeviceFunc();

  amd::Monitor dflock_;

  //Converts DeviceFunc to hipFunction_t(used by app) and vice versa.
  hipFunction_t asHipFunction() { return reinterpret_cast<hipFunction_t>(this); }
  static DeviceFunc* asFunction(hipFunction_t f) { return reinterpret_cast<DeviceFunc*>(f); }

  //Accessor for kernel_ and name_ populated during constructor.
  std::string name() const { return name_; }
  amd::Kernel* kernel() const { return kernel_; }

private:
  std::string name_;        //name of the func(not unique identifier)
  amd::Kernel* kernel_;     //Kernel ptr referencing to ROCclr Symbol
};

//Abstract Structures
class Function {
public:
  Function(std::string name, FatBinaryInfo** modules=nullptr);
  ~Function();

  //Return DeviceFunc for this this dynamically loaded module
  hipError_t getDynFunc(hipFunction_t* hfunc, hipModule_t hmod);

  //Return Device Func & attr . Generate/build if not already done so.
  hipError_t getStatFunc(hipFunction_t *hfunc, int deviceId);
  hipError_t getStatFuncAttr(hipFuncAttributes* func_attr, int deviceId);
  void resize_dFunc(size_t size) { dFunc_.resize(size); }
  FatBinaryInfo** moduleInfo() { return modules_; };

private:
  std::vector<DeviceFunc*> dFunc_;  //DeviceFuncObj per Device
  std::string name_;                //name of the func(not unique identifier)
  FatBinaryInfo** modules_;      // static module where it is referenced
};

class Var {
public:
  //Types of variable
  enum DeviceVarKind {
    DVK_Variable = 0,
    DVK_Surface,
    DVK_Texture,
    DVK_Managed
  };

  Var(std::string name, DeviceVarKind dVarKind, size_t size, int type, int norm,
      FatBinaryInfo** modules = nullptr);

  Var(std::string name, DeviceVarKind dVarKind, void *pointer, size_t size, unsigned align,
      FatBinaryInfo** modules = nullptr);

  ~Var();

  //Return DeviceVar for this dynamically loaded module
  hipError_t getDeviceVar(DeviceVar** dvar, int deviceId, hipModule_t hmod);

  //Return DeviceVar for module Generate/build if not already done so.
  hipError_t getStatDeviceVar(DeviceVar** dvar, int deviceId);
  void resize_dVar(size_t size) { dVar_.resize(size); }

  FatBinaryInfo** moduleInfo() { return modules_; };
  DeviceVarKind getVarKind() const { return dVarKind_; }
  size_t getSize() const { return size_; }
  
  void* getManagedVarPtr() { return managedVarPtr_; };
  void setManagedVarInfo(void* pointer, size_t size) {
    managedVarPtr_ = pointer;
    size_ = size;
    dVarKind_ = DVK_Managed;
  }
private:
  std::vector<DeviceVar*> dVar_;   // DeviceVarObj per Device
  std::string name_;               // Variable name (not unique identifier)
  DeviceVarKind dVarKind_;         // Variable kind
  size_t size_;                    // Size of the variable
  int type_;                       // Type(Textures/Surfaces only)
  int norm_;                       // Type(Textures/Surfaces only)
  FatBinaryInfo** modules_;        // static module where it is referenced

  void *managedVarPtr_;            // Managed memory pointer with size_ & align_
  unsigned int align_;             // Managed memory alignment
};

}; //namespace: hip
#endif /* HIP_GLOBAL_HPP */

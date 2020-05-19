#ifndef HIP_CODE_OBJECT_HPP
#define HIP_CODE_OBJECT_HPP

#include "hip_global.hpp"

#include <unordered_map>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_internal.hpp"
#include "device/device.hpp"
#include "platform/program.hpp"

//Forward Declaration for friend usage
class PlatformState;

namespace hip {

//Code Object base class
class CodeObject {
public:
  virtual ~CodeObject() {}

  //Functions to add_dev_prog and build
  static hipError_t add_program(int deviceId, hipModule_t hmod, const void* binary_ptr,
                                size_t binary_size);
  static hipError_t build_module(hipModule_t hmod, const std::vector<amd::Device*>& devices);

  //ClangOFFLOADBundle info
  #define CLANG_OFFLOAD_BUNDLER_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"
  #define HIP_AMDGCN_AMDHSA_TRIPLE "hip-amdgcn-amd-amdhsa"
  #define HCC_AMDGCN_AMDHSA_TRIPLE "hcc-amdgcn-amd-amdhsa-"

  //Clang Offload bundler description & Header
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

protected:
  CodeObject() {}
  //Given an ptr to image or file, extracts to code object
  //for corresponding devices
  hipError_t extractCodeObjectFromFatBinary(const void*,
             const std::vector<const char*>&,
             std::vector<std::pair<const void*, size_t>>&);

  uint64_t ElfSize(const void* emi);

private:
  bool isCompatibleCodeObject(const std::string& codeobj_target_id,
                                     const char* device_name);

  friend const std::vector<hipModule_t>& modules();
};

//Dynamic Code Object
class DynCO : public CodeObject {
  amd::Monitor dclock_{"Guards Static Code object", true};

public:
  DynCO();
  virtual ~DynCO();

  //LoadsCodeObject and its data
  hipError_t loadCodeObject(const char* fname, const void* image=nullptr);
  hipModule_t module() { return reinterpret_cast<hipModule_t>(as_cl(program_)); };

  //Gets GlobalVar/Functions from a dynamically loaded code object
  hipError_t getDynFunc(hipFunction_t* hfunc, std::string func_name);
  hipError_t getDeviceVar(DeviceVar** dvar, std::string var_name, int deviceId);

private:
  amd::Program* program_;

  //Maps for vars/funcs, could be keyed in with std::string name
  std::unordered_map<std::string, Function*> functions_;
  std::unordered_map<std::string, Var*> vars_;

  //Load Code Object Data(Vars/UndefinedVars/Funcs)
  hipError_t loadCodeObjectData(const void* mmap_ptr, size_t mmap_size);

  //Populate Global Vars/Funcs from an code object(@ module_load)
  hipError_t populateDynGlobalFuncs();
  hipError_t populateDynGlobalVars();
};

//Static Code Object
class StatCO: public CodeObject {
  amd::Monitor sclock_{"Guards Static Code object", true};
public:
  StatCO();
  virtual ~StatCO();

  //Add/Remove/Digest Fat Binaries passed to us from "__hipRegisterFatBinary"
  FatBinaryInfoType* addFatBinary(const void* data, bool initialized);
  hipError_t removeFatBinary(FatBinaryInfoType* module);
  hipError_t digestFatBinary(const void* data, FatBinaryInfoType& programs);

  //Register vars/funcs given to use from __hipRegister[Var/Func]
  hipError_t registerStatFunction(const void* hostFunction, Function* func);
  hipError_t registerStatGlobalVar(const void* hostVar, Var* var);

  //Retrive Vars/Funcs for a given hostSidePtr(const void*), unless stated otherwise.
  hipError_t getStatFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId);
  hipError_t getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction, int deviceId);
  hipError_t getStatGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                              size_t* size_ptr);
  hipError_t getStatGlobalVarByName(std::string hostVar, int deviceId, hipModule_t hmod,
                                    hipDeviceptr_t* dev_ptr, size_t* size_ptr);

private:
  friend class ::PlatformState;
  //Populated during __hipRegisterFatBinary
  std::unordered_map<const void*, FatBinaryInfoType> modules_;
  //Populated during __hipRegisterFuncs
  std::unordered_map<const void*, Function*> functions_;
  //Populated during __hipRegisterVars
  std::unordered_map<const void*, Var*> vars_;
};

}; //namespace: hip

#endif /* HIP_CODE_OBJECT_HPP */

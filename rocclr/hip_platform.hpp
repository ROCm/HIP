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
#pragma once

#include "hip_internal.hpp"
#include "hip_fatbin.hpp"
#include "device/device.hpp"
#include "hip_code_object.hpp"

namespace hip_impl {

hipError_t ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* maxBlocksPerCU, int* numBlocksPerGrid, int* bestBlockSize,
    const amd::Device& device, hipFunction_t func, int  blockSize,
    size_t dynamicSMemSize, bool bCalcPotentialBlkSz);
} /* namespace hip_impl*/

class PlatformState {
  amd::Monitor lock_{"Guards PlatformState globals", true};

  /* Singleton object */
  static PlatformState* platform_;
  PlatformState() {}
  ~PlatformState() {}

public:
  void init();

  //Dynamic Code Objects functions
  hipError_t loadModule(hipModule_t* module, const char* fname, const void* image = nullptr);
  hipError_t unloadModule(hipModule_t hmod);

  hipError_t getDynFunc(hipFunction_t *hfunc, hipModule_t hmod, const char* func_name);
  hipError_t getDynGlobalVar(const char* hostVar, hipModule_t hmod, hipDeviceptr_t* dev_ptr,
                             size_t* size_ptr);
  hipError_t getDynTexRef(const char* hostVar, hipModule_t hmod, textureReference** texRef);

  hipError_t registerTexRef(textureReference* texRef, hipModule_t hmod, std::string name);
  hipError_t getDynTexGlobalVar(textureReference* texRef, hipDeviceptr_t* dev_ptr,
                                size_t* size_ptr);

  /* Singleton instance */
  static PlatformState& instance() {
    if (platform_ == nullptr) {
       // __hipRegisterFatBinary() will call this when app starts, thus
       // there is no multiple entry issue here.
       platform_ =  new PlatformState();
    }
    return *platform_;
  }

  //Static Code Objects functions
  hip::FatBinaryInfo** addFatBinary(const void* data);
  hipError_t removeFatBinary(hip::FatBinaryInfo** module);
  hipError_t digestFatBinary(const void* data, hip::FatBinaryInfo*& programs);

  hipError_t registerStatFunction(const void* hostFunction, hip::Function* func);
  hipError_t registerStatGlobalVar(const void* hostVar, hip::Var* var);
  hipError_t registerStatManagedVar(hip::Var* var);


  hipError_t getStatFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId);
  hipError_t getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction, int deviceId);
  hipError_t getStatGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                              size_t* size_ptr);

  hipError_t initStatManagedVarDevicePtr(int deviceId);

  //Exec Functions
  void setupArgument(const void *arg, size_t size, size_t offset);
  void configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);
  void popExec(ihipExec_t& exec);

private:
  //Dynamic Code Object map, keyin module to get the corresponding object
  std::unordered_map<hipModule_t, hip::DynCO*> dynCO_map_;
  hip::StatCO statCO_; //Static Code object var
  bool initialized_{false};
  std::unordered_map<textureReference*, std::pair<hipModule_t, std::string>> texRef_map_;
};

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
#include <libelf.h>
#include <fstream>

#include "hip_internal.hpp"
#include "platform/program.hpp"
#include "hip_event.hpp"
#include "hip_platform.hpp"

hipError_t ihipModuleLoadData(hipModule_t* module, const void* mmap_ptr, size_t mmap_size);

const std::string& FunctionName(const hipFunction_t f)
{
  return hip::Function::asFunction(f)->function_->name();
}

static uint64_t ElfSize(const void *emi)
{
  const Elf64_Ehdr *ehdr = (const Elf64_Ehdr*)emi;
  const Elf64_Shdr *shdr = (const Elf64_Shdr*)((char*)emi + ehdr->e_shoff);

  uint64_t max_offset = ehdr->e_shoff;
  uint64_t total_size = max_offset + ehdr->e_shentsize * ehdr->e_shnum;

  for (uint16_t i=0; i < ehdr->e_shnum; ++i){
    uint64_t cur_offset = static_cast<uint64_t>(shdr[i].sh_offset);
    if (max_offset < cur_offset) {
      max_offset = cur_offset;
      total_size = max_offset;
      if(SHT_NOBITS != shdr[i].sh_type) {
        total_size += static_cast<uint64_t>(shdr[i].sh_size);
      }
    }
  }
  return total_size;
}

hipError_t hipModuleLoad(hipModule_t* module, const char* fname)
{
  HIP_INIT_API(hipModuleLoad, module, fname);

  const void* mmap_ptr = nullptr;
  size_t mmap_size = 0;

  if (!fname) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (!amd::Os::MemoryMapFile(fname, &mmap_ptr, &mmap_size)) {
    HIP_RETURN(hipErrorFileNotFound);
  }

  HIP_RETURN(ihipModuleLoadData(module, mmap_ptr, mmap_size));
}

bool ihipModuleUnregisterGlobal(hipModule_t hmod) {
  std::vector< std::pair<hipModule_t, bool> >* modules =
    PlatformState::instance().unregisterVar(hmod);
  if (modules != nullptr) {
    delete modules;
  }
  return true;
}

hipError_t hipModuleUnload(hipModule_t hmod)
{
  HIP_INIT_API(hipModuleUnload, hmod);

  if (hmod == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));

  if(!PlatformState::instance().unregisterFunc(hmod)) {
    DevLogPrintfError("Cannot unregister module: 0x%x \n", hmod);
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  if(!ihipModuleUnregisterGlobal(hmod)) {
    DevLogPrintfError("Cannot unregister Global vars for module: 0x%x \n", hmod);
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  program->release();

  HIP_RETURN(hipSuccess);
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image)
{
  HIP_INIT_API(hipModuleLoadData, module, image);

  HIP_RETURN(ihipModuleLoadData(module, image, 0));
}

hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image,
                               unsigned int numOptions, hipJitOption* options,
                               void** optionsValues)
{
  /* TODO: Pass options to Program */
  HIP_INIT_API(hipModuleLoadDataEx, module, image);

  HIP_RETURN(ihipModuleLoadData(module, image, 0));
}

extern hipError_t __hipExtractCodeObjectFromFatBinary(const void* data,
                                                const std::vector<const char*>& devices,
                                                std::vector<std::pair<const void*, size_t>>& code_objs);

inline bool ihipModuleRegisterUndefined(amd::Program* program, hipModule_t* module) {

  std::vector<std::string> undef_vars;
  device::Program* dev_program
    = program->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  if (!dev_program->getUndefinedVarFromCodeObj(&undef_vars)) {
    DevLogPrintfError("Could not get undefined Variables for Module: 0x%x \n", *module);
    return false;
  }

  for (auto it = undef_vars.begin(); it != undef_vars.end(); ++it) {
    auto modules = new std::vector<std::pair<hipModule_t, bool> >(g_devices.size());
    for (size_t dev = 0; dev < g_devices.size(); ++dev) {
      modules->at(dev) = std::make_pair(*module, true);
    }

    texture<float, hipTextureType1D, hipReadModeElementType>* tex_hptr
      = new texture<float, hipTextureType1D, hipReadModeElementType>();
    memset(tex_hptr, 0x00, sizeof(texture<float, hipTextureType1D, hipReadModeElementType>));

    PlatformState::DeviceVar dvar{PlatformState::DVK_Variable,
                                  reinterpret_cast<char*>(tex_hptr),
                                  it->c_str(),
                                  sizeof(*tex_hptr),
                                  modules,
                                  std::vector<PlatformState::RegisteredVar>{g_devices.size()},
                                  true,
                                  /*type*/ 0,
                                  /*norm*/ 0};
    PlatformState::instance().registerVar(it->c_str(), dvar);
  }

  return true;
}

inline bool ihipModuleRegisterFunc(amd::Program* program, hipModule_t* module) {

  std::vector<std::string> func_names;
  device::Program* dev_program
    = program->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  // Get all the global func names from COMGR
  if (!dev_program->getGlobalFuncFromCodeObj(&func_names)) {
    DevLogPrintfError("Could not get Global Funcs from Code Obj for Module: 0x%x \n", *module);
    return false;
  }

  return PlatformState::instance().registerModFuncs(func_names, module);
}


inline bool ihipModuleRegisterGlobal(amd::Program* program, hipModule_t* module) {

  size_t var_size = 0;
  hipDeviceptr_t device_ptr = nullptr;
  std::vector<std::string> var_names;

  device::Program* dev_program
    = program->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  if (!dev_program->getGlobalVarFromCodeObj(&var_names)) {
    DevLogPrintfError("Could not get Global vars from Code Obj for Module: 0x%x \n", *module);
    return false;
  }

  for (auto it = var_names.begin(); it != var_names.end(); ++it) {
    auto modules = new std::vector<std::pair<hipModule_t, bool> >(g_devices.size());
    for (size_t dev = 0; dev < g_devices.size(); ++dev) {
      modules->at(dev) = std::make_pair(*module, true);
    }

    PlatformState::DeviceVar dvar{PlatformState::DVK_Variable,
                                  nullptr,
                                  it->c_str(),
                                  0,
                                  modules,
                                  std::vector<PlatformState::RegisteredVar>{g_devices.size()},
                                  false,
                                  /*type*/ 0,
                                  /*norm*/ 0};
    PlatformState::instance().registerVar(it->c_str(), dvar);
  }

  return true;
}

hipError_t ihipModuleLoadData(hipModule_t* module, const void* mmap_ptr, size_t mmap_size)
{
  /* initialize image it to the mmap_ptr, if this is of no_clang_offload bundle then they directly pass the image */
  const void* image = mmap_ptr;
  std::vector<std::pair<const void*, size_t>> code_objs;
  hipError_t code_obj_err = __hipExtractCodeObjectFromFatBinary(mmap_ptr,
                              {hip::getCurrentDevice()->devices()[0]->info().name_}, code_objs);
  if (code_obj_err == hipSuccess) {
    image = code_objs[0].first;
  } else if(code_obj_err == hipErrorNoBinaryForGpu) {
     return code_obj_err;
  }

  amd::Program* program = new amd::Program(*hip::getCurrentDevice()->asContext(),
                                           amd::Program::Language::Binary, mmap_ptr, mmap_size);
  if (program == NULL) {
    return hipErrorOutOfMemory;
  }

  program->setVarInfoCallBack(&getSvarInfo);

  if (CL_SUCCESS != program->addDeviceProgram(*hip::getCurrentDevice()->devices()[0], image,
                                              ElfSize(image), false)) {
    return hipErrorInvalidKernelFile;
  }

  *module = reinterpret_cast<hipModule_t>(as_cl(program));

  if (!ihipModuleRegisterGlobal(program, module)) {
    return hipErrorSharedObjectSymbolNotFound;
  }

  if (!ihipModuleRegisterUndefined(program, module)) {
    return hipErrorSharedObjectSymbolNotFound;
  }

  if(CL_SUCCESS != program->build(hip::getCurrentDevice()->devices(), nullptr, nullptr, nullptr)) {
    return hipErrorSharedObjectInitFailed;
  }

  if (!ihipModuleRegisterFunc(program, module)) {
    return hipErrorSharedObjectSymbolNotFound;
  }

  return hipSuccess;
}

hipError_t hipModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod, const char *name)
{
  HIP_INIT_API(hipModuleGetFunction, hfunc, hmod, name);

  if (!PlatformState::instance().findModFunc(hfunc, hmod, name)) {
    DevLogPrintfError("Cannot find the function: %s for module: 0x%x \n",
                      name, hmod);
    HIP_RETURN(hipErrorNotFound);
  }
  HIP_RETURN(hipSuccess);
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod, const char* name)
{
  HIP_INIT_API(hipModuleGetGlobal, dptr, bytes, hmod, name);

  /* Get address and size for the global symbol */
  if (!PlatformState::instance().getGlobalVar(name, ihipGetDevice(), hmod,
                                              dptr, bytes)) {
    DevLogPrintfError("Cannot find global Var: %s for module: 0x%x at device: %d \n",
                       name, hmod, ihipGetDevice());
    HIP_RETURN(hipErrorNotFound);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc) {
  HIP_INIT_API(hipFuncGetAttribute, value, attrib, hfunc);

  if ((value == nullptr) || (hfunc == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hip::Function* function = hip::Function::asFunction(hfunc);
  if (function == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  amd::Kernel* kernel = function->function_;
  if (kernel == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  const device::Kernel::WorkGroupInfo* wrkGrpInfo
    = kernel->getDeviceKernel(*(hip::getCurrentDevice()->devices()[0]))->workGroupInfo();
  if (wrkGrpInfo == nullptr) {
    HIP_RETURN(hipErrorMissingConfiguration);
  }

  switch(attrib) {
    case HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
      *value = static_cast<int>(wrkGrpInfo->localMemSize_
                      - wrkGrpInfo->privateMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
      *value = static_cast<int>(wrkGrpInfo->wavefrontPerSIMD_
                * wrkGrpInfo->wavefrontSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
      *value = 0;
      break;
    case HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
      *value = static_cast<int>(wrkGrpInfo->localMemSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_NUM_REGS:
      *value = static_cast<int>(wrkGrpInfo->availableGPRs_);
      break;
    case HIP_FUNC_ATTRIBUTE_PTX_VERSION:
      *value = 30; // Defaults to 3.0 as HCC
      break;
    case HIP_FUNC_ATTRIBUTE_BINARY_VERSION:
      *value = static_cast<int>(kernel->signature().version());
      break;
    case HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA:
      *value = 0;
      break;
    case HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
      *value = static_cast<int>(wrkGrpInfo->availableLDSSize_);
      break;
    case HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
      *value = 0;
      break;
     default:
       HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func)
{
  HIP_INIT_API(hipFuncGetAttributes, attr, func);

  if (!PlatformState::instance().getFuncAttr(func, attr)) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t ihipModuleLaunchKernel(hipFunction_t f,
                                 uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                 uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                 uint32_t sharedMemBytes, hipStream_t hStream,
                                 void **kernelParams, void **extra,
                                 hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags = 0,
                                 uint32_t params = 0, uint32_t gridId = 0, uint32_t numGrids = 0,
                                 uint64_t prevGridSum = 0, uint64_t allGridSum = 0, uint32_t firstDevice = 0) {
  HIP_INIT_API(ihipModuleLaunchKernel, f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
    sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags, params);

  hip::Function* function = hip::Function::asFunction(f);
  amd::Kernel* kernel = function->function_;

  amd::ScopedLock lock(function->lock_);

  hip::Event* eStart = reinterpret_cast<hip::Event*>(startEvent);
  hip::Event* eStop = reinterpret_cast<hip::Event*>(stopEvent);
  amd::HostQueue* queue = hip::getQueue(hStream);
  const amd::Device& device = queue->vdev()->device();

  // Make sure dispatch doesn't exceed max workgroup size limit
  if (blockDimX * blockDimY * blockDimZ > device.info().maxWorkGroupSize_) {
    return hipErrorInvalidConfiguration;
  }

  if (params & amd::NDRangeKernelCommand::CooperativeGroups) {
    if (!device.info().cooperativeGroups_) {
      return hipErrorLaunchFailure;
    }
    int num_blocks = 0;
    int num_grids = 0;
    int block_size = blockDimX * blockDimY * blockDimZ;
    hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, &num_grids, device, f, block_size, sharedMemBytes, true);
    if (((gridDimX * gridDimY * gridDimZ) / block_size) > unsigned(num_grids)) {
      return hipErrorCooperativeLaunchTooLarge;
    }
  }
  if (params & amd::NDRangeKernelCommand::CooperativeMultiDeviceGroups) {
    if (!device.info().cooperativeMultiDeviceGroups_) {
      return hipErrorLaunchFailure;
    }
  }
  if (!queue) {
    return hipErrorOutOfMemory;
  }

  size_t globalWorkOffset[3] = {0};
  size_t globalWorkSize[3] = { gridDimX, gridDimY, gridDimZ };
  size_t localWorkSize[3] = { blockDimX, blockDimY, blockDimZ };
  amd::NDRangeContainer ndrange(3, globalWorkOffset, globalWorkSize, localWorkSize);
  amd::Command::EventWaitList waitList;

  address kernargs = nullptr;

  // 'extra' is a struct that contains the following info: {
  //   HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
  //   HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargs_size,
  //   HIP_LAUNCH_PARAM_END }
  if (extra != nullptr) {
    if (extra[0] != HIP_LAUNCH_PARAM_BUFFER_POINTER ||
        extra[2] != HIP_LAUNCH_PARAM_BUFFER_SIZE || extra[4] != HIP_LAUNCH_PARAM_END) {
      return hipErrorNotInitialized;
    }
    kernargs = reinterpret_cast<address>(extra[1]);
  }

    const amd::KernelSignature& signature = kernel->signature();
    for (size_t i = 0; i < signature.numParameters(); ++i) {
      const amd::KernelParameterDescriptor& desc = signature.at(i);
    if (kernelParams == nullptr) {
      assert(kernargs != nullptr);
      kernel->parameters().set(i, desc.size_, kernargs + desc.offset_,
                               desc.type_ == T_POINTER/*svmBound*/);
    } else {
      assert(extra == nullptr);
      kernel->parameters().set(i, desc.size_, kernelParams[i], desc.type_ == T_POINTER/*svmBound*/);
    }
  }

  amd::NDRangeKernelCommand* command = new amd::NDRangeKernelCommand(
    *queue, waitList, *kernel, ndrange, sharedMemBytes,
    params, gridId, numGrids, prevGridSum, allGridSum, firstDevice);
  if (!command) {
    return hipErrorOutOfMemory;
  }

  // Capture the kernel arguments
  if (CL_SUCCESS != command->captureAndValidate()) {
    delete command;
    return hipErrorOutOfMemory;
  }

  command->enqueue();

  if(startEvent != nullptr) {
    eStart->addMarker(queue, command);
    command->retain();
  }
  if(stopEvent != nullptr) {
    eStop->addMarker(queue, command);
    command->retain();
  }
  command->release();

  return hipSuccess;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                 uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                 uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                 uint32_t sharedMemBytes, hipStream_t hStream,
                                 void **kernelParams, void **extra)
{
  HIP_INIT_API(hipModuleLaunchKernel, f, gridDimX, gridDimY, gridDimZ,
               blockDimX, blockDimY, blockDimZ,
               sharedMemBytes, hStream,
               kernelParams, extra);

  HIP_RETURN(ihipModuleLaunchKernel(f, gridDimX * blockDimX, gridDimY * blockDimY, gridDimZ * blockDimZ,
                                blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, hStream, kernelParams, extra, nullptr, nullptr));
}

hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags)
{
  HIP_INIT_API(hipExtModuleLaunchKernel, f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
               localWorkSizeX, localWorkSizeY, localWorkSizeZ,
               sharedMemBytes, hStream,
               kernelParams, extra, startEvent, stopEvent, flags);

  HIP_RETURN(ihipModuleLaunchKernel(f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
      localWorkSizeZ, sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent, flags));
}



hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t gridDimX,
                                    uint32_t gridDimY, uint32_t gridDimZ,
                                    uint32_t blockDimX, uint32_t blockDimY,
                                    uint32_t blockDimZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent,
                                    hipEvent_t stopEvent)
{
  HIP_INIT_API(hipHccModuleLaunchKernel, f, gridDimX, gridDimY, gridDimZ,
               blockDimX, blockDimY, blockDimZ,
               sharedMemBytes, hStream,
               kernelParams, extra, startEvent, stopEvent);

  HIP_RETURN(ihipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent));
}

hipError_t hipModuleLaunchKernelExt(hipFunction_t f, uint32_t gridDimX,
                                    uint32_t gridDimY, uint32_t gridDimZ,
                                    uint32_t blockDimX, uint32_t blockDimY,
                                    uint32_t blockDimZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent,
                                    hipEvent_t stopEvent)
{
  HIP_INIT_API(hipModuleLaunchKernelExt, f, gridDimX, gridDimY, gridDimZ,
               blockDimX, blockDimY, blockDimZ,
               sharedMemBytes, hStream,
               kernelParams, extra, startEvent, stopEvent);

  HIP_RETURN(ihipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent));
}

hipError_t hipLaunchCooperativeKernel(const void* f,
                                      dim3 gridDim, dim3 blockDim,
                                      void **kernelParams, uint32_t sharedMemBytes, hipStream_t hStream)
{
  HIP_INIT_API(hipLaunchCooperativeKernel, f, gridDim, blockDim,
               sharedMemBytes, hStream);

  int deviceId = ihipGetDevice();
  hipFunction_t func = PlatformState::instance().getFunc(f, deviceId);
  if (func == nullptr) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  HIP_RETURN(ihipModuleLaunchKernel(func, gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z,
                                blockDim.x, blockDim.y, blockDim.z,
                                sharedMemBytes, hStream, kernelParams, nullptr, nullptr, nullptr, 0,
                                amd::NDRangeKernelCommand::CooperativeGroups));
}

hipError_t ihipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                  int numDevices, unsigned int flags, uint32_t extFlags)
{
  int numActiveGPUs = 0;
  ihipDeviceGetCount(&numActiveGPUs);

  if ((numDevices > numActiveGPUs) || (launchParamsList == nullptr)) {
    return hipErrorInvalidValue;
  }

  hipError_t result = hipErrorUnknown;
  uint64_t allGridSize = 0;
  std::vector<const amd::Device*> mgpu_list(numDevices);

  for (int i = 0; i < numDevices; ++i) {
    const hipLaunchParams& launch = launchParamsList[i];
    allGridSize += launch.gridDim.x * launch.gridDim.y * launch.gridDim.z;

    // Make sure block dimensions are valid
    if (0 == launch.blockDim.x * launch.blockDim.y * launch.blockDim.z) {
      return hipErrorInvalidConfiguration;
    }
    if (launch.stream != nullptr) {
      // Validate devices to make sure it dosn't have duplicates
      amd::HostQueue* queue = reinterpret_cast<hip::Stream*>(launch.stream)->asHostQueue();
      auto device = &queue->vdev()->device();
      for (int j = 0; j < numDevices; ++j) {
        if (mgpu_list[j] == device) {
          return hipErrorInvalidDevice;
        }
      }
      mgpu_list[i] = device;
    } else {
      return hipErrorInvalidResourceHandle;
    }
  }
  uint64_t prevGridSize = 0;
  uint32_t firstDevice = 0;

  // Sync the execution streams on all devices 
  if ((flags & hipCooperativeLaunchMultiDeviceNoPreSync) == 0) {
    for (int i = 0; i < numDevices; ++i) {
      amd::HostQueue* queue =
        reinterpret_cast<hip::Stream*>(launchParamsList[i].stream)->asHostQueue();
      queue->finish();
    }
  }

  for (int i = 0; i < numDevices; ++i) {
    const hipLaunchParams& launch = launchParamsList[i];
    amd::HostQueue* queue = reinterpret_cast<hip::Stream*>(launch.stream)->asHostQueue();
    hipFunction_t func = nullptr;
    // The order of devices in the launch may not match the order in the global array
    for (size_t dev = 0; dev < g_devices.size(); ++dev) {
      // Find the matching device and request the kernel function
      if (&queue->vdev()->device() == g_devices[dev]->devices()[0]) {
        func = PlatformState::instance().getFunc(launch.func, dev);
        // Save ROCclr index of the first device in the launch
        if (i == 0) {
          firstDevice = queue->vdev()->device().index();
        }
        break;
      }
    }
    if (func == nullptr) {
      result = hipErrorInvalidDeviceFunction;
      HIP_RETURN(result);
    }

    result = ihipModuleLaunchKernel(func,
      launch.gridDim.x * launch.blockDim.x,
      launch.gridDim.y * launch.blockDim.y,
      launch.gridDim.z * launch.blockDim.z,
      launch.blockDim.x, launch.blockDim.y, launch.blockDim.z,
      launch.sharedMem, launch.stream, launch.args, nullptr, nullptr, nullptr,
      flags, extFlags, i, numDevices, prevGridSize, allGridSize, firstDevice);
    if (result != hipSuccess) {
      break;
    }
    prevGridSize += launch.gridDim.x * launch.gridDim.y * launch.gridDim.z;
  }

  // Sync the execution streams on all devices 
  if ((flags & hipCooperativeLaunchMultiDeviceNoPostSync) == 0) {
    for (int i = 0; i < numDevices; ++i) {
      amd::HostQueue* queue =
        reinterpret_cast<hip::Stream*>(launchParamsList[i].stream)->asHostQueue();
      queue->finish();
    }
  }

  return result;
}

hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                 int numDevices, unsigned int flags)
{
  HIP_INIT_API(hipLaunchCooperativeKernelMultiDevice, launchParamsList, numDevices, flags);

  return ihipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags,
                                                (amd::NDRangeKernelCommand::CooperativeGroups |
                                                 amd::NDRangeKernelCommand::CooperativeMultiDeviceGroups));
}

hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                              int numDevices, unsigned int flags) {
  HIP_INIT_API(hipExtLaunchMultiKernelMultiDevice, launchParamsList, numDevices, flags);

  return ihipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags, 0);
}

hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name) {
  HIP_INIT_API(hipModuleGetTexRef, texRef, hmod, name);

  /* input args check */
  if ((texRef == nullptr) || (name == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

   /* Get address and size for the global symbol */
  if (!PlatformState::instance().getTexRef(name, hmod, texRef)) {
    DevLogPrintfError("Cannot get texRef for name: %s at module:0x%x \n",
                      name, hmod);
    HIP_RETURN(hipErrorNotFound);
  }

  // Texture references created by HIP driver API
  // have the default read mode set to normalized float.
  (*texRef)->readMode = hipReadModeNormalizedFloat;

  HIP_RETURN(hipSuccess);
}

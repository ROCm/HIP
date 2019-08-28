/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>
#include <libelf.h>
#include <fstream>

#include "hip_internal.hpp"
#include "platform/program.hpp"
#include "hip_event.hpp"

hipError_t ihipModuleLoadData(hipModule_t *module, const void *image);

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
  HIP_INIT_API(module, fname);

  if (!fname) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  std::ifstream file(fname, std::ios::binary);

  if (!file.is_open()) {
    HIP_RETURN(hipErrorFileNotFound);
  }

  std::vector<char> tmp{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};

  HIP_RETURN(ihipModuleLoadData(module, tmp.data()));
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
  HIP_INIT_API(hmod);

  if (hmod == nullptr) {
    HIP_RETURN(hipErrorUnknown);
  }

  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));

  if(!ihipModuleUnregisterGlobal(hmod)) {
    HIP_RETURN(hipErrorUnknown);
  }

  program->release();

  HIP_RETURN(hipSuccess);
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image)
{
  HIP_INIT_API(module, image);

  HIP_RETURN(ihipModuleLoadData(module, image));
}

extern bool __hipExtractCodeObjectFromFatBinary(const void* data,
                                                const std::vector<const char*>& devices,
                                                std::vector<std::pair<const void*, size_t>>& code_objs);

bool ihipModuleRegisterGlobal(amd::Program* program, hipModule_t* module) {

  size_t var_size = 0;
  hipDeviceptr_t device_ptr = nullptr;
  std::vector<std::string> var_names;

  device::Program* dev_program
    = program->getDeviceProgram(*hip::getCurrentContext()->devices()[0]);

  if (!dev_program->getGlobalVarFromCodeObj(&var_names)) {
    return false;
  }

  for (auto it = var_names.begin(); it != var_names.end(); ++it) {
    auto modules = new std::vector<std::pair<hipModule_t, bool> >{g_devices.size()};
    for (size_t dev = 0; dev < g_devices.size(); ++dev) {
      modules->at(dev) = std::make_pair(*module, false);
    }

    PlatformState::DeviceVar dvar{nullptr, it->c_str(), 0, modules,
      std::vector<PlatformState::RegisteredVar>{ g_devices.size()} };
    PlatformState::instance().registerVar(it->c_str(), dvar);
  }

  return true;
}

hipError_t ihipModuleLoadData(hipModule_t *module, const void *image)
{
  std::vector<std::pair<const void*, size_t>> code_objs;
  if (__hipExtractCodeObjectFromFatBinary(image, {hip::getCurrentContext()->devices()[0]->info().name_}, code_objs))
    image = code_objs[0].first;

  amd::Program* program = new amd::Program(*hip::getCurrentContext());
  if (program == NULL) {
    return hipErrorOutOfMemory;
  }

  program->setVarInfoCallBack(&getSvarInfo);

  if (CL_SUCCESS != program->addDeviceProgram(*hip::getCurrentContext()->devices()[0], image, ElfSize(image)) ||
    CL_SUCCESS != program->build(hip::getCurrentContext()->devices(), nullptr, nullptr, nullptr)) {
      return hipErrorUnknown;
  }

      *module = reinterpret_cast<hipModule_t>(as_cl(program));

  if (!ihipModuleRegisterGlobal(program, module)) {
      return hipErrorUnknown;
  }

  return hipSuccess;
}

hipError_t hipModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod, const char *name)
{
  HIP_INIT_API(hfunc, hmod, name);

  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));

  const amd::Symbol* symbol = program->findSymbol(name);
  if (!symbol) {
    HIP_RETURN(hipErrorNotFound);
  }

  amd::Kernel* kernel = new amd::Kernel(*program, *symbol, name);
  if (!kernel) {
    HIP_RETURN(hipErrorOutOfMemory);
  }

  hip::Function* f = new hip::Function(kernel);
  *hfunc = f->asHipFunction();

  HIP_RETURN(hipSuccess);
}

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod, const char* name)
{
  HIP_INIT_API(dptr, bytes, hmod, name);

  /* Get address and size for the global symbol */
  if (!PlatformState::instance().getGlobalVar(name, ihipGetDevice(), dptr,
                                              bytes)) {
    HIP_RETURN(hipErrorUnknown);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func)
{
  HIP_INIT_API(attr, func);

  if (!PlatformState::instance().getFuncAttr(func, attr)) {
    HIP_RETURN(hipErrorUnknown);
  }

  HIP_RETURN(hipSuccess);
}


hipError_t ihipModuleLaunchKernel(hipFunction_t f,
                                 uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                 uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                 uint32_t sharedMemBytes, hipStream_t hStream,
                                 void **kernelParams, void **extra,
                                 hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags = 0,
                                 uint32_t params = 0)
{
  HIP_INIT_API(f, gridDimX, gridDimY, gridDimZ,
               blockDimX, blockDimY, blockDimZ,
               sharedMemBytes, hStream,
               kernelParams, extra, startEvent, stopEvent, flags);

  hip::Function* function = hip::Function::asFunction(f);
  amd::Kernel* kernel = function->function_;
  amd::Device* device = hip::getCurrentContext()->devices()[0];

  amd::ScopedLock lock(function->lock_);

  hip::Event* eStart = reinterpret_cast<hip::Event*>(startEvent);
  hip::Event* eStop = reinterpret_cast<hip::Event*>(stopEvent);
  amd::HostQueue* queue;
  if (hStream == nullptr) {
    hip::syncStreams();
    queue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();
    queue = reinterpret_cast<hip::Stream*>(hStream)->asHostQueue();
  }
  if ((params & amd::NDRangeKernelCommand::CooperativeGroups) &&
      !device->info().cooperativeGroups_) {
    return hipErrorLaunchFailure;
  }
  if ((params & amd::NDRangeKernelCommand::CooperativeMultiDeviceGroups) &&
      !device->info().cooperativeMultiDeviceGroups_) {
    return hipErrorLaunchFailure;
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

  if(startEvent != nullptr) {
    amd::Command* startCommand = new hip::TimerMarker(*queue);
    startCommand->enqueue();
    eStart->addMarker(queue, startCommand);
  }

  amd::NDRangeKernelCommand* command = new amd::NDRangeKernelCommand(
    *queue, waitList, *kernel, ndrange, sharedMemBytes, params);
  if (!command) {
    return hipErrorOutOfMemory;
  }

  // Capture the kernel arguments
  if (CL_SUCCESS != command->captureAndValidate()) {
    delete command;
    return hipErrorMemoryAllocation;
  }

  command->enqueue();

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
  HIP_RETURN(ihipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, hStream, kernelParams, extra, startEvent, stopEvent));
}

hipError_t hipLaunchCooperativeKernel(const void* f,
                                      dim3 gridDim, dim3 blockDim,
                                      void **kernelParams, uint32_t sharedMemBytes, hipStream_t hStream)
{
  int deviceId = ihipGetDevice();
  hipFunction_t func = PlatformState::instance().getFunc(f, deviceId);
  if (func == nullptr) {
    HIP_RETURN(hipErrorUnknown);
  }

  HIP_RETURN(ihipModuleLaunchKernel(func, gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z,
                                blockDim.x, blockDim.y, blockDim.z,
                                sharedMemBytes, hStream, kernelParams, nullptr, nullptr, nullptr, 0,
                                amd::NDRangeKernelCommand::CooperativeGroups));
}

hipError_t ihipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                  int numDevices, unsigned int flags, uint32_t extFlags)
{
  int currentDevice = ihipGetDevice();
  int numActiveGPUs = 0;
  ihipDeviceGetCount(&numActiveGPUs);

  if ((numDevices > numActiveGPUs) || (launchParamsList == nullptr)) {
    return hipErrorInvalidValue;
  }
 
  hipError_t result = hipErrorUnknown;
  for (int i = 0; i < numDevices; ++i) {
    hipSetDevice(i);
    const hipLaunchParams& launch = launchParamsList[i];
    amd::HostQueue* queue = reinterpret_cast<hip::Stream*>(launch.stream)->asHostQueue();
    hipFunction_t func = PlatformState::instance().getFunc(launch.func, i);
    if (func == nullptr) {
      HIP_RETURN(result);
    }
    result = ihipModuleLaunchKernel(func,
      launch.gridDim.x * launch.blockDim.x,
      launch.gridDim.y * launch.blockDim.y,
      launch.gridDim.z * launch.blockDim.z,
      launch.blockDim.x, launch.blockDim.y, launch.blockDim.z,
      launch.sharedMem, launch.stream,
      launch.args, nullptr, nullptr, nullptr, flags, extFlags);
  }

  hipSetDevice(currentDevice);
  return result;
}

hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                 int numDevices, unsigned int flags)
{
  HIP_INIT_API(launchParamsList, numDevices, flags);

  return ihipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags,
                                                (amd::NDRangeKernelCommand::CooperativeGroups |
                                                 amd::NDRangeKernelCommand::CooperativeMultiDeviceGroups));
}

hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                              int numDevices, unsigned int flags) {
  HIP_INIT_API(launchParamsList, numDevices, flags);

  return ihipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags, 0);
}

hipError_t hipModuleGetTexRef(textureReference** texRef, hipModule_t hmod, const char* name) {
  HIP_INIT_API(texRef, hmod, name);

  hipDeviceptr_t dptr = nullptr;
  size_t bytes = 0;

  /* input args check */
  if ((texRef == nullptr) || (name == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }

   /* Get address and size for the global symbol */
  if (!PlatformState::instance().getGlobalVar(name, ihipGetDevice(), &dptr,
                                              &bytes)) {
    HIP_RETURN(hipErrorUnknown);
  }

  *texRef = reinterpret_cast<textureReference*>(dptr);
  HIP_RETURN(hipSuccess);
}


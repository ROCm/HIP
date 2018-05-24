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

hipError_t hipModuleLoad(hipModule_t *module, const char *fname)
{
  HIP_INIT_API(module, fname);

  if (!fname) {
    return hipErrorInvalidValue;
  }

  std::ifstream file{fname};

  if (!file.is_open()) {
    return hipErrorFileNotFound;
  }

  std::vector<char> tmp{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};

  return ihipModuleLoadData(module, tmp.data());
}


hipError_t hipModuleUnload(hipModule_t hmod)
{
  HIP_INIT_API(hmod);

  if (hmod == nullptr) {
    return hipErrorUnknown;
  }

  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));

  program->release();

  return hipSuccess;
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image)
{
  HIP_INIT_API(module, image);

  return ihipModuleLoadData(module, image);
}

hipError_t ihipModuleLoadData(hipModule_t *module, const void *image)
{
  amd::Program* program = new amd::Program(*hip::getCurrentContext());
  if (program == NULL) {
    return hipErrorOutOfMemory;
  }

  if (CL_SUCCESS != program->addDeviceProgram(*hip::getCurrentContext()->devices()[0], image, ElfSize(image)) ||
    CL_SUCCESS != program->build(hip::getCurrentContext()->devices(), nullptr, nullptr, nullptr)) {
    return hipErrorUnknown;
  }

  *module = reinterpret_cast<hipModule_t>(as_cl(program));

  return hipSuccess;
}

hipError_t hipModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod, const char *name)
{
  HIP_INIT_API(hfunc, hmod, name);

  amd::Program* program = as_amd(reinterpret_cast<cl_program>(hmod));

  const amd::Symbol* symbol = program->findSymbol(name);
  if (!symbol) {
    return hipErrorNotFound;
  }

  amd::Kernel* kernel = new amd::Kernel(*program, *symbol, name);
  if (!kernel) {
    return hipErrorOutOfMemory;
  }

  *hfunc = reinterpret_cast<hipFunction_t>(as_cl(kernel));

  return hipSuccess;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                 uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                 uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                 uint32_t sharedMemBytes, hipStream_t hStream,
                                 void **kernelParams, void **extra)
{
  HIP_INIT_API(f, gridDimX, gridDimY, gridDimZ,
               blockDimX, blockDimY, blockDimZ,
               sharedMemBytes, hStream,
               kernelParams, extra);

  amd::Kernel* kernel = as_amd(reinterpret_cast<cl_kernel>(f));
  amd::Device* device = hip::getCurrentContext()->devices()[0];

  amd::HostQueue* queue;
  if (hStream == nullptr) {
    hip::syncStreams();
    queue = hip::getNullStream();
  } else {
    hip::getNullStream()->finish();
    queue = as_amd(reinterpret_cast<cl_command_queue>(hStream))->asHostQueue();
  }
  if (!queue) {
    return hipErrorOutOfMemory;
  }

  size_t globalWorkOffset[3] = {0};
  size_t globalWorkSize[3] = { gridDimX * blockDimX, gridDimY * blockDimY, gridDimZ * blockDimZ};
  size_t localWorkSize[3] = { blockDimX, blockDimY, blockDimZ };
  amd::NDRangeContainer ndrange(3, globalWorkOffset, globalWorkSize, localWorkSize);
  amd::Command::EventWaitList waitList;

  // 'extra' is a struct that contains the following info: {
  //   HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
  //   HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargs_size,
  //   HIP_LAUNCH_PARAM_END }
  if (extra[0] != HIP_LAUNCH_PARAM_BUFFER_POINTER ||
      extra[2] != HIP_LAUNCH_PARAM_BUFFER_SIZE || extra[4] != HIP_LAUNCH_PARAM_END) {
    return hipErrorNotInitialized;
  }
  address kernargs = reinterpret_cast<address>(extra[1]);

  const amd::KernelSignature& signature = kernel->signature();
  for (size_t i = 0; i < signature.numParameters(); ++i) {
    const amd::KernelParameterDescriptor& desc = signature.at(i);
    if (kernelParams == nullptr) {
      assert(extra);
      kernel->parameters().set(i, desc.size_, kernargs + desc.offset_,
                               desc.type_ == T_POINTER/*svmBound*/);
    } else {
      assert(!extra);
      kernel->parameters().set(i, desc.size_, kernelParams[i], desc.type_ == T_POINTER/*svmBound*/);
    }
  }

  amd::NDRangeKernelCommand* command = new amd::NDRangeKernelCommand(*queue, waitList, *kernel, ndrange);
  if (!command) {
    return hipErrorOutOfMemory;
  }

  // Capture the kernel arguments
  if (CL_SUCCESS != command->captureAndValidate()) {
    delete command;
    return hipErrorMemoryAllocation;
  }

  command->enqueue();
  command->release();

  return hipSuccess;
}



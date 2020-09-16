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
#include <hip/hcc_detail/texture_types.h>
#include "hip_platform.hpp"
#include "hip_internal.hpp"
#include "platform/program.hpp"
#include "platform/runtime.hpp"

#include <unordered_map>

constexpr unsigned __hipFatMAGIC2 = 0x48495046; // "HIPF"

thread_local std::stack<ihipExec_t> execStack_;
PlatformState* PlatformState::platform_; // Initiaized as nullptr by default

struct __CudaFatBinaryWrapper {
  unsigned int magic;
  unsigned int version;
  void*        binary;
  void*        dummy1;
};

hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes,
    hipModule_t hmod, const char* name);

hipError_t ihipCreateGlobalVarObj(const char* name, hipModule_t hmod, amd::Memory** amd_mem_obj,
                                  hipDeviceptr_t* dptr, size_t* bytes);

extern hipError_t ihipModuleLaunchKernel(hipFunction_t f,
                                 uint32_t gridDimX, uint32_t gridDimY, uint32_t gridDimZ,
                                 uint32_t blockDimX, uint32_t blockDimY, uint32_t blockDimZ,
                                 uint32_t sharedMemBytes, hipStream_t hStream,
                                 void **kernelParams, void **extra,
                                 hipEvent_t startEvent, hipEvent_t stopEvent, uint32_t flags = 0,
                                 uint32_t params = 0, uint32_t gridId = 0, uint32_t numGrids = 0,
                                 uint64_t prevGridSum = 0, uint64_t allGridSum = 0, uint32_t firstDevice = 0);
static bool isCompatibleCodeObject(const std::string& codeobj_target_id,
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

extern "C" hip::FatBinaryInfo** __hipRegisterFatBinary(const void* data)
{
  const __CudaFatBinaryWrapper* fbwrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(data);
  if (fbwrapper->magic != __hipFatMAGIC2 || fbwrapper->version != 1) {
    DevLogPrintfError("Cannot Register fat binary. FatMagic: %u version: %u ",
                      fbwrapper->magic, fbwrapper->version);
    return nullptr;
  }

  return PlatformState::instance().addFatBinary(fbwrapper->binary);
}

bool PlatformState::getShadowVarInfo(std::string var_name, hipModule_t hmod,
                                     void** var_addr, size_t* var_size) {

  amd::ScopedLock lock(lock_);
  if (hipSuccess == getDynGlobalVar(var_name.c_str(), ihipGetDevice(), hmod, var_addr, var_size)) {
    return true;
  }

  if (hipSuccess == getStatGlobalVarByName(var_name, ihipGetDevice(), hmod, var_addr, var_size)) {
    return true;
  }

  return false;
}

bool CL_CALLBACK getSvarInfo(cl_program program, std::string var_name, void** var_addr,
                             size_t* var_size) {
  return PlatformState::instance().getShadowVarInfo(var_name, reinterpret_cast<hipModule_t>(program),
                                                    var_addr, var_size);
}

extern "C" void __hipRegisterFunction(
  hip::FatBinaryInfo** modules,
  const void*  hostFunction,
  char*        deviceFunction,
  const char*  deviceName,
  unsigned int threadLimit,
  uint3*       tid,
  uint3*       bid,
  dim3*        blockDim,
  dim3*        gridDim,
  int*         wSize) {
  static int enable_deferred_loading { []() {
    char *var = getenv("HIP_ENABLE_DEFERRED_LOADING");
    return var ? atoi(var) : 1;
  }() };

  hip::Function* func = new hip::Function(std::string(deviceName), modules);
  PlatformState::instance().registerStatFunction(hostFunction, func);

  if (!enable_deferred_loading) {
    HIP_INIT();
    hipFunction_t hfunc = nullptr;
    hipError_t hip_error = hipSuccess;
    for (size_t dev_idx = 0; dev_idx < g_devices.size(); ++dev_idx) {
      hip_error = PlatformState::instance().getStatFunc(&hfunc, hostFunction, dev_idx);
      guarantee((hip_error == hipSuccess) && "Cannot Retrieve Static function");
    }
  }
}

// Registers a device-side global variable.
// For each global variable in device code, there is a corresponding shadow
// global variable in host code. The shadow host variable is used to keep
// track of the value of the device side global variable between kernel
// executions.
extern "C" void __hipRegisterVar(
  hip::FatBinaryInfo** modules,   // The device modules containing code object
  void*       var,       // The shadow variable in host code
  char*       hostVar,   // Variable name in host code
  char*       deviceVar, // Variable name in device code
  int         ext,       // Whether this variable is external
  size_t      size,      // Size of the variable
  int         constant,  // Whether this variable is constant
  int         global)    // Unknown, always 0
{
  hip::Var* var_ptr = new hip::Var(std::string(hostVar), hip::Var::DeviceVarKind::DVK_Variable, size, 0, 0, modules);
  PlatformState::instance().registerStatGlobalVar(var, var_ptr);
}

extern "C" void __hipRegisterSurface(hip::FatBinaryInfo** modules,      // The device modules containing code object
                                     void* var,        // The shadow variable in host code
                                     char* hostVar,    // Variable name in host code
                                     char* deviceVar,  // Variable name in device code
                                     int type, int ext) {
  hip::Var* var_ptr = new hip::Var(std::string(hostVar), hip::Var::DeviceVarKind::DVK_Surface, sizeof(surfaceReference), 0, 0, modules);
  PlatformState::instance().registerStatGlobalVar(var, var_ptr);
}

extern "C" void __hipRegisterTexture(hip::FatBinaryInfo** modules,      // The device modules containing code object
                                     void* var,        // The shadow variable in host code
                                     char* hostVar,    // Variable name in host code
                                     char* deviceVar,  // Variable name in device code
                                     int type, int norm, int ext) {
  hip::Var* var_ptr = new hip::Var(std::string(hostVar), hip::Var::DeviceVarKind::DVK_Texture, sizeof(textureReference), 0, 0, modules);
  PlatformState::instance().registerStatGlobalVar(var, var_ptr);
}

extern "C" void __hipUnregisterFatBinary(hip::FatBinaryInfo** modules)
{
  HIP_INIT();

  PlatformState::instance().removeFatBinary(modules);
}

extern "C" hipError_t hipConfigureCall(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem,
  hipStream_t stream)
{
  HIP_INIT_API(hipConfigureCall, gridDim, blockDim, sharedMem, stream);

  PlatformState::instance().configureCall(gridDim, blockDim, sharedMem, stream);

  HIP_RETURN(hipSuccess);
}

extern "C" hipError_t __hipPushCallConfiguration(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem,
  hipStream_t stream)
{
  HIP_INIT_API(__hipPushCallConfiguration, gridDim, blockDim, sharedMem, stream);

  PlatformState::instance().configureCall(gridDim, blockDim, sharedMem, stream);

  HIP_RETURN(hipSuccess);
}

extern "C" hipError_t __hipPopCallConfiguration(dim3 *gridDim,
                                                dim3 *blockDim,
                                                size_t *sharedMem,
                                                hipStream_t *stream) {
  HIP_INIT_API(__hipPopCallConfiguration, gridDim, blockDim, sharedMem, stream);

  ihipExec_t exec;
  PlatformState::instance().popExec(exec);
  *gridDim = exec.gridDim_;
  *blockDim = exec.blockDim_;
  *sharedMem = exec.sharedMem_;
  *stream = exec.hStream_;

  HIP_RETURN(hipSuccess);
}

extern "C" hipError_t hipSetupArgument(
  const void *arg,
  size_t size,
  size_t offset)
{
  HIP_INIT_API(hipSetupArgument, arg, size, offset);

  PlatformState::instance().setupArgument(arg, size, offset);

  HIP_RETURN(hipSuccess);
}

extern "C" hipError_t hipLaunchByPtr(const void *hostFunction)
{
  HIP_INIT_API(hipLaunchByPtr, hostFunction);

  ihipExec_t exec;
  PlatformState::instance().popExec(exec);

  hip::Stream* stream = reinterpret_cast<hip::Stream*>(exec.hStream_);
  int deviceId = (stream != nullptr)? stream->DeviceId() : ihipGetDevice();
  if (deviceId == -1) {
    DevLogPrintfError("Wrong DeviceId: %d \n", deviceId);
    HIP_RETURN(hipErrorNoDevice);
  }
  hipFunction_t func = nullptr;
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, hostFunction, deviceId);
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    DevLogPrintfError("Could not retrieve hostFunction: 0x%x \n", hostFunction);
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }

  size_t size = exec.arguments_.size();
  void *extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &exec.arguments_[0],
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
      HIP_LAUNCH_PARAM_END
    };

  HIP_RETURN(hipModuleLaunchKernel(func,
    exec.gridDim_.x, exec.gridDim_.y, exec.gridDim_.z,
    exec.blockDim_.x, exec.blockDim_.y, exec.blockDim_.z,
    exec.sharedMem_, exec.hStream_, nullptr, extra));
}

hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
  HIP_INIT_API(hipGetSymbolAddress, devPtr, symbol);

  hipError_t hip_error = hipSuccess;
  size_t sym_size = 0;

  HIP_RETURN_ONFAIL(PlatformState::instance().getStatGlobalVar(symbol, ihipGetDevice(), devPtr, &sym_size));

  HIP_RETURN(hipSuccess, *devPtr);
}

hipError_t hipGetSymbolSize(size_t* sizePtr, const void* symbol) {
  HIP_INIT_API(hipGetSymbolSize, sizePtr, symbol);

  hipDeviceptr_t device_ptr = nullptr;
  HIP_RETURN_ONFAIL(PlatformState::instance().getStatGlobalVar(symbol, ihipGetDevice(), &device_ptr, sizePtr));

  HIP_RETURN(hipSuccess, *sizePtr);
}

hipError_t ihipCreateGlobalVarObj(const char* name, hipModule_t hmod, amd::Memory** amd_mem_obj,
                                  hipDeviceptr_t* dptr, size_t* bytes)
{
  HIP_INIT();

  amd::Program* program = nullptr;
  device::Program* dev_program = nullptr;

  /* Get Device Program pointer*/
  program = as_amd(reinterpret_cast<cl_program>(hmod));
  dev_program = program->getDeviceProgram(*hip::getCurrentDevice()->devices()[0]);

  if (dev_program == nullptr) {
    DevLogPrintfError("Cannot get Device Function for module: 0x%x \n", hmod);
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  /* Find the global Symbols */
  if (!dev_program->createGlobalVarObj(amd_mem_obj, dptr, bytes, name)) {
    DevLogPrintfError("Cannot create Global Var obj for symbol: %s \n", name);
    HIP_RETURN(hipErrorInvalidSymbol);
  }

  HIP_RETURN(hipSuccess);
}


namespace hip_impl {
hipError_t ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* maxBlocksPerCU, int* numBlocksPerGrid, int* bestBlockSize,
    const amd::Device& device, hipFunction_t func, int inputBlockSize,
    size_t dynamicSMemSize, bool bCalcPotentialBlkSz)
{
  hip::DeviceFunc* function = hip::DeviceFunc::asFunction(func);
  const amd::Kernel& kernel = *function->kernel();

  const device::Kernel::WorkGroupInfo* wrkGrpInfo = kernel.getDeviceKernel(device)->workGroupInfo();
  if (bCalcPotentialBlkSz == false) {
    if (inputBlockSize == 0) {
      return hipErrorInvalidValue;
    }
    *bestBlockSize = 0;
    // Make sure the requested block size is smaller than max supported
    if (inputBlockSize > int(device.info().maxWorkGroupSize_)) {
        *maxBlocksPerCU = 0;
        *numBlocksPerGrid = 0;
        return hipSuccess;
    }
  }
  else {
    if (inputBlockSize > int(device.info().maxWorkGroupSize_) ||
            inputBlockSize == 0) {
      // The user wrote the kernel to work with a workgroup size
      // bigger than this hardware can support. Or they do not care
      // about the size So just assume its maximum size is
      // constrained by hardware
      inputBlockSize = device.info().maxWorkGroupSize_;
    }
  }
  // Find wave occupancy per CU => simd_per_cu * GPR usage
  constexpr size_t MaxWavesPerSimd = 8;  // Limited by SPI 32 per CU, hence 8 per SIMD
  size_t VgprWaves = MaxWavesPerSimd;
  if (wrkGrpInfo->usedVGPRs_ > 0) {
    VgprWaves = wrkGrpInfo->availableVGPRs_ / amd::alignUp(wrkGrpInfo->usedVGPRs_, 4);
  }

  size_t GprWaves = VgprWaves;
  if (wrkGrpInfo->usedSGPRs_ > 0) {
    size_t maxSGPRs;
    if (device.info().gfxipMajor_ < 8) {
      maxSGPRs = 512;
    }
    else if (device.info().gfxipMajor_ < 10) {
      maxSGPRs = 800;
    }
    else {
      maxSGPRs = SIZE_MAX; // gfx10+ does not share SGPRs between waves
    }
    const size_t SgprWaves = maxSGPRs / amd::alignUp(wrkGrpInfo->usedSGPRs_, 16);
    GprWaves = std::min(VgprWaves, SgprWaves);
  }

  const size_t alu_occupancy = device.info().simdPerCU_ * std::min(MaxWavesPerSimd, GprWaves);
  const int alu_limited_threads = alu_occupancy * wrkGrpInfo->wavefrontSize_;

  int lds_occupancy_wgs = INT_MAX;
  const size_t total_used_lds = wrkGrpInfo->usedLDSSize_ + dynamicSMemSize;
  if (total_used_lds != 0) {
    lds_occupancy_wgs = static_cast<int>(device.info().localMemSize_ / total_used_lds);
  }
  // Calculate how many blocks of inputBlockSize we can fit per CU
  // Need to align with hardware wavefront size. If they want 65 threads, but
  // waves are 64, then we need 128 threads per block.
  // So this calculates how many blocks we can fit.
  *maxBlocksPerCU = alu_limited_threads / amd::alignUp(inputBlockSize, wrkGrpInfo->wavefrontSize_);
  // Unless those blocks are further constrained by LDS size.
  *maxBlocksPerCU = std::min(*maxBlocksPerCU, lds_occupancy_wgs);

  // Some callers of this function want to return the block size, in threads, that
  // leads to the maximum occupancy. In that case, inputBlockSize is the maximum
  // workgroup size the user wants to allow, or that the hardware can allow.
  // It is either the number of threads that we are limited to due to occupancy, or
  // the maximum available block size for this kernel, which could have come from the
  // user. e.g., if the user indicates the maximum block size is 64 threads, but we
  // calculate that 128 threads can fit in each CU, we have to give up and return 64.
  *bestBlockSize = std::min(alu_limited_threads, amd::alignUp(inputBlockSize, wrkGrpInfo->wavefrontSize_));
  // If the best block size is smaller than the block size used to fit the maximum,
  // then we need to make the grid bigger for full occupancy.
  const int bestBlocksPerCU = alu_limited_threads / (*bestBlockSize);
  // Unless those blocks are further constrained by LDS size.
  *numBlocksPerGrid = device.info().maxComputeUnits_ * std::min(bestBlocksPerCU, lds_occupancy_wgs);

  return hipSuccess;
}
}

extern "C" {
hipError_t hipOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             const void* f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit)
{
  HIP_INIT_API(hipOccupancyMaxPotentialBlockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  if ((gridSize == nullptr) || (blockSize == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipFunction_t func = nullptr;
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, f, ihipGetDevice());
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    return HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int num_blocks = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &max_blocks_per_grid, &best_block_size, device, func, blockSizeLimit, dynSharedMemPerBlk,true);
  if (ret == hipSuccess) {
    *blockSize = best_block_size;
    *gridSize = max_blocks_per_grid;
  }
  HIP_RETURN(ret);
}

hipError_t hipModuleOccupancyMaxPotentialBlockSize(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit)
{
  HIP_INIT_API(hipModuleOccupancyMaxPotentialBlockSize, f, dynSharedMemPerBlk, blockSizeLimit);
  if ((gridSize == nullptr) || (blockSize == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int num_blocks = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSizeLimit, dynSharedMemPerBlk,true);
  if (ret == hipSuccess) {
    *blockSize = best_block_size;
    *gridSize = max_blocks_per_grid;
  }
  HIP_RETURN(ret);
}

hipError_t hipModuleOccupancyMaxPotentialBlockSizeWithFlags(int* gridSize, int* blockSize,
                                             hipFunction_t f, size_t dynSharedMemPerBlk,
                                             int blockSizeLimit, unsigned int flags)
{
  HIP_INIT_API(hipModuleOccupancyMaxPotentialBlockSizeWithFlags, f, dynSharedMemPerBlk, blockSizeLimit, flags);
  if ((gridSize == nullptr) || (blockSize == nullptr)) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];
  int max_blocks_per_grid = 0;
  int num_blocks = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSizeLimit, dynSharedMemPerBlk,true);
  if (ret == hipSuccess) {
    *blockSize = best_block_size;
    *gridSize = max_blocks_per_grid;
  }
  HIP_RETURN(ret);
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                             hipFunction_t f, int blockSize, size_t dynSharedMemPerBlk)
{
  HIP_INIT_API(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor, f, blockSize, dynSharedMemPerBlk);
  if (numBlocks == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSize, dynSharedMemPerBlk, false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                              hipFunction_t f, int blockSize,
                                                              size_t dynSharedMemPerBlk, unsigned int flags)
{
  HIP_INIT_API(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, f, blockSize, dynSharedMemPerBlk, flags);
  if (numBlocks == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &max_blocks_per_grid, &best_block_size, device, f, blockSize, dynSharedMemPerBlk, false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                        const void* f, int blockSize, size_t dynamicSMemSize)
{
  HIP_INIT_API(hipOccupancyMaxActiveBlocksPerMultiprocessor, f, blockSize, dynamicSMemSize);
  if (numBlocks == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipFunction_t func = nullptr;
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, f, ihipGetDevice());
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    return HIP_RETURN(hipErrorInvalidValue);
  }

  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &max_blocks_per_grid, &best_block_size, device, func, blockSize, dynamicSMemSize, false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                                 const void* f,
                                                                 int  blockSize, size_t dynamicSMemSize, unsigned int flags)
{
  HIP_INIT_API(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags, f, blockSize, dynamicSMemSize, flags);
  if (numBlocks == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  hipFunction_t func = nullptr;
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, f, ihipGetDevice());
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    return HIP_RETURN(hipErrorInvalidValue);
  }

  const amd::Device& device = *hip::getCurrentDevice()->devices()[0];

  int num_blocks = 0;
  int max_blocks_per_grid = 0;
  int best_block_size = 0;
  hipError_t ret = hip_impl::ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks, &max_blocks_per_grid, &best_block_size, device, func, blockSize, dynamicSMemSize, false);
  *numBlocks = num_blocks;
  HIP_RETURN(ret);
}
}


#if defined(ATI_OS_LINUX)

namespace hip_impl {

void hipLaunchKernelGGLImpl(
  uintptr_t function_address,
  const dim3& numBlocks,
  const dim3& dimBlocks,
  uint32_t sharedMemBytes,
  hipStream_t stream,
  void** kernarg)
{
  HIP_INIT();

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  int deviceId = (s != nullptr)? s->DeviceId() : ihipGetDevice();
  if (deviceId == -1) {
    DevLogPrintfError("Wrong Device Id: %d \n", deviceId);
  }

  hipFunction_t func = nullptr;
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, reinterpret_cast<void*>(function_address), deviceId);
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    DevLogPrintfError("Cannot find the static function: 0x%x", function_address);
  }

  hipModuleLaunchKernel(func,
    numBlocks.x, numBlocks.y, numBlocks.z,
    dimBlocks.x, dimBlocks.y, dimBlocks.z,
    sharedMemBytes, stream, nullptr, kernarg);
}

void hipLaunchCooperativeKernelGGLImpl(
  uintptr_t function_address,
  const dim3& numBlocks,
  const dim3& dimBlocks,
  uint32_t sharedMemBytes,
  hipStream_t stream,
  void** kernarg)
{
  HIP_INIT();

  hipLaunchCooperativeKernel(reinterpret_cast<void*>(function_address),
    numBlocks, dimBlocks, kernarg, sharedMemBytes, stream);
}

}

#endif // defined(ATI_OS_LINUX)

hipError_t ihipLaunchKernel(const void* hostFunction,
                                         dim3 gridDim,
                                         dim3 blockDim,
                                         void** args,
                                         size_t sharedMemBytes,
                                         hipStream_t stream,
                                         hipEvent_t startEvent,
                                         hipEvent_t stopEvent,
                                         int flags)
{
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  int deviceId = (s != nullptr)? s->DeviceId() : ihipGetDevice();
  if (deviceId == -1) {
    DevLogPrintfError("Wrong Device Id: %d \n", deviceId);
    HIP_RETURN(hipErrorNoDevice);
  }

  hipFunction_t func =  nullptr;
  hipError_t hip_error = PlatformState::instance().getStatFunc(&func, hostFunction, deviceId);
  if ((hip_error != hipSuccess) || (func == nullptr)) {
    HIP_RETURN(hipErrorInvalidDeviceFunction);
  }
  size_t globalWorkSizeX = gridDim.x * blockDim.x;
  size_t globalWorkSizeY = gridDim.y * blockDim.y;
  size_t globalWorkSizeZ = gridDim.z * blockDim.z;
  if (globalWorkSizeX > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeY > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeZ > std::numeric_limits<uint32_t>::max()) {
    HIP_RETURN(hipErrorInvalidConfiguration);
  }
  HIP_RETURN(ihipModuleLaunchKernel(func, static_cast<uint32_t>(globalWorkSizeX),
                                    static_cast<uint32_t>(globalWorkSizeY),
                                    static_cast<uint32_t>(globalWorkSizeZ),
                                    blockDim.x, blockDim.y, blockDim.z,
                                    sharedMemBytes, stream, args, nullptr, startEvent, stopEvent,
                                    flags));
}

// conversion routines between float and half precision

static inline std::uint32_t f32_as_u32(float f) { union { float f; std::uint32_t u; } v; v.f = f; return v.u; }

static inline float u32_as_f32(std::uint32_t u) { union { float f; std::uint32_t u; } v; v.u = u; return v.f; }

static inline int clamp_int(int i, int l, int h) { return std::min(std::max(i, l), h); }


// half float, the f16 is in the low 16 bits of the input argument

static inline float __convert_half_to_float(std::uint32_t a) noexcept {

  std::uint32_t u = ((a << 13) + 0x70000000U) & 0x8fffe000U;

  std::uint32_t v = f32_as_u32(u32_as_f32(u) * u32_as_f32(0x77800000U)/*0x1.0p+112f*/) + 0x38000000U;

  u = (a & 0x7fff) != 0 ? v : u;

  return u32_as_f32(u) * u32_as_f32(0x07800000U)/*0x1.0p-112f*/;

}

// float half with nearest even rounding
// The lower 16 bits of the result is the bit pattern for the f16
static inline std::uint32_t __convert_float_to_half(float a) noexcept {
  std::uint32_t u = f32_as_u32(a);
  int e = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
  std::uint32_t m = ((u >> 11) & 0xffe) | ((u & 0xfff) != 0);
  std::uint32_t i = 0x7c00 | (m != 0 ? 0x0200 : 0);
  std::uint32_t n = ((std::uint32_t)e << 12) | m;
  std::uint32_t s = (u >> 16) & 0x8000;
  int b = clamp_int(1-e, 0, 13);
  std::uint32_t d = (0x1000 | m) >> b;
  d |= (d << b) != (0x1000 | m);
  std::uint32_t v = e < 1 ? d : n;
  v = (v >> 2) + (((v & 0x7) == 3) | ((v & 0x7) > 5));
  v = e > 30 ? 0x7c00 : v;
  v = e == 143 ? i : v;
  return s | v;
}

extern "C"
#if !defined(_MSC_VER)
__attribute__((weak))
#endif
float  __gnu_h2f_ieee(unsigned short h){
  return __convert_half_to_float((std::uint32_t) h);
}

extern "C"
#if !defined(_MSC_VER)
__attribute__((weak))
#endif
unsigned short  __gnu_f2h_ieee(float f){
  return (unsigned short)__convert_float_to_half(f);
}

void PlatformState::init()
{
  amd::ScopedLock lock(lock_);
  if(initialized_ || g_devices.empty()) {
    return;
  }
  initialized_ = true;
  for (auto& it : statCO_.modules_) {
    digestFatBinary(it.first, it.second);
  }
  for (auto &it : statCO_.vars_) {
    it.second->resize_dVar(g_devices.size());
  }
  for (auto &it : statCO_.functions_) {
    it.second->resize_dFunc(g_devices.size());
  }
}

hipError_t PlatformState::loadModule(hipModule_t *module, const char* fname, const void* image) {
  amd::ScopedLock lock(lock_);

  hip::DynCO* dynCo = new hip::DynCO();
  hipError_t hip_error = dynCo->loadCodeObject(fname, image);
  if (hip_error != hipSuccess) {
    delete dynCo;
    return hip_error;
  }

  *module = dynCo->module();
  assert(*module != nullptr);

  if (dynCO_map_.find(*module) != dynCO_map_.end()) {
    return hipErrorAlreadyMapped;
  }
  dynCO_map_.insert(std::make_pair(*module, dynCo));

  return hipSuccess;
}

hipError_t PlatformState::unloadModule(hipModule_t hmod) {
  amd::ScopedLock lock(lock_);

  auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    return hipErrorNotFound;
  }

  delete it->second;
  dynCO_map_.erase(hmod);

  auto tex_it = texRef_map_.begin();
  while (tex_it != texRef_map_.end()) {
    if (tex_it->second.first == hmod) {
      tex_it = texRef_map_.erase(tex_it);
    } else {
      ++tex_it;
    }
  }

  return hipSuccess;
}

hipError_t PlatformState::getDynFunc(hipFunction_t* hfunc, hipModule_t hmod,
                                         const char* func_name) {
  amd::ScopedLock lock(lock_);

  auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    DevLogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }
  if (0 == strlen(func_name)) {
    return hipErrorNotFound;
  }

  return it->second->getDynFunc(hfunc, func_name);
}

hipError_t PlatformState::getDynGlobalVar(const char* hostVar, int deviceId, hipModule_t hmod,
                                          hipDeviceptr_t* dev_ptr, size_t* size_ptr) {
  amd::ScopedLock lock(lock_);

  auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    DevLogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }

  hip::DeviceVar* dvar = nullptr;
  IHIP_RETURN_ONFAIL(it->second->getDeviceVar(&dvar, hostVar, deviceId));
  *dev_ptr = dvar->device_ptr();
  *size_ptr = dvar->size();

  return hipSuccess;
}

hipError_t PlatformState::registerTexRef(textureReference* texRef, hipModule_t hmod,
                                         std::string name) {
  amd::ScopedLock lock(lock_);
  texRef_map_.insert(std::make_pair(texRef, std::make_pair(hmod, name)));
  return hipSuccess;
}

hipError_t PlatformState::getDynTexGlobalVar(textureReference* texRef, int deviceId,
                                             hipDeviceptr_t* dev_ptr, size_t* size_ptr) {
  amd::ScopedLock lock(lock_);

  auto tex_it = texRef_map_.find(texRef);
  if (tex_it == texRef_map_.end()) {
    DevLogPrintfError("Cannot find the texRef Entry: 0x%x", texRef);
    return hipErrorNotFound;
  }

  auto it = dynCO_map_.find(tex_it->second.first);
  if (it == dynCO_map_.end()) {
    DevLogPrintfError("Cannot find the module: 0x%x", tex_it->second.first);
    return hipErrorNotFound;
  }

  hip::DeviceVar* dvar = nullptr;
  IHIP_RETURN_ONFAIL(it->second->getDeviceVar(&dvar, tex_it->second.second, deviceId));
  *dev_ptr = dvar->device_ptr();
  *size_ptr = dvar->size();

  return hipSuccess;
}

hipError_t PlatformState::getDynTexRef(const char* hostVar, hipModule_t hmod, textureReference** texRef) {
  amd::ScopedLock lock(lock_);

  auto it = dynCO_map_.find(hmod);
  if (it == dynCO_map_.end()) {
    DevLogPrintfError("Cannot find the module: 0x%x", hmod);
    return hipErrorNotFound;
  }

  hip::DeviceVar* dvar = nullptr;
  IHIP_RETURN_ONFAIL(it->second->getDeviceVar(&dvar, hostVar, ihipGetDevice()));

  dvar->shadowVptr = new texture<char>();
  *texRef =  reinterpret_cast<textureReference*>(dvar->shadowVptr);
  return hipSuccess;
}

hipError_t PlatformState::digestFatBinary(const void* data, hip::FatBinaryInfo*& programs) {
 return statCO_.digestFatBinary(data, programs);
}

hip::FatBinaryInfo** PlatformState::addFatBinary(const void* data) {
  return statCO_.addFatBinary(data, initialized_);
}

hipError_t PlatformState::removeFatBinary(hip::FatBinaryInfo** module) {
  return statCO_.removeFatBinary(module);
}

hipError_t PlatformState::registerStatFunction(const void* hostFunction, hip::Function* func) {
  return statCO_.registerStatFunction(hostFunction, func);
}

hipError_t PlatformState::registerStatGlobalVar(const void* hostVar, hip::Var* var) {
  return statCO_.registerStatGlobalVar(hostVar, var);
}

hipError_t PlatformState::getStatFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId) {
  return statCO_.getStatFunc(hfunc, hostFunction, deviceId);
}

hipError_t PlatformState::getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction, int deviceId) {
  return statCO_.getStatFuncAttr(func_attr, hostFunction, deviceId);
}

hipError_t PlatformState::getStatGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                                           size_t* size_ptr) {
  return statCO_.getStatGlobalVar(hostVar, deviceId, dev_ptr, size_ptr);
}

hipError_t PlatformState::getStatGlobalVarByName(std::string hostVar, int deviceId, hipModule_t hmod,
                                                 hipDeviceptr_t* dev_ptr, size_t* size_ptr) {
  return statCO_.getStatGlobalVarByName(hostVar, deviceId, hmod, dev_ptr, size_ptr);
}

void PlatformState::setupArgument(const void *arg, size_t size, size_t offset) {
  auto& arguments = execStack_.top().arguments_;

  if (arguments.size() < offset + size) {
    arguments.resize(offset + size);
  }

  ::memcpy(&arguments[offset], arg, size);
}

void PlatformState::configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                                  hipStream_t stream) {
  execStack_.push(ihipExec_t{gridDim, blockDim, sharedMem, stream});
}

void PlatformState::popExec(ihipExec_t& exec) {
  exec = std::move(execStack_.top());
  execStack_.pop();
}

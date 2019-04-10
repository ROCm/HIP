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

#include "CUDA2HIP.h"

// Map of all CUDA Runtime API functions
const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_FUNCTION_MAP{
  // 5.1. Device Management
  // no analogue
  {"cudaChooseDevice",                                        {"hipChooseDevice",                                        "", CONV_DEVICE, API_RUNTIME}},
  // cuDeviceGetAttribute
  {"cudaDeviceGetAttribute",                                  {"hipDeviceGetAttribute",                                  "", CONV_DEVICE, API_RUNTIME}},
  // cuDeviceGetByPCIBusId
  {"cudaDeviceGetByPCIBusId",                                 {"hipDeviceGetByPCIBusId",                                 "", CONV_DEVICE, API_RUNTIME}},
  // no analogue
  {"cudaDeviceGetCacheConfig",                                {"hipDeviceGetCacheConfig",                                "", CONV_DEVICE, API_RUNTIME}},
  // cuCtxGetLimit
  {"cudaDeviceGetLimit",                                      {"hipDeviceGetLimit",                                      "", CONV_DEVICE, API_RUNTIME}},
  // cuDeviceGetP2PAttribute
  {"cudaDeviceGetP2PAttribute",                               {"hipDeviceGetP2PAttribute",                               "", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuDeviceGetPCIBusId
  {"cudaDeviceGetPCIBusId",                                   {"hipDeviceGetPCIBusId",                                   "", CONV_DEVICE, API_RUNTIME}},
  // cuCtxGetSharedMemConfig
  {"cudaDeviceGetSharedMemConfig",                            {"hipDeviceGetSharedMemConfig",                            "", CONV_DEVICE, API_RUNTIME}},
  // cuCtxGetStreamPriorityRange
  {"cudaDeviceGetStreamPriorityRange",                        {"hipDeviceGetStreamPriorityRange",                        "", CONV_DEVICE, API_RUNTIME}},
  // no analogue
  {"cudaDeviceReset",                                         {"hipDeviceReset",                                         "", CONV_DEVICE, API_RUNTIME}},
  // no analogue
  {"cudaDeviceSetCacheConfig",                                {"hipDeviceSetCacheConfig",                                "", CONV_DEVICE, API_RUNTIME}},
  // cuCtxSetLimit
  {"cudaDeviceSetLimit",                                      {"hipDeviceSetLimit",                                      "", CONV_DEVICE, API_RUNTIME}},
  // cuCtxSetSharedMemConfig
  {"cudaDeviceSetSharedMemConfig",                            {"hipDeviceSetSharedMemConfig",                            "", CONV_DEVICE, API_RUNTIME}},
  // cuCtxSynchronize
  {"cudaDeviceSynchronize",                                   {"hipDeviceSynchronize",                                   "", CONV_DEVICE, API_RUNTIME}},
  // cuDeviceGet
  // NOTE: cuDeviceGet has no attr: int ordinal
  {"cudaGetDevice",                                           {"hipGetDevice",                                           "", CONV_DEVICE, API_RUNTIME}},
  // cuDeviceGetCount
  {"cudaGetDeviceCount",                                      {"hipGetDeviceCount",                                      "", CONV_DEVICE, API_RUNTIME}},
  // cuCtxGetFlags
  // TODO: rename to hipGetDeviceFlags
  {"cudaGetDeviceFlags",                                      {"hipCtxGetFlags",                                         "", CONV_DEVICE, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuDeviceGetProperties due to different attributes: CUdevprop and cudaDeviceProp
  {"cudaGetDeviceProperties",                                 {"hipGetDeviceProperties",                                 "", CONV_DEVICE, API_RUNTIME}},
  // cuIpcCloseMemHandle
  {"cudaIpcCloseMemHandle",                                   {"hipIpcCloseMemHandle",                                   "", CONV_DEVICE, API_RUNTIME}},
  // cuIpcGetEventHandle
  {"cudaIpcGetEventHandle",                                   {"hipIpcGetEventHandle",                                   "", CONV_DEVICE, API_RUNTIME}},
  // cuIpcGetMemHandle
  {"cudaIpcGetMemHandle",                                     {"hipIpcGetMemHandle",                                     "", CONV_DEVICE, API_RUNTIME}},
  // cuIpcOpenEventHandle
  {"cudaIpcOpenEventHandle",                                  {"hipIpcOpenEventHandle",                                  "", CONV_DEVICE, API_RUNTIME}},
  // cuIpcOpenMemHandle
  {"cudaIpcOpenMemHandle",                                    {"hipIpcOpenMemHandle",                                    "", CONV_DEVICE, API_RUNTIME}},
  // no analogue
  {"cudaSetDevice",                                           {"hipSetDevice",                                           "", CONV_DEVICE, API_RUNTIME}},
  // cuCtxGetFlags
  {"cudaSetDeviceFlags",                                      {"hipSetDeviceFlags",                                      "", CONV_DEVICE, API_RUNTIME}},
  // no analogue
  {"cudaSetValidDevices",                                     {"hipSetValidDevices",                                     "", CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.2. Thread Management [DEPRECATED]
  // no analogue
  {"cudaThreadExit",                                          {"hipDeviceReset",                                         "", CONV_THREAD, API_RUNTIME}},
  // no analogue
  {"cudaThreadGetCacheConfig",                                {"hipDeviceGetCacheConfig",                                "", CONV_THREAD, API_RUNTIME}},
  // no analogue
  {"cudaThreadGetLimit",                                      {"hipThreadGetLimit",                                      "", CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaThreadSetCacheConfig",                                {"hipDeviceSetCacheConfig",                                "", CONV_THREAD, API_RUNTIME}},
  // no analogue
  {"cudaThreadSetLimit",                                      {"hipThreadSetLimit",                                      "", CONV_THREAD, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuCtxSynchronize
  {"cudaThreadSynchronize",                                   {"hipDeviceSynchronize",                                   "", CONV_THREAD, API_RUNTIME}},

  // 5.3. Error Handling
  // no analogue
  // NOTE: cudaGetErrorName and cuGetErrorName have different signatures
  {"cudaGetErrorName",                                        {"hipGetErrorName",                                        "", CONV_ERROR, API_RUNTIME}},
  // no analogue
  // NOTE: cudaGetErrorString and cuGetErrorString have different signatures
  {"cudaGetErrorString",                                      {"hipGetErrorString",                                      "", CONV_ERROR, API_RUNTIME}},
  // no analogue
  {"cudaGetLastError",                                        {"hipGetLastError",                                        "", CONV_ERROR, API_RUNTIME}},
  // no analogue
  {"cudaPeekAtLastError",                                     {"hipPeekAtLastError",                                     "", CONV_ERROR, API_RUNTIME}},

  // 5.4. Stream Management
  // cuStreamAddCallback
  {"cudaStreamAddCallback",                                   {"hipStreamAddCallback",                                   "", CONV_STREAM, API_RUNTIME}},
  // cuStreamAttachMemAsync
  {"cudaStreamAttachMemAsync",                                {"hipStreamAttachMemAsync",                                "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuStreamBeginCapture
  {"cudaStreamBeginCapture",                                  {"hipStreamBeginCapture",                                  "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuStreamCreate due to different signatures
  {"cudaStreamCreate",                                        {"hipStreamCreate",                                        "", CONV_STREAM, API_RUNTIME}},
  // cuStreamCreate
  {"cudaStreamCreateWithFlags",                               {"hipStreamCreateWithFlags",                               "", CONV_STREAM, API_RUNTIME}},
  // cuStreamCreateWithPriority
  {"cudaStreamCreateWithPriority",                            {"hipStreamCreateWithPriority",                            "", CONV_STREAM, API_RUNTIME}},
  // cuStreamDestroy
  {"cudaStreamDestroy",                                       {"hipStreamDestroy",                                       "", CONV_STREAM, API_RUNTIME}},
  // cuStreamEndCapture
  {"cudaStreamEndCapture",                                    {"hipStreamEndCapture",                                    "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuStreamGetFlags
  {"cudaStreamGetFlags",                                      {"hipStreamGetFlags",                                      "", CONV_STREAM, API_RUNTIME}},
  // cuStreamGetPriority
  {"cudaStreamGetPriority",                                   {"hipStreamGetPriority",                                   "", CONV_STREAM, API_RUNTIME}},
  // cuStreamIsCapturing
  {"cudaStreamIsCapturing",                                   {"hipStreamIsCapturing",                                   "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuStreamGetCaptureInfo
  {"cudaStreamGetCaptureInfo",                                {"hipStreamGetCaptureInfo",                                "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuStreamQuery
  {"cudaStreamQuery",                                         {"hipStreamQuery",                                         "", CONV_STREAM, API_RUNTIME}},
  // cuStreamSynchronize
  {"cudaStreamSynchronize",                                   {"hipStreamSynchronize",                                   "", CONV_STREAM, API_RUNTIME}},
  // cuStreamWaitEvent
  {"cudaStreamWaitEvent",                                     {"hipStreamWaitEvent",                                     "", CONV_STREAM, API_RUNTIME}},
  // cuThreadExchangeStreamCaptureMode
  {"cudaThreadExchangeStreamCaptureMode",                     {"hipThreadExchangeStreamCaptureMode",                     "", CONV_STREAM, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.5.Event Management
  // no analogue
  // NOTE: Not equal to cuEventCreate due to different signatures
  {"cudaEventCreate",                                         {"hipEventCreate",                                         "", CONV_EVENT, API_RUNTIME}},
  // cuEventCreate
  {"cudaEventCreateWithFlags",                                {"hipEventCreateWithFlags",                                "", CONV_EVENT, API_RUNTIME}},
  // cuEventDestroy
  {"cudaEventDestroy",                                        {"hipEventDestroy",                                        "", CONV_EVENT, API_RUNTIME}},
  // cuEventElapsedTime
  {"cudaEventElapsedTime",                                    {"hipEventElapsedTime",                                    "", CONV_EVENT, API_RUNTIME}},
  // cuEventQuery
  {"cudaEventQuery",                                          {"hipEventQuery",                                          "", CONV_EVENT, API_RUNTIME}},
  // cuEventRecord
  {"cudaEventRecord",                                         {"hipEventRecord",                                         "", CONV_EVENT, API_RUNTIME}},
  // cuEventSynchronize
  {"cudaEventSynchronize",                                    {"hipEventSynchronize",                                    "", CONV_EVENT, API_RUNTIME}},

  // 5.6. External Resource Interoperability
  // cuDestroyExternalMemory
  {"cudaDestroyExternalMemory",                               {"hipDestroyExternalMemory",                               "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuDestroyExternalSemaphore
  {"cudaDestroyExternalSemaphore",                            {"hipDestroyExternalSemaphore",                            "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuExternalMemoryGetMappedBuffer
  {"cudaExternalMemoryGetMappedBuffer",                       {"hipExternalMemoryGetMappedBuffer",                       "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuExternalMemoryGetMappedMipmappedArray
  {"cudaExternalMemoryGetMappedMipmappedArray",               {"hipExternalMemoryGetMappedMipmappedArray",               "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuImportExternalMemory
  {"cudaImportExternalMemory",                                {"hipImportExternalMemory",                                "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuImportExternalSemaphore
  {"cudaImportExternalSemaphore",                             {"hipImportExternalSemaphore",                             "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuSignalExternalSemaphoresAsync
  {"cudaSignalExternalSemaphoresAsync",                       {"hipSignalExternalSemaphoresAsync",                       "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuWaitExternalSemaphoresAsync
  {"cudaWaitExternalSemaphoresAsync",                         {"hipWaitExternalSemaphoresAsync",                         "", CONV_EXT_RES, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.7. Execution Control
  // no analogue
  {"cudaFuncGetAttributes",                                   {"hipFuncGetAttributes",                                   "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaFuncSetAttribute",                                    {"hipFuncSetAttribute",                                    "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuFuncSetCacheConfig due to different signatures
  {"cudaFuncSetCacheConfig",                                  {"hipFuncSetCacheConfig",                                  "", CONV_DEVICE, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuFuncSetSharedMemConfig due to different signatures
  {"cudaFuncSetSharedMemConfig",                              {"hipFuncSetSharedMemConfig",                              "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGetParameterBuffer",                                  {"hipGetParameterBuffer",                                  "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGetParameterBufferV2",                                {"hipGetParameterBufferV2",                                "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuLaunchCooperativeKernel due to different signatures
  {"cudaLaunchCooperativeKernel",                             {"hipLaunchCooperativeKernel",                             "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuLaunchCooperativeKernelMultiDevice due to different signatures
  {"cudaLaunchCooperativeKernelMultiDevice",                  {"hipLaunchCooperativeKernelMultiDevice",                  "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuLaunchHostFunc
  {"cudaLaunchHostFunc",                                      {"hipLaunchHostFunc",                                      "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuLaunchKernel due to different signatures
  {"cudaLaunchKernel",                                        {"hipLaunchKernel",                                        "", CONV_EXECUTION, API_RUNTIME}},
  // no analogue
  {"cudaSetDoubleForDevice",                                  {"hipSetDoubleForDevice",                                  "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaSetDoubleForHost",                                    {"hipSetDoubleForHost",                                    "", CONV_EXECUTION, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.8. Occupancy
  // 
  {"cudaOccupancyMaxActiveBlocksPerMultiprocessor",           {"hipOccupancyMaxActiveBlocksPerMultiprocessor",           "", CONV_OCCUPANCY, API_RUNTIME}},
  // cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  {"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",  {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",  "", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuOccupancyMaxPotentialBlockSize
  {"cudaOccupancyMaxPotentialBlockSize",                      {"hipOccupancyMaxPotentialBlockSize",                      "", CONV_OCCUPANCY, API_RUNTIME}},
  // cuOccupancyMaxPotentialBlockSizeWithFlags
  {"cudaOccupancyMaxPotentialBlockSizeWithFlags",             {"hipOccupancyMaxPotentialBlockSizeWithFlags",             "", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaOccupancyMaxPotentialBlockSizeVariableSMem",          {"hipOccupancyMaxPotentialBlockSizeVariableSMem",          "", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", {"hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags", "", CONV_OCCUPANCY, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.9. Execution Control [DEPRECATED]
  // no analogue
  {"cudaConfigureCall",                                       {"hipConfigureCall",                                       "", CONV_EXECUTION, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cudaLaunch due to different signatures
  {"cudaLaunch",                                              {"hipLaunchByPtr",                                         "", CONV_EXECUTION, API_RUNTIME}},
  // no analogue
  {"cudaSetupArgument",                                       {"hipSetupArgument",                                       "", CONV_EXECUTION, API_RUNTIME}},

  // 5.10. Memory Management
  // no analogue
  {"cudaArrayGetInfo",                                        {"hipArrayGetInfo",                                        "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuMemFree
  {"cudaFree",                                                {"hipFree",                                                "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaFreeArray",                                           {"hipFreeArray",                                           "", CONV_MEMORY, API_RUNTIME}},
  // cuMemFreeHost
  {"cudaFreeHost",                                            {"hipHostFree",                                            "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayDestroy due to different signatures
  {"cudaFreeMipmappedArray",                                  {"hipFreeMipmappedArray",                                  "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayGetLevel due to different signatures
  {"cudaGetMipmappedArrayLevel",                              {"hipGetMipmappedArrayLevel",                              "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGetSymbolAddress",                                    {"hipGetSymbolAddress",                                    "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaGetSymbolSize",                                       {"hipGetSymbolSize",                                       "", CONV_MEMORY, API_RUNTIME}},
  // cuMemHostAlloc
  // NOTE: hipHostAlloc deprecated - use hipHostMalloc instead
  {"cudaHostAlloc",                                           {"hipHostMalloc",                                          "", CONV_MEMORY, API_RUNTIME}},
  // cuMemHostGetDevicePointer
  {"cudaHostGetDevicePointer",                                {"hipHostGetDevicePointer",                                "", CONV_MEMORY, API_RUNTIME}},
  // cuMemHostGetFlags
  {"cudaHostGetFlags",                                        {"hipHostGetFlags",                                        "", CONV_MEMORY, API_RUNTIME}},
  // cuMemHostRegister
  {"cudaHostRegister",                                        {"hipHostRegister",                                        "", CONV_MEMORY, API_RUNTIME}},
  // cuMemHostUnregister
  {"cudaHostUnregister",                                      {"hipHostUnregister",                                      "", CONV_MEMORY, API_RUNTIME}},
  // cuMemAlloc
  {"cudaMalloc",                                              {"hipMalloc",                                              "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMalloc3D",                                            {"hipMalloc3D",                                            "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMalloc3DArray",                                       {"hipMalloc3DArray",                                       "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMallocArray",                                         {"hipMallocArray",                                         "", CONV_MEMORY, API_RUNTIME}},
  // cuMemHostAlloc
  {"cudaMallocHost",                                          {"hipHostMalloc",                                          "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMallocManaged",                                       {"hipMallocManaged",                                       "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMipmappedArrayCreate due to different signatures
  {"cudaMallocMipmappedArray",                                {"hipMallocMipmappedArray",                                "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemAllocPitch due to different signatures
  {"cudaMallocPitch",                                         {"hipMallocPitch",                                         "", CONV_MEMORY, API_RUNTIME}},
  // cuMemAdvise
  {"cudaMemAdvise",                                           {"hipMemAdvise",                                           "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy due to different signatures
  {"cudaMemcpy",                                              {"hipMemcpy",                                              "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuMemcpy2D due to different signatures
  {"cudaMemcpy2D",                                            {"hipMemcpy2D",                                            "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpy2DArrayToArray",                                {"hipMemcpy2DArrayToArray",                                "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy2DAsync due to different signatures
  {"cudaMemcpy2DAsync",                                       {"hipMemcpy2DAsync",                                       "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpy2DFromArray",                                   {"hipMemcpy2DFromArray",                                   "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaMemcpy2DFromArrayAsync",                              {"hipMemcpy2DFromArrayAsync",                              "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaMemcpy2DToArray",                                     {"hipMemcpy2DToArray",                                     "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpy2DToArrayAsync",                                {"hipMemcpy2DToArrayAsync",                                "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3D due to different signatures
  {"cudaMemcpy3D",                                            {"hipMemcpy3D",                                            "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3DAsync due to different signatures
  {"cudaMemcpy3DAsync",                                       {"hipMemcpy3DAsync",                                       "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3DPeer due to different signatures
  {"cudaMemcpy3DPeer",                                        {"hipMemcpy3DPeer",                                        "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpy3DPeerAsync due to different signatures
  {"cudaMemcpy3DPeerAsync",                                   {"hipMemcpy3DPeerAsync",                                   "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpyAtoA due to different signatures
  {"cudaMemcpyArrayToArray",                                  {"hipMemcpyArrayToArray",                                  "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuMemcpyAsync due to different signatures
  {"cudaMemcpyAsync",                                         {"hipMemcpyAsync",                                         "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpyFromArray",                                     {"hipMemcpyFromArray",                                     "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpyFromArrayAsync",                                {"hipMemcpyFromArrayAsync",                                "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaMemcpyFromSymbol",                                    {"hipMemcpyFromSymbol",                                    "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpyFromSymbolAsync",                               {"hipMemcpyFromSymbolAsync",                               "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuMemcpyPeer due to different signatures
  {"cudaMemcpyPeer",                                          {"hipMemcpyPeer",                                          "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuMemcpyPeerAsync due to different signatures
  {"cudaMemcpyPeerAsync",                                     {"hipMemcpyPeerAsync",                                     "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpyToArray",                                       {"hipMemcpyToArray",                                       "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpyToArrayAsync",                                  {"hipMemcpyToArrayAsync",                                  "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpyToSymbol",                                      {"hipMemcpyToSymbol",                                      "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemcpyToSymbolAsync",                                 {"hipMemcpyToSymbolAsync",                                 "", CONV_MEMORY, API_RUNTIME}},
  // cuMemGetInfo
  {"cudaMemGetInfo",                                          {"hipMemGetInfo",                                          "", CONV_MEMORY, API_RUNTIME}},
  // TODO: double check cuMemPrefetchAsync
  {"cudaMemPrefetchAsync",                                    {"hipMemPrefetchAsync",                                    "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuMemRangeGetAttribute
  {"cudaMemRangeGetAttribute",                                {"hipMemRangeGetAttribute",                                "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuMemRangeGetAttributes
  {"cudaMemRangeGetAttributes",                               {"hipMemRangeGetAttributes",                               "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuMemsetD32
  {"cudaMemset",                                              {"hipMemset",                                              "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemset2D",                                            {"hipMemset2D",                                            "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemset2DAsync",                                       {"hipMemset2DAsync",                                       "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"cudaMemset3D",                                            {"hipMemset3D",                                            "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaMemset3DAsync",                                       {"hipMemset3DAsync",                                       "", CONV_MEMORY, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuMemsetD32Async
  {"cudaMemsetAsync",                                         {"hipMemsetAsync",                                         "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"make_cudaExtent",                                         {"make_hipExtent",                                         "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"make_cudaPitchedPtr",                                     {"make_hipPitchedPtr",                                     "", CONV_MEMORY, API_RUNTIME}},
  // no analogue
  {"make_cudaPos",                                            {"make_hipPos",                                            "", CONV_MEMORY, API_RUNTIME}},

  // 5.11.Unified Addressing
  // no analogue
  // NOTE: Not equal to cuPointerGetAttributes due to different signatures
  {"cudaPointerGetAttributes",                                {"hipPointerGetAttributes",                                "", CONV_ADDRESSING, API_RUNTIME}},

  // 5.12. Peer Device Memory Access
  // cuDeviceCanAccessPeer
  {"cudaDeviceCanAccessPeer",                                 {"hipDeviceCanAccessPeer",                                 "", CONV_PEER, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuCtxDisablePeerAccess due to different signatures
  {"cudaDeviceDisablePeerAccess",                             {"hipDeviceDisablePeerAccess",                             "", CONV_PEER, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuCtxEnablePeerAccess due to different signatures
  {"cudaDeviceEnablePeerAccess",                              {"hipDeviceEnablePeerAccess",                              "", CONV_PEER, API_RUNTIME}},

  // 5.13. OpenGL Interoperability
  // cuGLGetDevices
  {"cudaGLGetDevices",                                        {"hipGLGetDevices",                                        "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsGLRegisterBuffer
  {"cudaGraphicsGLRegisterBuffer",                            {"hipGraphicsGLRegisterBuffer",                            "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsGLRegisterImage
  {"cudaGraphicsGLRegisterImage",                             {"hipGraphicsGLRegisterImage",                             "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuWGLGetDevice
  {"cudaWGLGetDevice",                                        {"hipWGLGetDevice",                                        "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.14. OpenGL Interoperability [DEPRECATED]
  // no analogue
  // NOTE: Not equal to cuGLMapBufferObject due to different signatures
  {"cudaGLMapBufferObject",                                   {"hipGLMapBufferObject",                                   "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cuGLMapBufferObjectAsync due to different signatures
  {"cudaGLMapBufferObjectAsync",                              {"hipGLMapBufferObjectAsync",                              "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGLRegisterBufferObject
  {"cudaGLRegisterBufferObject",                              {"hipGLRegisterBufferObject",                              "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGLSetBufferObjectMapFlags
  {"cudaGLSetBufferObjectMapFlags",                           {"hipGLSetBufferObjectMapFlags",                           "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaGLSetGLDevice",                                       {"hipGLSetGLDevice",                                       "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGLUnmapBufferObject
  {"cudaGLUnmapBufferObject",                                 {"hipGLUnmapBufferObject",                                 "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGLUnmapBufferObjectAsync
  {"cudaGLUnmapBufferObjectAsync",                            {"hipGLUnmapBufferObjectAsync",                            "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGLUnregisterBufferObject
  {"cudaGLUnregisterBufferObject",                            {"hipGLUnregisterBufferObject",                            "", CONV_OPENGL, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.15. Direct3D 9 Interoperability
  // cuD3D9GetDevice
  {"cudaD3D9GetDevice",                                       {"hipD3D9GetDevice",                                       "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9GetDevices
  {"cudaD3D9GetDevices",                                      {"hipD3D9GetDevices",                                      "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9GetDirect3DDevice
  {"cudaD3D9GetDirect3DDevice",                               {"hipD3D9GetDirect3DDevice",                               "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaD3D9SetDirect3DDevice",                               {"hipD3D9SetDirect3DDevice",                               "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsD3D9RegisterResource
  {"cudaGraphicsD3D9RegisterResource",                        {"hipGraphicsD3D9RegisterResource",                        "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.16.Direct3D 9 Interoperability[DEPRECATED]
  // cuD3D9MapResources
  {"cudaD3D9MapResources",                                    {"hipD3D9MapResources",                                    "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9RegisterResource
  {"cudaD3D9RegisterResource",                                {"hipD3D9RegisterResource",                                "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9ResourceGetMappedArray
  {"cudaD3D9ResourceGetMappedArray",                          {"hipD3D9ResourceGetMappedArray",                          "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaD3D9ResourceGetMappedPitch
  {"cudaD3D9ResourceGetMappedPitch",                          {"hipD3D9ResourceGetMappedPitch",                          "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9ResourceGetMappedPointer
  {"cudaD3D9ResourceGetMappedPointer",                        {"hipD3D9ResourceGetMappedPointer",                        "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9ResourceGetMappedSize
  {"cudaD3D9ResourceGetMappedSize",                           {"hipD3D9ResourceGetMappedSize",                           "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9ResourceGetSurfaceDimensions
  {"cudaD3D9ResourceGetSurfaceDimensions",                    {"hipD3D9ResourceGetSurfaceDimensions",                    "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9ResourceSetMapFlags
  {"cudaD3D9ResourceSetMapFlags",                             {"hipD3D9ResourceSetMapFlags",                             "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9UnmapResources
  {"cudaD3D9UnmapResources",                                  {"hipD3D9UnmapResources",                                  "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D9UnregisterResource
  {"cudaD3D9UnregisterResource",                              {"hipD3D9UnregisterResource",                              "", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.17. Direct3D 10 Interoperability
  // cuD3D10GetDevice
  {"cudaD3D10GetDevice",                                      {"hipD3D10GetDevice",                                      "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10GetDevices
  {"cudaD3D10GetDevices",                                     {"hipD3D10GetDevices",                                     "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsD3D10RegisterResource
  {"cudaGraphicsD3D10RegisterResource",                       {"hipGraphicsD3D10RegisterResource",                       "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.18. Direct3D 10 Interoperability [DEPRECATED]
  // cudaD3D10GetDirect3DDevice
  {"cudaD3D10GetDirect3DDevice",                              {"hipD3D10GetDirect3DDevice",                              "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10MapResources
  {"cudaD3D10MapResources",                                   {"hipD3D10MapResources",                                   "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10RegisterResource
  {"cudaD3D10RegisterResource",                               {"hipD3D10RegisterResource",                               "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10ResourceGetMappedArray
  {"cudaD3D10ResourceGetMappedArray",                         {"hipD3D10ResourceGetMappedArray",                         "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaD3D10ResourceGetMappedPitch
  {"cudaD3D10ResourceGetMappedPitch",                         {"hipD3D10ResourceGetMappedPitch",                         "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10ResourceGetMappedPointer
  {"cudaD3D10ResourceGetMappedPointer",                       {"hipD3D10ResourceGetMappedPointer",                       "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10ResourceGetMappedSize
  {"cudaD3D10ResourceGetMappedSize",                          {"hipD3D10ResourceGetMappedSize",                          "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10ResourceGetSurfaceDimensions
  {"cudaD3D10ResourceGetSurfaceDimensions",                   {"hipD3D10ResourceGetSurfaceDimensions",                   "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10ResourceSetMapFlags
  {"cudaD3D10ResourceSetMapFlags",                            {"hipD3D10ResourceSetMapFlags",                            "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaD3D10SetDirect3DDevice",                              {"hipD3D10SetDirect3DDevice",                              "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10UnmapResources
  {"cudaD3D10UnmapResources",                                 {"hipD3D10UnmapResources",                                 "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D10UnregisterResource
  {"cudaD3D10UnregisterResource",                             {"hipD3D10UnregisterResource",                             "", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.19. Direct3D 11 Interoperability
  // cuD3D11GetDevice
  {"cudaD3D11GetDevice",                                      {"hipD3D11GetDevice",                                      "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuD3D11GetDevices
  {"cudaD3D11GetDevices",                                     {"hipD3D11GetDevices",                                     "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsD3D11RegisterResource
  {"cudaGraphicsD3D11RegisterResource",                       {"hipGraphicsD3D11RegisterResource",                       "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.20. Direct3D 11 Interoperability [DEPRECATED]
  // cuD3D11GetDirect3DDevice
  {"cudaD3D11GetDirect3DDevice",                              {"hipD3D11GetDirect3DDevice",                              "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaD3D11SetDirect3DDevice",                              {"hipD3D11SetDirect3DDevice",                              "", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.21. VDPAU Interoperability
  // cuGraphicsVDPAURegisterOutputSurface
  {"cudaGraphicsVDPAURegisterOutputSurface",                  {"hipGraphicsVDPAURegisterOutputSurface",                  "", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsVDPAURegisterVideoSurface
  {"cudaGraphicsVDPAURegisterVideoSurface",                   {"hipGraphicsVDPAURegisterVideoSurface",                   "", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuVDPAUGetDevice
  {"cudaVDPAUGetDevice",                                      {"hipVDPAUGetDevice",                                      "", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"cudaVDPAUSetVDPAUDevice",                                 {"hipVDPAUSetDevice",                                      "", CONV_VDPAU, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.22. EGL Interoperability
  // cuEGLStreamConsumerAcquireFrame
  {"cudaEGLStreamConsumerAcquireFrame",                       {"hipEGLStreamConsumerAcquireFrame",                       "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerConnect
  {"cudaEGLStreamConsumerConnect",                            {"hipEGLStreamConsumerConnect",                            "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerConnectWithFlags
  {"cudaEGLStreamConsumerConnectWithFlags",                   {"hipEGLStreamConsumerConnectWithFlags",                   "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerDisconnect
  {"cudaEGLStreamConsumerDisconnect",                         {"hipEGLStreamConsumerDisconnect",                         "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuEGLStreamConsumerReleaseFrame
  {"cudaEGLStreamConsumerReleaseFrame",                       {"hipEGLStreamConsumerReleaseFrame",                       "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerConnect
  {"cudaEGLStreamProducerConnect",                            {"hipEGLStreamProducerConnect",                            "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerDisconnect
  {"cudaEGLStreamProducerDisconnect",                         {"hipEGLStreamProducerDisconnect",                         "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerPresentFrame
  {"cudaEGLStreamProducerPresentFrame",                       {"hipEGLStreamProducerPresentFrame",                       "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuEGLStreamProducerReturnFrame
  {"cudaEGLStreamProducerReturnFrame",                        {"hipEGLStreamProducerReturnFrame",                        "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuEventCreateFromEGLSync
  {"cudaEventCreateFromEGLSync",                              {"hipEventCreateFromEGLSync",                              "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsEGLRegisterImage
  {"cudaGraphicsEGLRegisterImage",                            {"hipGraphicsEGLRegisterImage",                            "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsResourceGetMappedEglFrame
  {"cudaGraphicsResourceGetMappedEglFrame",                   {"hipGraphicsResourceGetMappedEglFrame",                   "", CONV_EGL, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.23. Graphics Interoperability
  // cuGraphicsMapResources
  {"cudaGraphicsMapResources",                                {"hipGraphicsMapResources",                                "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsResourceGetMappedMipmappedArray
  {"cudaGraphicsResourceGetMappedMipmappedArray",             {"hipGraphicsResourceGetMappedMipmappedArray",             "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsResourceGetMappedPointer
  {"cudaGraphicsResourceGetMappedPointer",                    {"hipGraphicsResourceGetMappedPointer",                    "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsResourceSetMapFlags
  {"cudaGraphicsResourceSetMapFlags",                         {"hipGraphicsResourceSetMapFlags",                         "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsSubResourceGetMappedArray
  {"cudaGraphicsSubResourceGetMappedArray",                   {"hipGraphicsSubResourceGetMappedArray",                   "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsUnmapResources
  {"cudaGraphicsUnmapResources",                              {"hipGraphicsUnmapResources",                              "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphicsUnregisterResource
  {"cudaGraphicsUnregisterResource",                          {"hipGraphicsUnregisterResource",                          "", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.24. Texture Reference Management
  // no analogue
  {"cudaBindTexture",                                         {"hipBindTexture",                                         "", CONV_TEXTURE, API_RUNTIME}},
  // no analogue
  {"cudaBindTexture2D",                                       {"hipBindTexture2D",                                       "", CONV_TEXTURE, API_RUNTIME}},
  // no analogue
  {"cudaBindTextureToArray",                                  {"hipBindTextureToArray",                                  "", CONV_TEXTURE, API_RUNTIME}},
  // no analogue
  // NOTE: Unsupported yet on NVCC path
  {"cudaBindTextureToMipmappedArray",                         {"hipBindTextureToMipmappedArray",                         "", CONV_TEXTURE, API_RUNTIME}},
  // no analogue
  {"cudaCreateChannelDesc",                                   {"hipCreateChannelDesc",                                   "", CONV_TEXTURE, API_RUNTIME}},
  // no analogue
  {"cudaGetChannelDesc",                                      {"hipGetChannelDesc",                                      "", CONV_TEXTURE, API_RUNTIME}},
  // no analogue
  {"cudaGetTextureAlignmentOffset",                           {"hipGetTextureAlignmentOffset",                           "", CONV_TEXTURE, API_RUNTIME}},
  // TODO: double check cuModuleGetTexRef
  // NOTE: Unsupported yet on NVCC path
  {"cudaGetTextureReference",                                 {"hipGetTextureReference",                                 "", CONV_TEXTURE, API_RUNTIME}},
  // no analogue
  {"cudaUnbindTexture",                                       {"hipUnbindTexture",                                       "", CONV_TEXTURE, API_RUNTIME}},

  // 5.25. Surface Reference Management
  // no analogue
  {"cudaBindSurfaceToArray",                                  {"hipBindSurfaceToArray",                                  "", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
  // TODO: double check cuModuleGetSurfRef
  {"cudaGetSurfaceReference",                                 {"hipGetSurfaceReference",                                 "", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.26. Texture Object Management
  // no analogue
  // NOTE: Not equal to cuTexObjectCreate due to different signatures
  {"cudaCreateTextureObject",                                 {"hipCreateTextureObject",                                 "", CONV_TEXTURE, API_RUNTIME}},
  // cuTexObjectDestroy
  {"cudaDestroyTextureObject",                                {"hipDestroyTextureObject",                                "", CONV_TEXTURE, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuTexObjectGetResourceDesc due to different signatures
  {"cudaGetTextureObjectResourceDesc",                        {"hipGetTextureObjectResourceDesc",                        "", CONV_TEXTURE, API_RUNTIME}},
  // cuTexObjectGetResourceViewDesc
  {"cudaGetTextureObjectResourceViewDesc",                    {"hipGetTextureObjectResourceViewDesc",                    "", CONV_TEXTURE, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectTextureDesc due to different signatures
  {"cuTexObjectGetTextureDesc",                               {"hipGetTextureObjectTextureDesc",                         "", CONV_TEXTURE, API_RUNTIME}},

  // 5.27. Surface Object Management
  // no analogue
  // NOTE: Not equal to cuSurfObjectCreate due to different signatures
  {"cudaCreateSurfaceObject",                                 {"hipCreateSurfaceObject",                                 "", CONV_SURFACE, API_RUNTIME}},
  // cuSurfObjectDestroy
  {"cudaDestroySurfaceObject",                                {"hipDestroySurfaceObject",                                "", CONV_SURFACE, API_RUNTIME}},
  // no analogue
  // NOTE: Not equal to cuSurfObjectGetResourceDesc due to different signatures
  {"cudaGetSurfaceObjectResourceDesc",                        {"hipGetSurfaceObjectResourceDesc",                        "", CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.28.Version Management
  // cuDriverGetVersion
  {"cudaDriverGetVersion",                                    {"hipDriverGetVersion",                                    "", CONV_VERSION, API_RUNTIME}},
  // no analogue
  {"cudaRuntimeGetVersion",                                   {"hipRuntimeGetVersion",                                   "", CONV_VERSION, API_RUNTIME}},

  // 5.29. Graph Management
  // cuGraphAddChildGraphNode
  {"cudaGraphAddChildGraphNode",                              {"hipGraphAddChildGraphNode",                              "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphAddDependencies
  {"cudaGraphAddDependencies",                                {"hipGraphAddDependencies",                                "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphAddEmptyNode
  {"cudaGraphAddEmptyNode",                                   {"hipGraphAddEmptyNode",                                   "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphAddHostNode
  {"cudaGraphAddHostNode",                                    {"hipGraphAddHostNode",                                    "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphAddKernelNode
  {"cudaGraphAddKernelNode",                                  {"hipGraphAddKernelNode",                                  "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphAddMemcpyNode
  {"cudaGraphAddMemcpyNode",                                  {"hipGraphAddMemcpyNode",                                  "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphAddMemsetNode
  {"cudaGraphAddMemsetNode",                                  {"hipGraphAddMemsetNode",                                  "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphChildGraphNodeGetGraph
  {"cudaGraphChildGraphNodeGetGraph",                         {"hipGraphChildGraphNodeGetGraph",                         "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphClone
  {"cudaGraphClone",                                          {"hipGraphClone",                                          "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphCreate
  {"cudaGraphCreate",                                         {"hipGraphCreate",                                         "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphDestroy
  {"cudaGraphDestroy",                                        {"hipGraphDestroy",                                        "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphDestroyNode
  {"cudaGraphDestroyNode",                                    {"hipGraphDestroyNode",                                    "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphExecDestroy
  {"cudaGraphExecDestroy",                                    {"hipGraphExecDestroy",                                    "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphGetEdges
  {"cudaGraphGetEdges",                                       {"hipGraphGetEdges",                                       "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphGetNodes
  {"cudaGraphGetNodes",                                       {"hipGraphGetNodes",                                       "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphGetRootNodes
  {"cudaGraphGetRootNodes",                                   {"hipGraphGetRootNodes",                                   "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphHostNodeGetParams
  {"cudaGraphHostNodeGetParams",                              {"hipGraphHostNodeGetParams",                              "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphHostNodeSetParams
  {"cudaGraphHostNodeSetParams",                              {"hipGraphHostNodeSetParams",                              "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphInstantiate
  {"cudaGraphInstantiate",                                    {"hipGraphInstantiate",                                    "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphExecKernelNodeSetParams
  {"cudaGraphExecKernelNodeSetParams",                        {"hipGraphExecKernelNodeSetParams",                        "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeGetParams
  {"cudaGraphKernelNodeGetParams",                            {"hipGraphKernelNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphKernelNodeSetParams
  {"cudaGraphKernelNodeSetParams",                            {"hipGraphKernelNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphLaunch
  {"cudaGraphLaunch",                                         {"hipGraphLaunch",                                         "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphMemcpyNodeGetParams
  {"cudaGraphMemcpyNodeGetParams",                            {"hipGraphMemcpyNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphMemcpyNodeSetParams
  {"cudaGraphMemcpyNodeSetParams",                            {"hipGraphMemcpyNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphMemsetNodeGetParams
  {"cudaGraphMemsetNodeGetParams",                            {"hipGraphMemsetNodeGetParams",                            "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphMemsetNodeSetParams
  {"cudaGraphMemsetNodeSetParams",                            {"hipGraphMemsetNodeSetParams",                            "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphNodeFindInClone
  {"cudaGraphNodeFindInClone",                                {"hipGraphNodeFindInClone",                                "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphNodeGetDependencies
  {"cudaGraphNodeGetDependencies",                            {"hipGraphNodeGetDependencies",                            "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphNodeGetDependentNodes
  {"cudaGraphNodeGetDependentNodes",                          {"hipGraphNodeGetDependentNodes",                          "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphNodeGetType
  {"cudaGraphNodeGetType",                                    {"hipGraphNodeGetType",                                    "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuGraphRemoveDependencies
  {"cudaGraphRemoveDependencies",                             {"hipGraphRemoveDependencies",                             "", CONV_GRAPH, API_RUNTIME, HIP_UNSUPPORTED}},

  // 5.32. Profiler Control
  // cuProfilerInitialize
  {"cudaProfilerInitialize",                                  {"hipProfilerInitialize",                                  "", CONV_PROFILER, API_RUNTIME, HIP_UNSUPPORTED}},
  // cuProfilerStart
  {"cudaProfilerStart",                                       {"hipProfilerStart",                                       "", CONV_PROFILER, API_RUNTIME}},
  // cuProfilerStop
  {"cudaProfilerStop",                                        {"hipProfilerStop",                                        "", CONV_PROFILER, API_RUNTIME}},
};

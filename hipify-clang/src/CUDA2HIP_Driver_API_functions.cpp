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

// Map of all CUDA Driver API functions
const std::map<llvm::StringRef, hipCounter> CUDA_DRIVER_FUNCTION_MAP{
  // 5.2. Error Handling
  // no analogue
  // NOTE: cudaGetErrorName and hipGetErrorName have different signature
  {"cuGetErrorName",                                       {"hipGetErrorName_",                                        "", CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: cudaGetErrorString and hipGetErrorString have different signature
  {"cuGetErrorString",                                     {"hipGetErrorString_",                                      "", CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.3. Initialization
  // no analogue
  {"cuInit",                                               {"hipInit",                                                 "", CONV_INIT, API_DRIVER}},

  // 5.4 Version Management
  // cudaDriverGetVersion
  {"cuDriverGetVersion",                                   {"hipDriverGetVersion",                                     "", CONV_VERSION, API_DRIVER}},

  // 5.5. Device Management
  // cudaGetDevice
  // NOTE: cudaGetDevice has additional attr: int ordinal
  {"cuDeviceGet",                                          {"hipGetDevice",                                            "", CONV_DEVICE, API_DRIVER}},
  // cudaDeviceGetAttribute
  {"cuDeviceGetAttribute",                                 {"hipDeviceGetAttribute",                                   "", CONV_DEVICE, API_DRIVER}},
  // cudaGetDeviceCount
  {"cuDeviceGetCount",                                     {"hipGetDeviceCount",                                       "", CONV_DEVICE, API_DRIVER}},
  // no analogue
  {"cuDeviceGetLuid",                                      {"hipDeviceGetLuid",                                        "", CONV_DEVICE, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuDeviceGetName",                                      {"hipDeviceGetName",                                        "", CONV_DEVICE, API_DRIVER}},
  // no analogue
  {"cuDeviceGetUuid",                                      {"hipDeviceGetUuid",                                        "", CONV_DEVICE, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuDeviceTotalMem",                                     {"hipDeviceTotalMem",                                       "", CONV_DEVICE, API_DRIVER}},
  {"cuDeviceTotalMem_v2",                                  {"hipDeviceTotalMem",                                       "", CONV_DEVICE, API_DRIVER}},

  // 5.6. Device Management [DEPRECATED]
  {"cuDeviceComputeCapability",                            {"hipDeviceComputeCapability",                              "", CONV_DEVICE, API_DRIVER}},
  {"cuDeviceGetProperties",                                {"hipGetDeviceProperties",                                  "", CONV_DEVICE, API_DRIVER}},

  // 5.7. Primary Context Management
  // no analogues
  {"cuDevicePrimaryCtxGetState",                           {"hipDevicePrimaryCtxGetState",                             "", CONV_CONTEXT, API_DRIVER}},
  {"cuDevicePrimaryCtxRelease",                            {"hipDevicePrimaryCtxRelease",                              "", CONV_CONTEXT, API_DRIVER}},
  {"cuDevicePrimaryCtxReset",                              {"hipDevicePrimaryCtxReset",                                "", CONV_CONTEXT, API_DRIVER}},
  {"cuDevicePrimaryCtxRetain",                             {"hipDevicePrimaryCtxRetain",                               "", CONV_CONTEXT, API_DRIVER}},
  {"cuDevicePrimaryCtxSetFlags",                           {"hipDevicePrimaryCtxSetFlags",                             "", CONV_CONTEXT, API_DRIVER}},

  // 5.8. Context Management
  // no analogues, except a few
  {"cuCtxCreate",                                          {"hipCtxCreate",                                            "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxCreate_v2",                                       {"hipCtxCreate",                                            "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxDestroy",                                         {"hipCtxDestroy",                                           "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxDestroy_v2",                                      {"hipCtxDestroy",                                           "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetApiVersion",                                   {"hipCtxGetApiVersion",                                     "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetCacheConfig",                                  {"hipCtxGetCacheConfig",                                    "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetCurrent",                                      {"hipCtxGetCurrent",                                        "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetDevice",                                       {"hipCtxGetDevice",                                         "", CONV_CONTEXT, API_DRIVER}},
  // cudaGetDeviceFlags
  // TODO: rename to hipGetDeviceFlags
  {"cuCtxGetFlags",                                        {"hipCtxGetFlags",                                          "", CONV_CONTEXT, API_DRIVER}},
  // cudaDeviceGetLimit
  {"cuCtxGetLimit",                                        {"hipDeviceGetLimit",                                       "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetSharedMemConfig",                              {"hipCtxGetSharedMemConfig",                                "", CONV_CONTEXT, API_DRIVER}},
  // cudaDeviceGetStreamPriorityRange
  {"cuCtxGetStreamPriorityRange",                          {"hipDeviceGetStreamPriorityRange",                         "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxPopCurrent",                                      {"hipCtxPopCurrent",                                        "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxPopCurrent_v2",                                   {"hipCtxPopCurrent",                                        "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxPushCurrent",                                     {"hipCtxPushCurrent",                                       "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxPushCurrent_v2",                                  {"hipCtxPushCurrent",                                       "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxSetCacheConfig",                                  {"hipCtxSetCacheConfig",                                    "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxSetCurrent",                                      {"hipCtxSetCurrent",                                        "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxSetLimit",                                        {"hipCtxSetLimit",                                          "", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuCtxSetSharedMemConfig",                              {"hipCtxSetSharedMemConfig",                                "", CONV_CONTEXT, API_DRIVER}},
  {"cuCtxSynchronize",                                     {"hipCtxSynchronize",                                       "", CONV_CONTEXT, API_DRIVER}},

  // 5.9. Context Management [DEPRECATED]
  // no analogues
  {"cuCtxAttach",                                          {"hipCtxAttach",                                            "", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuCtxDetach",                                          {"hipCtxDetach",                                            "", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.10. Module Management
  // no analogues
  {"cuLinkAddData",                                        {"hipLinkAddData",                                          "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkAddData_v2",                                     {"hipLinkAddData",                                          "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkAddFile",                                        {"hipLinkAddFile",                                          "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkAddFile_v2",                                     {"hipLinkAddFile",                                          "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkComplete",                                       {"hipLinkComplete",                                         "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkCreate",                                         {"hipLinkCreate",                                           "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkCreate_v2",                                      {"hipLinkCreate",                                           "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkDestroy",                                        {"hipLinkDestroy",                                          "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuModuleGetFunction",                                  {"hipModuleGetFunction",                                    "", CONV_MODULE, API_DRIVER}},
  {"cuModuleGetGlobal",                                    {"hipModuleGetGlobal",                                      "", CONV_MODULE, API_DRIVER}},
  {"cuModuleGetGlobal_v2",                                 {"hipModuleGetGlobal",                                      "", CONV_MODULE, API_DRIVER}},
  {"cuModuleGetSurfRef",                                   {"hipModuleGetSurfRef",                                     "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuModuleGetTexRef",                                    {"hipModuleGetTexRef",                                      "", CONV_MODULE, API_DRIVER}},
  {"cuModuleLoad",                                         {"hipModuleLoad",                                           "", CONV_MODULE, API_DRIVER}},
  {"cuModuleLoadData",                                     {"hipModuleLoadData",                                       "", CONV_MODULE, API_DRIVER}},
  {"cuModuleLoadDataEx",                                   {"hipModuleLoadDataEx",                                     "", CONV_MODULE, API_DRIVER}},
  {"cuModuleLoadFatBinary",                                {"hipModuleLoadFatBinary",                                  "", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuModuleUnload",                                       {"hipModuleUnload",                                         "", CONV_MODULE, API_DRIVER}},

  // 5.11. Memory Management
  // no analogue
  {"cuArray3DCreate",                                      {"hipArray3DCreate",                                        "", CONV_MEMORY, API_DRIVER}},
  {"cuArray3DCreate_v2",                                   {"hipArray3DCreate",                                        "", CONV_MEMORY, API_DRIVER}},
  {"cuArray3DGetDescriptor",                               {"hipArray3DGetDescriptor",                                 "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuArray3DGetDescriptor_v2",                            {"hipArray3DGetDescriptor",                                 "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuArrayCreate",                                        {"hipArrayCreate",                                          "", CONV_MEMORY, API_DRIVER}},
  {"cuArrayCreate_v2",                                     {"hipArrayCreate",                                          "", CONV_MEMORY, API_DRIVER}},
  {"cuArrayDestroy",                                       {"hipArrayDestroy",                                         "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuArrayGetDescriptor",                                 {"hipArrayGetDescriptor",                                   "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuArrayGetDescriptor_v2",                              {"hipArrayGetDescriptor",                                   "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaDeviceGetByPCIBusId
  {"cuDeviceGetByPCIBusId",                                {"hipDeviceGetByPCIBusId",                                  "", CONV_MEMORY, API_DRIVER}},
  // cudaDeviceGetPCIBusId
  {"cuDeviceGetPCIBusId",                                  {"hipDeviceGetPCIBusId",                                    "", CONV_MEMORY, API_DRIVER}},
  // cudaIpcCloseMemHandle
  {"cuIpcCloseMemHandle",                                  {"hipIpcCloseMemHandle",                                    "", CONV_MEMORY, API_DRIVER}},
  // cudaIpcGetEventHandle
  {"cuIpcGetEventHandle",                                  {"hipIpcGetEventHandle",                                    "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaIpcGetMemHandle
  {"cuIpcGetMemHandle",                                    {"hipIpcGetMemHandle",                                      "", CONV_MEMORY, API_DRIVER}},
  // cudaIpcOpenEventHandle
  {"cuIpcOpenEventHandle",                                 {"hipIpcOpenEventHandle",                                   "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaIpcOpenMemHandle
  {"cuIpcOpenMemHandle",                                   {"hipIpcOpenMemHandle",                                     "", CONV_MEMORY, API_DRIVER}},
  // cudaMalloc
  {"cuMemAlloc",                                           {"hipMalloc",                                               "", CONV_MEMORY, API_DRIVER}},
  {"cuMemAlloc_v2",                                        {"hipMalloc",                                               "", CONV_MEMORY, API_DRIVER}},
  // cudaHostAlloc
  {"cuMemAllocHost",                                       {"hipMemAllocHost",                                         "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemAllocHost_v2",                                    {"hipMemAllocHost",                                         "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemAllocManaged",                                    {"hipMemAllocManaged",                                      "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemAllocPitch",                                      {"hipMemAllocPitch",                                        "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemAllocPitch_v2",                                   {"hipMemAllocPitch",                                        "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy due to different signatures
  {"cuMemcpy",                                             {"hipMemcpy_",                                              "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy2D due to different signatures
  {"cuMemcpy2D",                                           {"hipMemcpy2D_",                                            "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpy2D_v2",                                        {"hipMemcpy2D_",                                            "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy2DAsync due to different signatures
  {"cuMemcpy2DAsync",                                      {"hipMemcpy2DAsync_",                                       "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpy2DAsync_v2",                                   {"hipMemcpy2DAsync_",                                       "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpy2DUnaligned",                                  {"hipMemcpy2DUnaligned",                                    "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpy2DUnaligned_v2",                               {"hipMemcpy2DUnaligned",                                    "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3D due to different signatures
  {"cuMemcpy3D",                                           {"hipMemcpy3D_",                                            "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpy3D_v2",                                        {"hipMemcpy3D_",                                            "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DAsync due to different signatures
  {"cuMemcpy3DAsync",                                      {"hipMemcpy3DAsync_",                                       "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpy3DAsync_v2",                                   {"hipMemcpy3DAsync_",                                       "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DPeer due to different signatures
  {"cuMemcpy3DPeer",                                       {"hipMemcpy3DPeer_",                                        "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpy3DPeerAsync due to different signatures
  {"cuMemcpy3DPeerAsync",                                  {"hipMemcpy3DPeerAsync_",                                   "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyAsync due to different signatures
  {"cuMemcpyAsync",                                        {"hipMemcpyAsync_",                                         "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyArrayToArray due to different signatures
  {"cuMemcpyAtoA",                                         {"hipMemcpyAtoA",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoA_v2",                                      {"hipMemcpyAtoA",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyAtoD",                                         {"hipMemcpyAtoD",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoD_v2",                                      {"hipMemcpyAtoD",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyAtoH",                                         {"hipMemcpyAtoH",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoH_v2",                                      {"hipMemcpyAtoH",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyAtoHAsync",                                    {"hipMemcpyAtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoHAsync_v2",                                 {"hipMemcpyAtoHAsync",                                      "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyDtoA",                                         {"hipMemcpyDtoA",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyDtoA_v2",                                      {"hipMemcpyDtoA",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyDtoD",                                         {"hipMemcpyDtoD",                                           "", CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyDtoD_v2",                                      {"hipMemcpyDtoD",                                           "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemcpyDtoDAsync",                                    {"hipMemcpyDtoDAsync",                                      "", CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyDtoDAsync_v2",                                 {"hipMemcpyDtoDAsync",                                      "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemcpyDtoH",                                         {"hipMemcpyDtoH",                                           "", CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyDtoH_v2",                                      {"hipMemcpyDtoH",                                           "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemcpyDtoHAsync",                                    {"hipMemcpyDtoHAsync",                                      "", CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyDtoHAsync_v2",                                 {"hipMemcpyDtoHAsync",                                      "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemcpyHtoA",                                         {"hipMemcpyHtoA",                                           "", CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyHtoA_v2",                                      {"hipMemcpyHtoA",                                           "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemcpyHtoAAsync",                                    {"hipMemcpyHtoAAsync",                                      "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyHtoAAsync_v2",                                 {"hipMemcpyHtoAAsync",                                      "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemcpyHtoD",                                         {"hipMemcpyHtoD",                                           "", CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyHtoD_v2",                                      {"hipMemcpyHtoD",                                           "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemcpyHtoDAsync",                                    {"hipMemcpyHtoDAsync",                                      "", CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyHtoDAsync_v2",                                 {"hipMemcpyHtoDAsync",                                      "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyPeer due to different signatures
  {"cuMemcpyPeer",                                         {"hipMemcpyPeer_",                                          "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMemcpyPeerAsync due to different signatures
  {"cuMemcpyPeerAsync",                                    {"hipMemcpyPeerAsync_",                                     "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaFree
  {"cuMemFree",                                            {"hipFree",                                                 "", CONV_MEMORY, API_DRIVER}},
  {"cuMemFree_v2",                                         {"hipFree",                                                 "", CONV_MEMORY, API_DRIVER}},
  // cudaFreeHost
  {"cuMemFreeHost",                                        {"hipHostFree",                                             "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemGetAddressRange",                                 {"hipMemGetAddressRange",                                   "", CONV_MEMORY, API_DRIVER}},
  {"cuMemGetAddressRange_v2",                              {"hipMemGetAddressRange",                                   "", CONV_MEMORY, API_DRIVER}},
  // cudaMemGetInfo
  {"cuMemGetInfo",                                         {"hipMemGetInfo",                                           "", CONV_MEMORY, API_DRIVER}},
  {"cuMemGetInfo_v2",                                      {"hipMemGetInfo",                                           "", CONV_MEMORY, API_DRIVER}},
  // cudaHostAlloc
  {"cuMemHostAlloc",                                       {"hipHostMalloc",                                           "", CONV_MEMORY, API_DRIVER}},
  // cudaHostGetDevicePointer
  {"cuMemHostGetDevicePointer",                            {"hipHostGetDevicePointer",                                 "", CONV_MEMORY, API_DRIVER}},
  {"cuMemHostGetDevicePointer_v2",                         {"hipHostGetDevicePointer",                                 "", CONV_MEMORY, API_DRIVER}},
  // cudaHostGetFlags
  {"cuMemHostGetFlags",                                    {"hipMemHostGetFlags",                                      "", CONV_MEMORY, API_DRIVER}},
  // cudaHostRegister
  {"cuMemHostRegister",                                    {"hipHostRegister",                                         "", CONV_MEMORY, API_DRIVER}},
  {"cuMemHostRegister_v2",                                 {"hipHostRegister",                                         "", CONV_MEMORY, API_DRIVER}},
  // cudaHostUnregister
  {"cuMemHostUnregister",                                  {"hipHostUnregister",                                       "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemsetD16",                                          {"hipMemsetD16",                                            "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD16_v2",                                       {"hipMemsetD16",                                            "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD16Async",                                     {"hipMemsetD16Async",                                       "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D16",                                        {"hipMemsetD2D16",                                          "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD2D16_v2",                                     {"hipMemsetD2D16",                                          "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D16Async",                                   {"hipMemsetD2D16Async",                                     "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D32",                                        {"hipMemsetD2D32",                                          "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD2D32_v2",                                     {"hipMemsetD2D32",                                          "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D32Async",                                   {"hipMemsetD2D32Async",                                     "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D8",                                         {"hipMemsetD2D8",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD2D8_v2",                                      {"hipMemsetD2D8",                                           "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuMemsetD2D8Async",                                    {"hipMemsetD2D8Async",                                      "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaMemset
  {"cuMemsetD32",                                          {"hipMemset",                                               "", CONV_MEMORY, API_DRIVER}},
  {"cuMemsetD32_v2",                                       {"hipMemset",                                               "", CONV_MEMORY, API_DRIVER}},
  // cudaMemsetAsync
  {"cuMemsetD32Async",                                     {"hipMemsetAsync",                                          "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemsetD8",                                           {"hipMemsetD8",                                             "", CONV_MEMORY, API_DRIVER}},
  {"cuMemsetD8_v2",                                        {"hipMemsetD8",                                             "", CONV_MEMORY, API_DRIVER}},
  // no analogue
  {"cuMemsetD8Async",                                      {"hipMemsetD8Async",                                        "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaMallocMipmappedArray due to different signatures
  {"cuMipmappedArrayCreate",                               {"hipMipmappedArrayCreate",                                 "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFreeMipmappedArray due to different signatures
  {"cuMipmappedArrayDestroy",                              {"hipMipmappedArrayDestroy",                                "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetMipmappedArrayLevel due to different signatures
  {"cuMipmappedArrayGetLevel",                             {"hipMipmappedArrayGetLevel",                               "", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.12. Unified Addressing
  // cudaMemAdvise
  {"cuMemAdvise",                                          {"hipMemAdvise",                                            "", CONV_ADDRESSING, API_DRIVER, HIP_UNSUPPORTED}},
  // TODO: double check cudaMemPrefetchAsync
  {"cuMemPrefetchAsync",                                   {"hipMemPrefetchAsync_",                                    "", CONV_ADDRESSING, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaMemRangeGetAttribute
  {"cuMemRangeGetAttribute",                               {"hipMemRangeGetAttribute",                                 "", CONV_ADDRESSING, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaMemRangeGetAttributes
  {"cuMemRangeGetAttributes",                              {"hipMemRangeGetAttributes",                                "", CONV_ADDRESSING, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuPointerGetAttribute",                                {"hipPointerGetAttribute",                                  "", CONV_ADDRESSING, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaPointerGetAttributes due to different signatures
  {"cuPointerGetAttributes",                               {"hipPointerGetAttributes",                                 "", CONV_ADDRESSING, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuPointerSetAttribute",                                {"hipPointerSetAttribute",                                  "", CONV_ADDRESSING, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.13. Stream Management
  // cudaStreamAddCallback
  {"cuStreamAddCallback",                                  {"hipStreamAddCallback",                                    "", CONV_STREAM, API_DRIVER}},
  // cudaStreamAttachMemAsync
  {"cuStreamAttachMemAsync",                               {"hipStreamAttachMemAsync",                                 "", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaStreamBeginCapture
  {"cuStreamBeginCapture",                                 {"hipStreamBeginCapture",                                   "", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaStreamCreateWithFlags
  {"cuStreamCreate",                                       {"hipStreamCreateWithFlags",                                "", CONV_STREAM, API_DRIVER}},
  // cudaStreamCreateWithPriority
  {"cuStreamCreateWithPriority",                           {"hipStreamCreateWithPriority",                             "", CONV_STREAM, API_DRIVER}},
  // cudaStreamDestroy
  {"cuStreamDestroy",                                      {"hipStreamDestroy",                                        "", CONV_STREAM, API_DRIVER}},
  {"cuStreamDestroy_v2",                                   {"hipStreamDestroy",                                        "", CONV_STREAM, API_DRIVER}},
  // cudaStreamEndCapture
  {"cuStreamEndCapture",                                   {"hipStreamEndCapture",                                     "", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuStreamGetCtx",                                       {"hipStreamGetContext",                                     "", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaStreamGetFlags
  {"cuStreamGetFlags",                                     {"hipStreamGetFlags",                                       "", CONV_STREAM, API_DRIVER}},
  // cudaStreamGetPriority
  {"cuStreamGetPriority",                                  {"hipStreamGetPriority",                                    "", CONV_STREAM, API_DRIVER}},
  // cudaStreamIsCapturing
  {"cuStreamIsCapturing",                                  {"hipStreamIsCapturing",                                    "", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaStreamQuery
  {"cuStreamQuery",                                        {"hipStreamQuery",                                          "", CONV_STREAM, API_DRIVER}},
  // cudaStreamSynchronize
  {"cuStreamSynchronize",                                  {"hipStreamSynchronize",                                    "", CONV_STREAM, API_DRIVER}},
  // cudaStreamWaitEvent
  {"cuStreamWaitEvent",                                    {"hipStreamWaitEvent",                                      "", CONV_STREAM, API_DRIVER}},

  // 5.14. Event Management
  // cudaEventCreateWithFlags
  {"cuEventCreate",                                        {"hipEventCreateWithFlags",                                 "", CONV_EVENT, API_DRIVER}},
  // cudaEventDestroy
  {"cuEventDestroy",                                       {"hipEventDestroy",                                         "", CONV_EVENT, API_DRIVER}},
  {"cuEventDestroy_v2",                                    {"hipEventDestroy",                                         "", CONV_EVENT, API_DRIVER}},
  // cudaEventElapsedTime
  {"cuEventElapsedTime",                                   {"hipEventElapsedTime",                                     "", CONV_EVENT, API_DRIVER}},
  // cudaEventQuery
  {"cuEventQuery",                                         {"hipEventQuery",                                           "", CONV_EVENT, API_DRIVER}},
  // cudaEventRecord
  {"cuEventRecord",                                        {"hipEventRecord",                                          "", CONV_EVENT, API_DRIVER}},
  // cudaEventSynchronize
  {"cuEventSynchronize",                                   {"hipEventSynchronize",                                     "", CONV_EVENT, API_DRIVER}},

  // 5.15. External Resource Interoperability
  // cudaDestroyExternalMemory
  {"cuDestroyExternalMemory",                              {"hipDestroyExternalMemory",                                "", CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaDestroyExternalSemaphore
  {"cuDestroyExternalSemaphore",                           {"hipDestroyExternalSemaphore",                             "", CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaExternalMemoryGetMappedBuffer
  {"cuExternalMemoryGetMappedBuffer",                      {"hipExternalMemoryGetMappedBuffer",                        "", CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaExternalMemoryGetMappedMipmappedArray
  {"cuExternalMemoryGetMappedMipmappedArray",              {"hipExternalMemoryGetMappedMipmappedArray",                "", CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaImportExternalMemory
  {"cuImportExternalMemory",                               {"hipImportExternalMemory",                                 "", CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaImportExternalSemaphore
  {"cuImportExternalSemaphore",                            {"hipImportExternalSemaphore",                              "", CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaSignalExternalSemaphoresAsync
  {"cuSignalExternalSemaphoresAsync",                      {"hipSignalExternalSemaphoresAsync",                        "", CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaWaitExternalSemaphoresAsync
  {"cuWaitExternalSemaphoresAsync",                        {"hipWaitExternalSemaphoresAsync",                          "", CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.16. Stream Memory Operations
  // no analogues
  {"cuStreamBatchMemOp",                                   {"hipStreamBatchMemOp",                                     "", CONV_STREAM_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuStreamWaitValue32",                                  {"hipStreamWaitValue32",                                    "", CONV_STREAM_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuStreamWaitValue64",                                  {"hipStreamWaitValue64",                                    "", CONV_STREAM_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuStreamWriteValue32",                                 {"hipStreamWriteValue32",                                   "", CONV_STREAM_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuStreamWriteValue64",                                 {"hipStreamWriteValue64",                                   "", CONV_STREAM_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.17.Execution Control
  // no analogue
  {"cuFuncGetAttribute",                                   {"hipFuncGetAttribute",                                     "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetAttribute due to different signatures
  {"cuFuncSetAttribute",                                   {"hipFuncSetAttribute",                                     "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetCacheConfig due to different signatures
  {"cuFuncSetCacheConfig",                                 {"hipFuncSetCacheConfig",                                   "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaFuncSetCacheConfig due to different signatures
  {"cuFuncSetSharedMemConfig",                             {"hipFuncSetSharedMemConfig",                               "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernel due to different signatures
  {"cuLaunchCooperativeKernel",                            {"hipLaunchCooperativeKernel",                              "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunchCooperativeKernelMultiDevice due to different signatures
  {"cuLaunchCooperativeKernelMultiDevice",                 {"hipLaunchCooperativeKernelMultiDevice",                   "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaLaunchHostFunc
  {"cuLaunchHostFunc",                                     {"hipLaunchHostFunc",                                       "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunchKernel due to different signatures
  {"cuLaunchKernel",                                       {"hipModuleLaunchKernel",                                   "", CONV_EXECUTION, API_DRIVER}},

  // 5.18.Execution Control [DEPRECATED]
  // no analogue
  {"cuFuncSetBlockShape",                                  {"hipFuncSetBlockShape",                                    "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuFuncSetSharedSize",                                  {"hipFuncSetSharedSize",                                    "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaLaunch due to different signatures
  {"cuLaunch",                                             {"hipLaunch",                                               "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuLaunchGrid",                                         {"hipLaunchGrid",                                           "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuLaunchGridAsync",                                    {"hipLaunchGridAsync",                                      "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuParamSetf",                                          {"hipParamSetf",                                            "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuParamSeti",                                          {"hipParamSeti",                                            "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuParamSetSize",                                       {"hipParamSetSize",                                         "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuParamSetTexRef",                                     {"hipParamSetTexRef",                                       "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuParamSetv",                                          {"hipParamSetv",                                            "", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.19. Graph Management
  // cudaGraphAddChildGraphNode
  {"cuGraphAddChildGraphNode",                             {"hipGraphAddChildGraphNode",                               "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphAddDependencies
  {"cuGraphAddDependencies",                               {"hipGraphAddDependencies",                                 "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphAddEmptyNode
  {"cuGraphAddEmptyNode",                                  {"hipGraphAddEmptyNode",                                    "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphAddHostNode
  {"cuGraphAddHostNode",                                   {"hipGraphAddHostNode",                                     "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphAddKernelNode
  {"cuGraphAddKernelNode",                                 {"hipGraphAddKernelNode",                                   "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphAddMemcpyNode
  {"cuGraphAddMemcpyNode",                                 {"hipGraphAddMemcpyNode",                                   "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphAddMemsetNode
  {"cuGraphAddMemsetNode",                                 {"hipGraphAddMemsetNode",                                   "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphChildGraphNodeGetGraph
  {"cuGraphChildGraphNodeGetGraph",                        {"hipGraphChildGraphNodeGetGraph",                          "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphClone
  {"cuGraphClone",                                         {"hipGraphClone",                                           "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphCreate
  {"cuGraphCreate",                                        {"hipGraphCreate",                                          "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphDestroy
  {"cuGraphDestroy",                                       {"hipGraphDestroy",                                         "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphDestroyNode
  {"cuGraphDestroyNode",                                   {"hipGraphDestroyNode",                                     "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphExecDestroy
  {"cuGraphExecDestroy",                                   {"hipGraphExecDestroy",                                     "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphGetEdges
  {"cuGraphGetEdges",                                      {"hipGraphGetEdges",                                        "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphGetNodes
  {"cuGraphGetNodes",                                      {"hipGraphGetNodes",                                        "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphGetRootNodes
  {"cuGraphGetRootNodes",                                  {"hipGraphGetRootNodes",                                    "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphHostNodeGetParams
  {"cuGraphHostNodeGetParams",                             {"hipGraphHostNodeGetParams",                               "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphHostNodeSetParams
  {"cuGraphHostNodeSetParams",                             {"hipGraphHostNodeSetParams",                               "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphInstantiate
  {"cuGraphInstantiate",                                   {"hipGraphInstantiate",                                     "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphKernelNodeGetParams
  {"cuGraphKernelNodeGetParams",                           {"hipGraphKernelNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphKernelNodeSetParams
  {"cuGraphKernelNodeSetParams",                           {"hipGraphKernelNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphLaunch
  {"cuGraphLaunch",                                        {"hipGraphLaunch",                                          "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphMemcpyNodeGetParams
  {"cuGraphMemcpyNodeGetParams",                           {"hipGraphMemcpyNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphMemcpyNodeSetParams
  {"cuGraphMemcpyNodeSetParams",                           {"hipGraphMemcpyNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphMemsetNodeGetParams
  {"cuGraphMemsetNodeGetParams",                           {"hipGraphMemsetNodeGetParams",                             "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphMemsetNodeSetParams
  {"cuGraphMemsetNodeSetParams",                           {"hipGraphMemsetNodeSetParams",                             "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphNodeFindInClone
  {"cuGraphNodeFindInClone",                               {"hipGraphNodeFindInClone",                                 "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphNodeGetDependencies
  {"cuGraphNodeGetDependencies",                           {"hipGraphNodeGetDependencies",                             "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphNodeGetDependentNodes
  {"cuGraphNodeGetDependentNodes",                         {"hipGraphNodeGetDependentNodes",                           "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphNodeGetType
  {"cuGraphNodeGetType",                                   {"hipGraphNodeGetType",                                     "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphRemoveDependencies
  {"cuGraphRemoveDependencies",                            {"hipGraphRemoveDependencies",                              "", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.20. Occupancy
  // cudaOccupancyMaxActiveBlocksPerMultiprocessor
  {"cuOccupancyMaxActiveBlocksPerMultiprocessor",          {"hipOccupancyMaxActiveBlocksPerMultiprocessor",            "", CONV_OCCUPANCY, API_DRIVER}},
  // cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",   "", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaOccupancyMaxPotentialBlockSize
  {"cuOccupancyMaxPotentialBlockSize",                     {"hipOccupancyMaxPotentialBlockSize",                       "", CONV_OCCUPANCY, API_DRIVER}},
  // cudaOccupancyMaxPotentialBlockSizeWithFlags
  {"cuOccupancyMaxPotentialBlockSizeWithFlags",            {"hipOccupancyMaxPotentialBlockSizeWithFlags",              "", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.21. Texture Reference Management
  // no analogues
  {"cuTexRefGetAddress",                                   {"hipTexRefGetAddress",                                     "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetAddress_v2",                                {"hipTexRefGetAddress",                                     "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetAddressMode",                               {"hipTexRefGetAddressMode",                                 "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetArray",                                     {"hipTexRefGetArray",                                       "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetBorderColor",                               {"hipTexRefGetBorderColor",                                 "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetFilterMode",                                {"hipTexRefGetFilterMode",                                  "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetFlags",                                     {"hipTexRefGetFlags",                                       "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetFormat",                                    {"hipTexRefGetFormat",                                      "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMaxAnisotropy",                             {"hipTexRefGetMaxAnisotropy",                               "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMipmapFilterMode",                          {"hipTexRefGetMipmapFilterMode",                            "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMipmapLevelBias",                           {"hipTexRefGetMipmapLevelBias",                             "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMipmapLevelClamp",                          {"hipTexRefGetMipmapLevelClamp",                            "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMipmappedArray",                            {"hipTexRefGetMipmappedArray",                              "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetAddress",                                   {"hipTexRefSetAddress",                                     "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetAddress_v2",                                {"hipTexRefSetAddress",                                     "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetAddress2D",                                 {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetAddress2D_v2",                              {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetAddress2D_v3",                              {"hipTexRefSetAddress2D",                                   "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetAddressMode",                               {"hipTexRefSetAddressMode",                                 "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetArray",                                     {"hipTexRefSetArray",                                       "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetBorderColor",                               {"hipTexRefSetBorderColor",                                 "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetFilterMode",                                {"hipTexRefSetFilterMode",                                  "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetFlags",                                     {"hipTexRefSetFlags",                                       "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetFormat",                                    {"hipTexRefSetFormat",                                      "", CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetMaxAnisotropy",                             {"hipTexRefSetMaxAnisotropy",                               "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetMipmapFilterMode",                          {"hipTexRefSetMipmapFilterMode",                            "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetMipmapLevelBias",                           {"hipTexRefSetMipmapLevelBias",                             "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetMipmapLevelClamp",                          {"hipTexRefSetMipmapLevelClamp",                            "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetMipmappedArray",                            {"hipTexRefSetMipmappedArray",                              "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.22. Texture Reference Management [DEPRECATED]
  // no analogues
  {"cuTexRefCreate",                                       {"hipTexRefCreate",                                         "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefDestroy",                                      {"hipTexRefDestroy",                                        "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.23. Surface Reference Management
  // no analogues
  {"cuSurfRefGetArray",                                    {"hipSurfRefGetArray",                                      "", CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuSurfRefSetArray",                                    {"hipSurfRefSetArray",                                      "", CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.24. Texture Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateTextureObject due to different signatures
  {"cuTexObjectCreate",                                    {"hipTexObjectCreate",                                      "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaDestroyTextureObject
  {"cuTexObjectDestroy",                                   {"hipTexObjectDestroy",                                     "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectResourceDesc due to different signatures
  {"cuTexObjectGetResourceDesc",                           {"hipTexObjectGetResourceDesc",                             "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGetTextureObjectResourceViewDesc
  {"cuTexObjectGetResourceViewDesc",                       {"hipTexObjectGetResourceViewDesc",                         "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetTextureObjectTextureDesc due to different signatures
  {"cuTexObjectGetTextureDesc",                            {"hipTexObjectGetTextureDesc",                              "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.25. Surface Object Management
  // no analogue
  // NOTE: Not equal to cudaCreateSurfaceObject due to different signatures
  {"cuSurfObjectCreate",                                   {"hipSurfObjectCreate",                                     "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaDestroySurfaceObject
  {"cuSurfObjectDestroy",                                  {"hipSurfObjectDestroy",                                    "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  // NOTE: Not equal to cudaGetSurfaceObjectResourceDesc due to different signatures
  {"cuSurfObjectGetResourceDesc",                          {"hipSurfObjectGetResourceDesc",                            "", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.26. Peer Context Memory Access
  // no analogue
  // NOTE: Not equal to cudaDeviceEnablePeerAccess due to different signatures
  {"cuCtxEnablePeerAccess",                                {"hipCtxEnablePeerAccess",                                  "", CONV_PEER, API_DRIVER}},
  // no analogue
  // NOTE: Not equal to cudaDeviceDisablePeerAccess due to different signatures
  {"cuCtxDisablePeerAccess",                               {"hipCtxDisablePeerAccess",                                 "", CONV_PEER, API_DRIVER}},
  // cudaDeviceCanAccessPeer
  {"cuDeviceCanAccessPeer",                                {"hipDeviceCanAccessPeer",                                  "", CONV_PEER, API_DRIVER}},
  // cudaDeviceGetP2PAttribute
  {"cuDeviceGetP2PAttribute",                              {"hipDeviceGetP2PAttribute",                                "", CONV_PEER, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.27. Graphics Interoperability
  // cudaGraphicsMapResources
  {"cuGraphicsMapResources",                               {"hipGraphicsMapResources",                                 "", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedMipmappedArray
  {"cuGraphicsResourceGetMappedMipmappedArray",            {"hipGraphicsResourceGetMappedMipmappedArray",              "", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedPointer
  {"cuGraphicsResourceGetMappedPointer",                   {"hipGraphicsResourceGetMappedPointer",                     "", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedPointer
  {"cuGraphicsResourceGetMappedPointer_v2",                {"hipGraphicsResourceGetMappedPointer",                     "", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceSetMapFlags
  {"cuGraphicsResourceSetMapFlags",                        {"hipGraphicsResourceSetMapFlags",                          "", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceSetMapFlags
  {"cuGraphicsResourceSetMapFlags_v2",                     {"hipGraphicsResourceSetMapFlags",                          "", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsSubResourceGetMappedArray
  {"cuGraphicsSubResourceGetMappedArray",                  {"hipGraphicsSubResourceGetMappedArray",                    "", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsUnmapResources
  {"cuGraphicsUnmapResources",                             {"hipGraphicsUnmapResources",                               "", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsUnregisterResource
  {"cuGraphicsUnregisterResource",                         {"hipGraphicsUnregisterResource",                           "", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.28. Profiler Control
  // cudaProfilerInitialize
  {"cuProfilerInitialize",                                 {"hipProfilerInitialize",                                   "", CONV_PROFILER, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaProfilerStart
  {"cuProfilerStart",                                      {"hipProfilerStart",                                        "", CONV_PROFILER, API_DRIVER}},
  // cudaProfilerStop
  {"cuProfilerStop",                                       {"hipProfilerStop",                                         "", CONV_PROFILER, API_DRIVER}},

  // 5.29. OpenGL Interoperability
  // cudaGLGetDevices
  {"cuGLGetDevices",                                       {"hipGLGetDevices",                                         "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsGLRegisterBuffer
  {"cuGraphicsGLRegisterBuffer",                           {"hipGraphicsGLRegisterBuffer",                             "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsGLRegisterImage
  {"cuGraphicsGLRegisterImage",                            {"hipGraphicsGLRegisterImage",                              "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaWGLGetDevice
  {"cuWGLGetDevice",                                       {"hipWGLGetDevice",                                         "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.29. OpenGL Interoperability [DEPRECATED]
  // no analogue
  {"cuGLCtxCreate",                                        {"hipGLCtxCreate",                                          "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuGLInit",                                             {"hipGLInit",                                               "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // NOTE: Not equal to cudaGLMapBufferObject due to different signatures
  {"cuGLMapBufferObject",                                  {"hipGLMapBufferObject",                                    "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // NOTE: Not equal to cudaGLMapBufferObjectAsync due to different signatures
  {"cuGLMapBufferObjectAsync",                             {"hipGLMapBufferObjectAsync",                               "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGLRegisterBufferObject
  {"cuGLRegisterBufferObject",                             {"hipGLRegisterBufferObject",                               "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGLSetBufferObjectMapFlags
  {"cuGLSetBufferObjectMapFlags",                          {"hipGLSetBufferObjectMapFlags",                            "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGLUnmapBufferObject
  {"cuGLUnmapBufferObject",                                {"hipGLUnmapBufferObject",                                  "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGLUnmapBufferObjectAsync
  {"cuGLUnmapBufferObjectAsync",                           {"hipGLUnmapBufferObjectAsync",                             "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGLUnregisterBufferObject
  {"cuGLUnregisterBufferObject",                           {"hipGLUnregisterBufferObject",                             "", CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.30.Direct3D 9 Interoperability
  // no analogue
  {"cuD3D9CtxCreate",                                      {"hipD3D9CtxCreate",                                        "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
    // no analogue
  {"cuD3D9CtxCreateOnDevice",                              {"hipD3D9CtxCreateOnDevice",                                "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9GetDevice
  {"cuD3D9GetDevice",                                      {"hipD3D9GetDevice",                                        "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9GetDevices
  {"cuD3D9GetDevices",                                     {"hipD3D9GetDevices",                                       "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9GetDirect3DDevice
  {"cuD3D9GetDirect3DDevice",                              {"hipD3D9GetDirect3DDevice",                                "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D9RegisterResource
  {"cuGraphicsD3D9RegisterResource",                       {"hipGraphicsD3D9RegisterResource",                         "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.30.Direct3D 9 Interoperability [DEPRECATED]
  // cudaD3D9MapResources
  {"cuD3D9MapResources",                                   {"hipD3D9MapResources",                                     "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9RegisterResource
  {"cuD3D9RegisterResource",                               {"hipD3D9RegisterResource",                                 "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9ResourceGetMappedArray
  {"cuD3D9ResourceGetMappedArray",                         {"hipD3D9ResourceGetMappedArray",                           "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9ResourceGetMappedPitch
  {"cuD3D9ResourceGetMappedPitch",                         {"hipD3D9ResourceGetMappedPitch",                           "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9ResourceGetMappedPointer
  {"cuD3D9ResourceGetMappedPointer",                       {"hipD3D9ResourceGetMappedPointer",                         "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9ResourceGetMappedSize
  {"cuD3D9ResourceGetMappedSize",                          {"hipD3D9ResourceGetMappedSize",                            "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9ResourceGetSurfaceDimensions
  {"cuD3D9ResourceGetSurfaceDimensions",                   {"hipD3D9ResourceGetSurfaceDimensions",                     "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9ResourceSetMapFlags
  {"cuD3D9ResourceSetMapFlags",                            {"hipD3D9ResourceSetMapFlags",                              "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9UnmapResources
  {"cuD3D9UnmapResources",                                 {"hipD3D9UnmapResources",                                   "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D9UnregisterResource
  {"cuD3D9UnregisterResource",                             {"hipD3D9UnregisterResource",                               "", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.31. Direct3D 10 Interoperability
  // cudaD3D10GetDevice
  {"cuD3D10GetDevice",                                     {"hipD3D10GetDevice",                                       "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10GetDevices
  {"cuD3D10GetDevices",                                    {"hipD3D10GetDevices",                                      "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D10RegisterResource
  {"cuGraphicsD3D10RegisterResource",                      {"hipGraphicsD3D10RegisterResource",                        "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.31. Direct3D 10 Interoperability [DEPRECATED]
  // no analogue
  {"cuD3D10CtxCreate",                                     {"hipD3D10CtxCreate",                                       "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuD3D10CtxCreateOnDevice",                             {"hipD3D10CtxCreateOnDevice",                               "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10GetDirect3DDevice
  {"cuD3D10GetDirect3DDevice",                             {"hipD3D10GetDirect3DDevice",                               "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10MapResources
  {"cuD3D10MapResources",                                  {"hipD3D10MapResources",                                    "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10RegisterResource
  {"cuD3D10RegisterResource",                              {"hipD3D10RegisterResource",                                "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10ResourceGetMappedArray
  {"cuD3D10ResourceGetMappedArray",                        {"hipD3D10ResourceGetMappedArray",                          "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10ResourceGetMappedPitch
  {"cuD3D10ResourceGetMappedPitch",                        {"hipD3D10ResourceGetMappedPitch",                          "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10ResourceGetMappedPointer
  {"cuD3D10ResourceGetMappedPointer",                      {"hipD3D10ResourceGetMappedPointer",                        "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10ResourceGetMappedSize
  {"cuD3D10ResourceGetMappedSize",                         {"hipD3D10ResourceGetMappedSize",                           "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10ResourceGetSurfaceDimensions
  {"cuD3D10ResourceGetSurfaceDimensions",                  {"hipD3D10ResourceGetSurfaceDimensions",                    "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10ResourceSetMapFlags
  {"cuD310ResourceSetMapFlags",                            {"hipD3D10ResourceSetMapFlags",                             "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10UnmapResources
  {"cuD3D10UnmapResources",                                {"hipD3D10UnmapResources",                                  "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D10UnregisterResource
  {"cuD3D10UnregisterResource",                            {"hipD3D10UnregisterResource",                              "", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.32. Direct3D 11 Interoperability
  // cudaD3D11GetDevice
  {"cuD3D11GetDevice",                                     {"hipD3D11GetDevice",                                       "", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D11GetDevices
  {"cuD3D11GetDevices",                                    {"hipD3D11GetDevices",                                      "", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsD3D11RegisterResource
  {"cuGraphicsD3D11RegisterResource",                      {"hipGraphicsD3D11RegisterResource",                        "", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.32. Direct3D 11 Interoperability [DEPRECATED]
  // no analogue
  {"cuD3D11CtxCreate",                                     {"hipD3D11CtxCreate",                                       "", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuD3D11CtxCreateOnDevice",                             {"hipD3D11CtxCreateOnDevice",                               "", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaD3D11GetDirect3DDevice
  {"cuD3D11GetDirect3DDevice",                             {"hipD3D11GetDirect3DDevice",                               "", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.33. VDPAU Interoperability
  // cudaGraphicsVDPAURegisterOutputSurface
  {"cuGraphicsVDPAURegisterOutputSurface",                 {"hipGraphicsVDPAURegisterOutputSurface",                   "", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsVDPAURegisterVideoSurface
  {"cuGraphicsVDPAURegisterVideoSurface",                  {"hipGraphicsVDPAURegisterVideoSurface",                    "", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaVDPAUGetDevice
  {"cuVDPAUGetDevice",                                     {"hipVDPAUGetDevice",                                       "", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},
  // no analogue
  {"cuVDPAUCtxCreate",                                     {"hipVDPAUCtxCreate",                                       "", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},

  // 5.34. EGL Interoperability
  // cudaEGLStreamConsumerAcquireFrame
  {"cuEGLStreamConsumerAcquireFrame",                      {"hipEGLStreamConsumerAcquireFrame",                        "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerConnect
  {"cuEGLStreamConsumerConnect",                           {"hipEGLStreamConsumerConnect",                             "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerConnectWithFlags
  {"cuEGLStreamConsumerConnectWithFlags",                  {"hipEGLStreamConsumerConnectWithFlags",                    "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerDisconnect
  {"cuEGLStreamConsumerDisconnect",                        {"hipEGLStreamConsumerDisconnect",                          "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaEGLStreamConsumerReleaseFrame
  {"cuEGLStreamConsumerReleaseFrame",                      {"hipEGLStreamConsumerReleaseFrame",                        "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerConnect
  {"cuEGLStreamProducerConnect",                           {"hipEGLStreamProducerConnect",                             "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerDisconnect
  {"cuEGLStreamProducerDisconnect",                        {"hipEGLStreamProducerDisconnect",                          "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerPresentFrame
  {"cuEGLStreamProducerPresentFrame",                      {"hipEGLStreamProducerPresentFrame",                        "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaEGLStreamProducerReturnFrame
  {"cuEGLStreamProducerReturnFrame",                       {"hipEGLStreamProducerReturnFrame",                         "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsEGLRegisterImage
  {"cuGraphicsEGLRegisterImage",                           {"hipGraphicsEGLRegisterImage",                             "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaGraphicsResourceGetMappedEglFrame
  {"cuGraphicsResourceGetMappedEglFrame",                  {"hipGraphicsResourceGetMappedEglFrame",                    "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
  // cudaEventCreateFromEGLSync
  {"cuEventCreateFromEGLSync",                             {"hipEventCreateFromEGLSync",                               "", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},
};

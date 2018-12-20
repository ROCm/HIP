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

// Maps the names of CUDA RUNTIME API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_TYPE_NAME_MAP {

  // 1. Structs

  // no analogue
  {"cudaChannelFormatDesc",                                            {"hipChannelFormatDesc",                                     "", CONV_TYPE, API_RUNTIME}},
  // no analogue
  {"cudaDeviceProp",                                                   {"hipDeviceProp_t",                                          "", CONV_TYPE, API_RUNTIME}},
  // NOTE: int warpSize is a field of cudaDeviceProp
  {"warpSize",                                                         {"hipWarpSize",                                              "", CONV_TYPE, API_RUNTIME}},

  // no analogue
  {"cudaEglFrame",                                                     {"hipEglFrame",                                              "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaEglFrame_st",                                                  {"hipEglFrame",                                              "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // no analogue
  {"cudaEglPlaneDesc",                                                 {"hipEglPlaneDesc",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaEglPlaneDesc_st",                                              {"hipEglPlaneDesc",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // no analogue
  {"cudaExtent",                                                       {"hipExtent",                                                "", CONV_TYPE, API_RUNTIME}},

  // CUDA_EXTERNAL_MEMORY_BUFFER_DESC
  {"cudaExternalMemoryBufferDesc",                                     {"HIP_EXTERNAL_MEMORY_BUFFER_DESC",                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_MEMORY_HANDLE_DESC
  {"cudaExternalMemoryHandleDesc",                                     {"HIP_EXTERNAL_MEMORY_HANDLE_DESC",                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC
  {"cudaExternalMemoryMipmappedArrayDesc",                             {"HIP_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC",                 "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
  {"cudaExternalSemaphoreHandleDesc",                                  {"HIP_EXTERNAL_SEMAPHORE_HANDLE_DESC",                       "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
  {"cudaExternalSemaphoreSignalParams",                                {"HIP_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS",                     "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
  {"cudaExternalSemaphoreWaitParams",                                  {"HIP_EXTERNAL_SEMAPHORE_WAIT_PARAMS",                       "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // no analogue
  {"cudaFuncAttributes",                                               {"hipFuncAttributes",                                        "", CONV_TYPE, API_RUNTIME}},

  // CUDA_HOST_NODE_PARAMS
  {"cudaHostNodeParams",                                               {"HIP_HOST_NODE_PARAMS",                                     "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUipcEventHandle
  {"cudaIpcEventHandle_t",                                             {"ihipIpcEventHandle_t",                                     "", CONV_TYPE, API_RUNTIME}},
  // CUipcEventHandle_st
  {"cudaIpcEventHandle_st",                                            {"ihipIpcEventHandle_t",                                     "", CONV_TYPE, API_RUNTIME}},

  // CUipcMemHandle
  {"cudaIpcMemHandle_t",                                               {"hipIpcMemHandle_t",                                        "", CONV_TYPE, API_RUNTIME}},
  // CUipcMemHandle_st
  {"cudaIpcMemHandle_st",                                              {"hipIpcMemHandle_st",                                       "", CONV_TYPE, API_RUNTIME}},

  // CUDA_KERNEL_NODE_PARAMS
  {"cudaKernelNodeParams",                                             {"hipKernelNodeParams",                                      "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // no analogue
  // CUDA_LAUNCH_PARAMS struct differs
  {"cudaLaunchParams",                                                 {"hipLaunchParams",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // no analogue
  // NOTE: HIP struct is bigger and contains cudaMemcpy3DParms only in the beginning
  {"cudaMemcpy3DParms",                                                {"hipMemcpy3DParms",                                         "", CONV_TYPE, API_RUNTIME}},

  // no analogue
  {"cudaMemcpy3DPeerParms",                                            {"hipMemcpy3DPeerParms",                                     "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_MEMSET_NODE_PARAMS
  {"cudaMemsetParams",                                                 {"hipMemsetParams",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // no analogue
  {"cudaPitchedPtr",                                                   {"hipPitchedPtr",                                            "", CONV_TYPE, API_RUNTIME}},

  // no analogue
  {"cudaPointerAttributes",                                            {"hipPointerAttribute_t",                                    "", CONV_TYPE, API_RUNTIME}},

  // no analogue
  {"cudaPos",                                                          {"hipPos",                                                   "", CONV_TYPE, API_RUNTIME}},

  // no analogue
  // NOTE: CUDA_RESOURCE_DESC struct differs
  {"cudaResourceDesc",                                                 {"hipResourceDesc",                                          "", CONV_TYPE, API_RUNTIME}},

  // NOTE: CUDA_RESOURCE_VIEW_DESC has reserved bytes in the end
  {"cudaResourceViewDesc",                                             {"hipResourceViewDesc",                                      "", CONV_TYPE, API_RUNTIME}},

  // no analogue
  // NOTE: CUDA_TEXTURE_DESC differs
  {"cudaTextureDesc",                                                  {"hipTextureDesc",                                           "", CONV_TYPE, API_RUNTIME}},

  // NOTE: the same struct and its name
  {"CUuuid_st",                                                        {"hipUUID",                                                  "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // NOTE: possibly CUsurfref is analogue
  {"surfaceReference",                                                 {"hipSurfaceReference",                                      "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // the same - CUevent_st
  {"CUevent_st",                                                       {"ihipEvent_t",                                              "", CONV_TYPE, API_RUNTIME}},
  // CUevent
  {"cudaEvent_t",                                                      {"hipEvent_t",                                               "", CONV_TYPE, API_RUNTIME}},

  // CUextMemory_st
  {"CUexternalMemory_st",                                              {"hipExtMemory_st",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // CUexternalMemory
  {"cudaExternalMemory_t",                                             {"hipExternalMemory",                                        "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUextSemaphore_st
  {"CUexternalSemaphore_st",                                           {"hipExtSemaphore_st",                                       "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // CUexternalSemaphore
  {"cudaExternalSemaphore_t",                                          {"hipExternalSemaphore",                                     "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // the same - CUgraph_st
  {"CUgraph_st",                                                       {"hipGraph_st",                                              "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // CUgraph
  {"cudaGraph_t",                                                      {"hipGraph",                                                 "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // the same -CUgraphExec_st
  {"CUgraphExec_st",                                                   {"hipGraphExec_st",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // CUgraphExec
  {"cudaGraphExec_t",                                                  {"hipGraphExec",                                             "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUgraphicsResource_st
  {"cudaGraphicsResource",                                             {"hipGraphicsResource_st",                                   "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // CUgraphicsResource
  {"cudaGraphicsResource_t",                                           {"hipGraphicsResource_t",                                    "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // the same - CUgraphNode_st
  {"CUgraphNode_st",                                                   {"hipGraphNode_st",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // CUgraphNode
  {"cudaGraphNode_t",                                                  {"hipGraphNode",                                             "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUeglStreamConnection_st
  {"CUeglStreamConnection_st",                                         {"hipEglStreamConnection",                                   "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // CUeglStreamConnection
  {"cudaEglStreamConnection",                                          {"hipEglStreamConnection",                                   "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUarray_st
  {"cudaArray",                                                        {"hipArray",                                                 "", CONV_TYPE, API_RUNTIME}},
  // CUarray
  {"cudaArray_t",                                                      {"hipArray_t",                                               "", CONV_TYPE, API_RUNTIME}},
  // no analogue
  {"cudaArray_const_t",                                                {"hipArray_const_t",                                         "", CONV_TYPE, API_RUNTIME}},

  // CUmipmappedArray_st
  {"cudaMipmappedArray",                                               {"hipMipmappedArray",                                        "", CONV_TYPE, API_RUNTIME}},
  // CUmipmappedArray
  {"cudaMipmappedArray_t",                                             {"hipMipmappedArray_t",                                      "", CONV_TYPE, API_RUNTIME}},
  // no analogue
  {"cudaMipmappedArray_const_t",                                       {"hipMipmappedArray_const_t",                                "", CONV_TYPE, API_RUNTIME}},

  // the same - CUstream_st
  {"CUstream_st",                                                      {"ihipStream_t",                                             "", CONV_TYPE, API_RUNTIME}},
  // CUstream
  {"cudaStream_t",                                                     {"hipStream_t",                                              "", CONV_TYPE, API_RUNTIME}},

  // 3. Enums

  // no analogue
  {"cudaCGScope",                                                      {"hipCGScope",                                               "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaCGScope enum values
  {"cudaCGScopeInvalid",                                               {"hipCGScopeInvalid",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  {"cudaCGScopeGrid",                                                  {"hipCGScopeGrid",                                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  {"cudaCGScopeMultiGrid",                                             {"hipCGScopeMultiGrid",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2

  // no analogue
  {"cudaChannelFormatKind",                                            {"hipChannelFormatKind",                                     "", CONV_TYPE, API_RUNTIME}},
  // cudaChannelFormatKind enum values
  {"cudaChannelFormatKindSigned",                                      {"hipChannelFormatKindSigned",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  {"cudaChannelFormatKindUnsigned",                                    {"hipChannelFormatKindUnsigned",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1
  {"cudaChannelFormatKindFloat",                                       {"hipChannelFormatKindFloat",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 2
  {"cudaChannelFormatKindNone",                                        {"hipChannelFormatKindNone",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 3

  // CUcomputemode
  {"cudaComputeMode",                                                  {"hipComputeMode",                                           "", CONV_TYPE, API_RUNTIME}},
  // cudaComputeMode enum values
  // CU_COMPUTEMODE_DEFAULT
  {"cudaComputeModeDefault",                                           {"hipComputeModeDefault",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  // CU_COMPUTEMODE_EXCLUSIVE
  {"cudaComputeModeExclusive",                                         {"hipComputeModeExclusive",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1
  // CU_COMPUTEMODE_PROHIBITED
  {"cudaComputeModeProhibited",                                        {"hipComputeModeProhibited",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 2
  // CU_COMPUTEMODE_EXCLUSIVE_PROCESS
  {"cudaComputeModeExclusiveProcess",                                  {"hipComputeModeExclusiveProcess",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 3

  // CUdevice_attribute
  {"cudaDeviceAttr",                                                   {"hipDeviceAttribute_t",                                     "", CONV_TYPE, API_RUNTIME}},
  // cudaDeviceAttr enum values
  // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
  {"cudaDevAttrMaxThreadsPerBlock",                                    {"hipDeviceAttributeMaxThreadsPerBlock",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, //  1
  // CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
  {"cudaDevAttrMaxBlockDimX",                                          {"hipDeviceAttributeMaxBlockDimX",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, //  2
  // CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
  {"cudaDevAttrMaxBlockDimY",                                          {"hipDeviceAttributeMaxBlockDimY",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, //  3
  // CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
  {"cudaDevAttrMaxBlockDimZ",                                          {"hipDeviceAttributeMaxBlockDimZ",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, //  4
  // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
  {"cudaDevAttrMaxGridDimX",                                           {"hipDeviceAttributeMaxGridDimX",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, //  5
  // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
  {"cudaDevAttrMaxGridDimY",                                           {"hipDeviceAttributeMaxGridDimY",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, //  6
  // CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
  {"cudaDevAttrMaxGridDimZ",                                           {"hipDeviceAttributeMaxGridDimZ",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, //  7
  // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
  {"cudaDevAttrMaxSharedMemoryPerBlock",                               {"hipDeviceAttributeMaxSharedMemoryPerBlock",                "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, //  8
  // CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
  {"cudaDevAttrTotalConstantMemory",                                   {"hipDeviceAttributeTotalConstantMemory",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, //  9
  // CU_DEVICE_ATTRIBUTE_WARP_SIZE
  {"cudaDevAttrWarpSize",                                              {"hipDeviceAttributeWarpSize",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 10
  // CU_DEVICE_ATTRIBUTE_MAX_PITCH
  {"cudaDevAttrMaxPitch",                                              {"hipDeviceAttributeMaxPitch",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 11
  // CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
  {"cudaDevAttrMaxRegistersPerBlock",                                  {"hipDeviceAttributeMaxRegistersPerBlock",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 12
  // CU_DEVICE_ATTRIBUTE_CLOCK_RATE
  {"cudaDevAttrClockRate",                                             {"hipDeviceAttributeClockRate",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 13
  // CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT
  {"cudaDevAttrTextureAlignment",                                      {"hipDeviceAttributeTextureAlignment",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 14
  // CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
  // NOTE: Is not deprecated as CUDA Driver's API analogue CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
  {"cudaDevAttrGpuOverlap",                                            {"hipDeviceAttributeGpuOverlap",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 15
  // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
  {"cudaDevAttrMultiProcessorCount",                                   {"hipDeviceAttributeMultiprocessorCount",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 16
  // CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT
  {"cudaDevAttrKernelExecTimeout",                                     {"hipDeviceAttributeKernelExecTimeout",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 17
  // CU_DEVICE_ATTRIBUTE_INTEGRATED
  {"cudaDevAttrIntegrated",                                            {"hipDeviceAttributeIntegrated",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 18
  // CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY
  {"cudaDevAttrCanMapHostMemory",                                      {"hipDeviceAttributeCanMapHostMemory",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 19
  // CU_DEVICE_ATTRIBUTE_COMPUTE_MODE
  {"cudaDevAttrComputeMode",                                           {"hipDeviceAttributeComputeMode",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 20
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH
  {"cudaDevAttrMaxTexture1DWidth",                                     {"hipDeviceAttributeMaxTexture1DWidth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 21
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH
  {"cudaDevAttrMaxTexture2DWidth",                                     {"hipDeviceAttributeMaxTexture2DWidth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 22
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT
  {"cudaDevAttrMaxTexture2DHeight",                                    {"hipDeviceAttributeMaxTexture2DHeight",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 23
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH
  {"cudaDevAttrMaxTexture3DWidth",                                     {"hipDeviceAttributeMaxTexture3DWidth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 24
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT
  {"cudaDevAttrMaxTexture3DHeight",                                    {"hipDeviceAttributeMaxTexture3DHeight",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 25
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH
  {"cudaDevAttrMaxTexture3DDepth",                                     {"hipDeviceAttributeMaxTexture3DDepth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 26
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH
  {"cudaDevAttrMaxTexture2DLayeredWidth",                              {"hipDeviceAttributeMaxTexture2DLayeredWidth",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 27
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT
  {"cudaDevAttrMaxTexture2DLayeredHeight",                             {"hipDeviceAttributeMaxTexture2DLayeredHeight",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 28
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS
  {"cudaDevAttrMaxTexture2DLayeredLayers",                             {"hipDeviceAttributeMaxTexture2DLayeredLayers",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 29
  // CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT
  {"cudaDevAttrSurfaceAlignment",                                      {"hipDeviceAttributeSurfaceAlignment",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 30
  // CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS
  {"cudaDevAttrConcurrentKernels",                                     {"hipDeviceAttributeConcurrentKernels",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 31
  // CU_DEVICE_ATTRIBUTE_ECC_ENABLED
  {"cudaDevAttrEccEnabled",                                            {"hipDeviceAttributeEccEnabled",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 32
  // CU_DEVICE_ATTRIBUTE_PCI_BUS_ID
  {"cudaDevAttrPciBusId",                                              {"hipDeviceAttributePciBusId",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 33
  // CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID
  {"cudaDevAttrPciDeviceId",                                           {"hipDeviceAttributePciDeviceId",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 34
  // CU_DEVICE_ATTRIBUTE_TCC_DRIVER
  {"cudaDevAttrTccDriver",                                             {"hipDeviceAttributeTccDriver",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 35
  // CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE
  {"cudaDevAttrMemoryClockRate",                                       {"hipDeviceAttributeMemoryClockRate",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 36
  // CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH
  {"cudaDevAttrGlobalMemoryBusWidth",                                  {"hipDeviceAttributeMemoryBusWidth",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 37
  // CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE
  {"cudaDevAttrL2CacheSize",                                           {"hipDeviceAttributeL2CacheSize",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 38
  // CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
  {"cudaDevAttrMaxThreadsPerMultiProcessor",                           {"hipDeviceAttributeMaxThreadsPerMultiProcessor",            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 39
  // CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT
  {"cudaDevAttrAsyncEngineCount",                                      {"hipDeviceAttributeAsyncEngineCount",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 40
  // CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING
  {"cudaDevAttrUnifiedAddressing",                                     {"hipDeviceAttributeUnifiedAddressing",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 41
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH
  {"cudaDevAttrMaxTexture1DLayeredWidth",                              {"hipDeviceAttributeMaxTexture1DLayeredWidth",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 42
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS
  {"cudaDevAttrMaxTexture1DLayeredLayers",                             {"hipDeviceAttributeMaxTexture1DLayeredLayers",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 43
  // 44 - no
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH
  {"cudaDevAttrMaxTexture2DGatherWidth",                               {"hipDeviceAttributeMaxTexture2DGatherWidth",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 45
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT
  {"cudaDevAttrMaxTexture2DGatherHeight",                              {"hipDeviceAttributeMaxTexture2DGatherHeight",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 46
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE
  {"cudaDevAttrMaxTexture3DWidthAlt",                                  {"hipDeviceAttributeMaxTexture3DWidthAlternate",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 47
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE
  {"cudaDevAttrMaxTexture3DHeightAlt",                                 {"hipDeviceAttributeMaxTexture3DHeightAlternate",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 48
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE
  {"cudaDevAttrMaxTexture3DDepthAlt",                                  {"hipDeviceAttributeMaxTexture3DDepthAlternate",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 49
  // CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID
  {"cudaDevAttrPciDomainId",                                           {"hipDeviceAttributePciDomainId",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 50
  // CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT
  {"cudaDevAttrTexturePitchAlignment",                                 {"hipDeviceAttributeTexturePitchAlignment",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 51
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH
  {"cudaDevAttrMaxTextureCubemapWidth",                                {"hipDeviceAttributeMaxTextureCubemapWidth",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 52
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH
  {"cudaDevAttrMaxTextureCubemapLayeredWidth",                         {"hipDeviceAttributeMaxTextureCubemapLayeredWidth",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 53
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS
  {"cudaDevAttrMaxTextureCubemapLayeredLayers",                        {"hipDeviceAttributeMaxTextureCubemapLayeredLayers",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 54
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH
  {"cudaDevAttrMaxSurface1DWidth",                                     {"hipDeviceAttributeMaxSurface1DWidth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 55
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH
  {"cudaDevAttrMaxSurface2DWidth",                                     {"hipDeviceAttributeMaxSurface2DWidth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 56
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT
  {"cudaDevAttrMaxSurface2DHeight",                                    {"hipDeviceAttributeMaxSurface2DHeight",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 57
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH
  {"cudaDevAttrMaxSurface3DWidth",                                     {"hipDeviceAttributeMaxSurface3DWidth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 58
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT
  {"cudaDevAttrMaxSurface3DHeight",                                    {"hipDeviceAttributeMaxSurface3DHeight",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 59
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH
  {"cudaDevAttrMaxSurface3DDepth",                                     {"hipDeviceAttributeMaxSurface3DDepth",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 60
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH
  {"cudaDevAttrMaxSurface1DLayeredWidth",                              {"hipDeviceAttributeMaxSurface1DLayeredWidth",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 61
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS
  {"cudaDevAttrMaxSurface1DLayeredLayers",                             {"hipDeviceAttributeMaxSurface1DLayeredLayers",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 62
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH
  {"cudaDevAttrMaxSurface2DLayeredWidth",                              {"hipDeviceAttributeMaxSurface2DLayeredWidth",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 63
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT
  {"cudaDevAttrMaxSurface2DLayeredHeight",                             {"hipDeviceAttributeMaxSurface2DLayeredHeight",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 64
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LA  YERS
  {"cudaDevAttrMaxSurface2DLayeredLayers",                             {"hipDeviceAttributeMaxSurface2DLayeredLayers",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 65
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH
  {"cudaDevAttrMaxSurfaceCubemapWidth",                                {"hipDeviceAttributeMaxSurfaceCubemapWidth",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 66
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH
  {"cudaDevAttrMaxSurfaceCubemapLayeredWidth",                         {"hipDeviceAttributeMaxSurfaceCubemapLayeredWidth",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 67
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS
  {"cudaDevAttrMaxSurfaceCubemapLayeredLayers",                        {"hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 68
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH
  {"cudaDevAttrMaxTexture1DLinearWidth",                               {"hipDeviceAttributeMaxTexture1DLinearWidth",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 69
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH
  {"cudaDevAttrMaxTexture2DLinearWidth",                               {"hipDeviceAttributeMaxTexture2DLinearWidth",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 70
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT
  {"cudaDevAttrMaxTexture2DLinearHeight",                              {"hipDeviceAttributeMaxTexture2DLinearHeight",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 71
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH
  {"cudaDevAttrMaxTexture2DLinearPitch",                               {"hipDeviceAttributeMaxTexture2DLinearPitch",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 72
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH
  {"cudaDevAttrMaxTexture2DMipmappedWidth",                            {"hipDeviceAttributeMaxTexture2DMipmappedWidth",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 73
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT
  {"cudaDevAttrMaxTexture2DMipmappedHeight",                           {"hipDeviceAttributeMaxTexture2DMipmappedHeight",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 74
  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
  {"cudaDevAttrComputeCapabilityMajor",                                {"hipDeviceAttributeComputeCapabilityMajor",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 75
  // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
  {"cudaDevAttrComputeCapabilityMinor",                                {"hipDeviceAttributeComputeCapabilityMinor",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 76
  // CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH
  {"cudaDevAttrMaxTexture1DMipmappedWidth",                            {"hipDeviceAttributeMaxTexture1DMipmappedWidth",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 77
  // CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED
  {"cudaDevAttrStreamPrioritiesSupported",                             {"hipDeviceAttributeStreamPrioritiesSupported",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 78
  // CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED
  {"cudaDevAttrGlobalL1CacheSupported",                                {"hipDeviceAttributeGlobalL1CacheSupported",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 79
  // CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED
  {"cudaDevAttrLocalL1CacheSupported",                                 {"hipDeviceAttributeLocalL1CacheSupported",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 80
  // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
  {"cudaDevAttrMaxSharedMemoryPerMultiprocessor",                      {"hipDeviceAttributeMaxSharedMemoryPerMultiprocessor",       "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 81
  // CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
  {"cudaDevAttrMaxRegistersPerMultiprocessor",                         {"hipDeviceAttributeMaxRegistersPerMultiprocessor",          "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 82
  // CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY
  {"cudaDevAttrManagedMemory",                                         {"hipDeviceAttributeManagedMemory",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 83
  // CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD
  {"cudaDevAttrIsMultiGpuBoard",                                       {"hipDeviceAttributeIsMultiGpuBoard",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 84
  // CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID
  {"cudaDevAttrMultiGpuBoardGroupID",                                  {"hipDeviceAttributeMultiGpuBoardGroupID",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 85
  // CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED
  {"cudaDevAttrHostNativeAtomicSupported",                             {"hipDeviceAttributeHostNativeAtomicSupported",              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 86
  // CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
  {"cudaDevAttrSingleToDoublePrecisionPerfRatio",                      {"hipDeviceAttributeSingleToDoublePrecisionPerfRatio",       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 87
  // CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS
  {"cudaDevAttrPageableMemoryAccess",                                  {"hipDeviceAttributePageableMemoryAccess",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 88
  // CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS
  {"cudaDevAttrConcurrentManagedAccess",                               {"hipDeviceAttributeConcurrentManagedAccess",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 89
  // CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED
  {"cudaDevAttrComputePreemptionSupported",                            {"hipDeviceAttributeComputePreemptionSupported",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 90
  // CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
  {"cudaDevAttrCanUseHostPointerForRegisteredMem",                     {"hipDeviceAttributeCanUseHostPointerForRegisteredMem",      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 91
  // CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS
  {"cudaDevAttrReserved92",                                            {"hipDeviceAttributeCanUseStreamMemOps",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 92
  // CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS
  {"cudaDevAttrReserved93",                                            {"hipDeviceAttributeCanUse64BitStreamMemOps",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 93
  // CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR
  {"cudaDevAttrReserved94",                                            {"hipDeviceAttributeCanUseStreamWaitValueNor",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 94
  // CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH
  {"cudaDevAttrCooperativeLaunch",                                     {"hipDeviceAttributeCooperativeLaunch",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 95
  // CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH
  {"cudaDevAttrCooperativeMultiDeviceLaunch",                          {"hipDeviceAttributeCooperativeMultiDeviceLaunch",           "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 96
  // CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
  {"cudaDevAttrMaxSharedMemoryPerBlockOptin",                          {"hipDeviceAttributeMaxSharedMemoryPerBlockOptin",           "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 97
  // CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES
  {"cudaDevAttrCanFlushRemoteWrites",                                  {"hipDeviceAttributeCanFlushRemoteWrites",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 98
  // CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED
  {"cudaDevAttrHostRegisterSupported",                                 {"hipDeviceAttributeHostRegisterSupported",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 99
  // CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
  {"cudaDevAttrPageableMemoryAccessUsesHostPageTables",                {"hipDeviceAttributePageableMemoryAccessUsesHostPageTables", "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 100
  // CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST
  {"cudaDevAttrDirectManagedMemAccessFromHost",                        {"hipDeviceAttributeDirectManagedMemAccessFromHost",         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 101

  // CUdevice_P2PAttribute
  {"cudaDeviceP2PAttr",                                                {"hipDeviceP2PAttribute",                                    "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaDeviceP2PAttr enum values
  // CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01
  {"cudaDevP2PAttrPerformanceRank",                                    {"hipDeviceP2PAttributePerformanceRank",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02
  {"cudaDevP2PAttrAccessSupported",                                    {"hipDeviceP2PAttributeAccessSupported",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03
  {"cudaDevP2PAttrNativeAtomicSupported",                              {"hipDeviceP2PAttributeNativeAtomicSupported",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3
  // CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED = 0x04
  {"cudaDevP2PAttrCudaArrayAccessSupported",                           {"hipDevP2PAttributeCudaArrayAccessSupported",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 4

  // cudaEGL.h - presented only on Linux in nvidia-cuda-dev package
  // CUeglColorFormat
  {"cudaEglColorFormat",                                               {"hipEglColorFormat",                                        "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaEglColorFormat enum values
  // CU_EGL_COLOR_FORMAT_YUV420_PLANAR = 0x00
  {"cudaEglColorFormatYUV420Planar",                                   {"hipEglColorFormatYUV420Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR= 0x01
  {"cudaEglColorFormatYUV420SemiPlanar ",                              {"hipEglColorFormatYUV420SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_EGL_COLOR_FORMAT_YUV422_PLANAR = 0x02
  {"cudaEglColorFormatYUV422Planar",                                   {"hipEglColorFormatYUV422Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR = 0x03
  {"cudaEglColorFormatYUV422SemiPlanar",                               {"hipEglColorFormatYUV422SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3
  // CU_EGL_COLOR_FORMAT_RGB = 0x04
  {"cudaEglColorFormatRGB",                                            {"hipEglColorFormatRGB",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 4
  // CU_EGL_COLOR_FORMAT_BGR = 0x05
  {"cudaEglColorFormatBGR",                                            {"hipEglColorFormatBGR",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 5
  // CU_EGL_COLOR_FORMAT_ARGB = 0x06
  {"cudaEglColorFormatARGB",                                           {"hipEglColorFormatARGB",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 6
  // CU_EGL_COLOR_FORMAT_RGBA = 0x07
  {"cudaEglColorFormatRGBA",                                           {"hipEglColorFormatRGBA",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 7
  // CU_EGL_COLOR_FORMAT_L = 0x08
  {"cudaEglColorFormatL",                                              {"hipEglColorFormatL",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 8
  // CU_EGL_COLOR_FORMAT_R = 0x09
  {"cudaEglColorFormatR",                                              {"hipEglColorFormatR",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 9
  // CU_EGL_COLOR_FORMAT_YUV444_PLANAR = 0x0A
  {"cudaEglColorFormatYUV444Planar",                                   {"hipEglColorFormatYUV444Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 10
  // CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR = 0x0B
  {"cudaEglColorFormatYUV444SemiPlanar",                               {"hipEglColorFormatYUV444SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 11
  // CU_EGL_COLOR_FORMAT_YUYV_422 = 0x0C
  {"cudaEglColorFormatYUYV422",                                        {"hipEglColorFormatYUYV422",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 12
  // CU_EGL_COLOR_FORMAT_UYVY_422 = 0x0D
  {"cudaEglColorFormatUYVY422",                                        {"hipEglColorFormatUYVY422",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 13
  // CU_EGL_COLOR_FORMAT_ABGR = 0x0E
  {"cudaEglColorFormatABGR",                                           {"hipEglColorFormatABGR",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 14
  // CU_EGL_COLOR_FORMAT_BGRA = 0x0F
  {"cudaEglColorFormatBGRA",                                           {"hipEglColorFormatBGRA",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 15
  // CU_EGL_COLOR_FORMAT_A = 0x10
  {"cudaEglColorFormatA",                                              {"hipEglColorFormatA",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 16
  // CU_EGL_COLOR_FORMAT_RG = 0x11
  {"cudaEglColorFormatRG",                                             {"hipEglColorFormatRG",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 17
  // CU_EGL_COLOR_FORMAT_AYUV = 0x12
  {"cudaEglColorFormatAYUV",                                           {"hipEglColorFormatAYUV",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 18
  // CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR = 0x13
  {"cudaEglColorFormatYVU444SemiPlanar",                               {"hipEglColorFormatYVU444SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 19
  // CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR = 0x14
  {"cudaEglColorFormatYVU422SemiPlanar",                               {"hipEglColorFormatYVU422SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 20
  // CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR = 0x15
  {"cudaEglColorFormatYVU420SemiPlanar",                               {"hipEglColorFormatYVU420SemiPlanar",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 21
  // CU_EGL_COLOR_FORMAT_Y10V10U10_444_SEMIPLANAR = 0x16
  {"cudaEglColorFormatY10V10U10_444SemiPlanar",                        {"hipEglColorFormatY10V10U10_444SemiPlanar",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 22
  // CU_EGL_COLOR_FORMAT_Y10V10U10_420_SEMIPLANAR = 0x17
  {"cudaEglColorFormatY10V10U10_420SemiPlanar",                        {"hipEglColorFormatY10V10U10_420SemiPlanar",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 23
  // CU_EGL_COLOR_FORMAT_Y12V12U12_444_SEMIPLANAR = 0x18
  {"cudaEglColorFormatY12V12U12_444SemiPlanar",                        {"hipEglColorFormatY12V12U12_444SemiPlanar",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 24
  // CU_EGL_COLOR_FORMAT_Y12V12U12_420_SEMIPLANAR = 0x19
  {"cudaEglColorFormatY12V12U12_420SemiPlanar",                        {"hipEglColorFormatY12V12U12_420SemiPlanar",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 25
  // CU_EGL_COLOR_FORMAT_VYUY_ER = 0x1A
  {"cudaEglColorFormatVYUY_ER",                                        {"hipEglColorFormatVYUY_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 26
  // CU_EGL_COLOR_FORMAT_UYVY_ER = 0x1B
  {"cudaEglColorFormatUYVY_ER",                                        {"hipEglColorFormatUYVY_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 27
  // CU_EGL_COLOR_FORMAT_YUYV_ER = 0x1C
  {"cudaEglColorFormatYUYV_ER",                                        {"hipEglColorFormatYUYV_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 28
  // CU_EGL_COLOR_FORMAT_YVYU_ER = 0x1D
  {"cudaEglColorFormatYVYU_ER",                                        {"hipEglColorFormatYVYU_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 29
  // CU_EGL_COLOR_FORMAT_YUV_ER = 0x1E
  {"cudaEglColorFormatYUV_ER",                                         {"hipEglColorFormatYUV_ER",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 30
  // CU_EGL_COLOR_FORMAT_YUVA_ER = 0x1F
  {"cudaEglColorFormatYUVA_ER",                                        {"hipEglColorFormatYUVA_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 31
  // CU_EGL_COLOR_FORMAT_AYUV_ER = 0x20
  {"cudaEglColorFormatAYUV_ER",                                        {"hipEglColorFormatAYUV_ER",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 32
  // CU_EGL_COLOR_FORMAT_YUV444_PLANAR_ER = 0x21
  {"cudaEglColorFormatYUV444Planar_ER",                                {"hipEglColorFormatYUV444Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 33
  // CU_EGL_COLOR_FORMAT_YUV422_PLANAR_ER = 0x22
  {"cudaEglColorFormatYUV422Planar_ER",                                {"hipEglColorFormatYUV422Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 34
  // CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER = 0x23
  {"cudaEglColorFormatYUV420Planar_ER",                                {"hipEglColorFormatYUV420Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 35
  // CU_EGL_COLOR_FORMAT_YUV444_SEMIPLANAR_ER = 0x24
  {"cudaEglColorFormatYUV444SemiPlanar_ER",                            {"hipEglColorFormatYUV444SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 36
  // CU_EGL_COLOR_FORMAT_YUV422_SEMIPLANAR_ER = 0x25
  {"cudaEglColorFormatYUV422SemiPlanar_ER",                            {"hipEglColorFormatYUV422SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 37
  // CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER = 0x26
  {"cudaEglColorFormatYUV420SemiPlanar_ER",                            {"hipEglColorFormatYUV420SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 38
  // CU_EGL_COLOR_FORMAT_YVU444_PLANAR_ER = 0x27
  {"cudaEglColorFormatYVU444Planar_ER",                                {"hipEglColorFormatYVU444Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 39
  // CU_EGL_COLOR_FORMAT_YVU422_PLANAR_ER = 0x28
  {"cudaEglColorFormatYVU422Planar_ER",                                {"hipEglColorFormatYVU422Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 40
  // CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER = 0x29
  {"cudaEglColorFormatYVU420Planar_ER",                                {"hipEglColorFormatYVU420Planar_ER",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 41
  // CU_EGL_COLOR_FORMAT_YVU444_SEMIPLANAR_ER = 0x2A
  {"cudaEglColorFormatYVU444SemiPlanar_ER",                            {"hipEglColorFormatYVU444SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 42
  // CU_EGL_COLOR_FORMAT_YVU422_SEMIPLANAR_ER = 0x2B
  {"cudaEglColorFormatYVU422SemiPlanar_ER",                            {"hipEglColorFormatYVU422SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 43
  // CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER = 0x2C
  {"cudaEglColorFormatYVU420SemiPlanar_ER",                            {"hipEglColorFormatYVU420SemiPlanar_ER",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 44
  // CU_EGL_COLOR_FORMAT_BAYER_RGGB = 0x2D
  {"cudaEglColorFormatBayerRGGB",                                      {"hipEglColorFormatBayerRGGB",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 45
  // CU_EGL_COLOR_FORMAT_BAYER_BGGR = 0x2E
  {"cudaEglColorFormatBayerBGGR",                                      {"hipEglColorFormatBayerBGGR",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 46
  // CU_EGL_COLOR_FORMAT_BAYER_GRBG = 0x2F
  {"cudaEglColorFormatBayerGRBG",                                      {"hipEglColorFormatBayerGRBG",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 47
  // CU_EGL_COLOR_FORMAT_BAYER_GBRG = 0x30
  {"cudaEglColorFormatBayerGBRG",                                      {"hipEglColorFormatBayerGBRG",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 48
  // CU_EGL_COLOR_FORMAT_BAYER10_RGGB = 0x31
  {"cudaEglColorFormatBayer10RGGB",                                    {"hipEglColorFormatBayer10RGGB",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 49
  // CU_EGL_COLOR_FORMAT_BAYER10_BGGR = 0x32
  {"cudaEglColorFormatBayer10BGGR",                                    {"hipEglColorFormatBayer10BGGR",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 50
  // CU_EGL_COLOR_FORMAT_BAYER10_GRBG = 0x33
  {"cudaEglColorFormatBayer10GRBG",                                    {"hipEglColorFormatBayer10GRBG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 51
  // CU_EGL_COLOR_FORMAT_BAYER10_GBRG = 0x34
  {"cudaEglColorFormatBayer10GBRG",                                    {"hipEglColorFormatBayer10GBRG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 52
  // CU_EGL_COLOR_FORMAT_BAYER12_RGGB = 0x35
  {"cudaEglColorFormatBayer12RGGB",                                    {"hipEglColorFormatBayer12RGGB",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 53
  // CU_EGL_COLOR_FORMAT_BAYER12_BGGR = 0x36
  {"cudaEglColorFormatBayer12BGGR",                                    {"hipEglColorFormatBayer12BGGR",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 54
  // CU_EGL_COLOR_FORMAT_BAYER12_GRBG = 0x37
  {"cudaEglColorFormatBayer12GRBG",                                    {"hipEglColorFormatBayer12GRBG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 55
  // CU_EGL_COLOR_FORMAT_BAYER12_GBRG = 0x38
  {"cudaEglColorFormatBayer12GBRG",                                    {"hipEglColorFormatBayer12GBRG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 56
  // CU_EGL_COLOR_FORMAT_BAYER14_RGGB = 0x39
  {"cudaEglColorFormatBayer14RGGB",                                    {"hipEglColorFormatBayer14RGGB",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 57
  // CU_EGL_COLOR_FORMAT_BAYER14_BGGR = 0x3A
  {"cudaEglColorFormatBayer14BGGR",                                    {"hipEglColorFormatBayer14BGGR",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 58
  // CU_EGL_COLOR_FORMAT_BAYER14_GRBG = 0x3B
  {"cudaEglColorFormatBayer14GRBG",                                    {"hipEglColorFormatBayer14GRBG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 59
  // CU_EGL_COLOR_FORMAT_BAYER14_GBRG = 0x3C
  {"cudaEglColorFormatBayer14GBRG",                                    {"hipEglColorFormatBayer14GBRG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 60
  // CU_EGL_COLOR_FORMAT_BAYER20_RGGB = 0x3D
  {"cudaEglColorFormatBayer20RGGB",                                    {"hipEglColorFormatBayer20RGGB",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 61
  // CU_EGL_COLOR_FORMAT_BAYER20_BGGR = 0x3E
  {"cudaEglColorFormatBayer20BGGR",                                    {"hipEglColorFormatBayer20BGGR",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 62
  // CU_EGL_COLOR_FORMAT_BAYER20_GRBG = 0x3F
  {"cudaEglColorFormatBayer20GRBG",                                    {"hipEglColorFormatBayer20GRBG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 63
  // CU_EGL_COLOR_FORMAT_BAYER20_GBRG = 0x40
  {"cudaEglColorFormatBayer20GBRG",                                    {"hipEglColorFormatBayer20GBRG",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 64
  // CU_EGL_COLOR_FORMAT_YVU444_PLANAR = 0x41
  {"cudaEglColorFormatYVU444Planar",                                   {"hipEglColorFormatYVU444Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 65
  // CU_EGL_COLOR_FORMAT_YVU422_PLANAR = 0x42
  {"cudaEglColorFormatYVU422Planar",                                   {"hipEglColorFormatYVU422Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 66
  // CU_EGL_COLOR_FORMAT_YVU420_PLANAR = 0x43
  {"cudaEglColorFormatYVU420Planar",                                   {"hipEglColorFormatYVU420Planar",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 67
  // CU_EGL_COLOR_FORMAT_BAYER_ISP_RGGB = 0x44
  {"cudaEglColorFormatBayerIspRGGB",                                   {"hipEglColorFormatBayerIspRGGB",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 68
  // CU_EGL_COLOR_FORMAT_BAYER_ISP_BGGR = 0x45
  {"cudaEglColorFormatBayerIspBGGR",                                   {"hipEglColorFormatBayerIspBGGR",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 69
  // CU_EGL_COLOR_FORMAT_BAYER_ISP_GRBG = 0x46
  {"cudaEglColorFormatBayerIspGRBG",                                   {"hipEglColorFormatBayerIspGRBG",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 70
  // CU_EGL_COLOR_FORMAT_BAYER_ISP_GBRG = 0x47
  {"cudaEglColorFormatBayerIspGBRG",                                   {"hipEglColorFormatBayerIspGBRG",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 71

  // CUeglFrameType
  {"cudaEglFrameType",                                                 {"hipEglFrameType",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaEglFrameType enum values
  // CU_EGL_FRAME_TYPE_ARRAY
  {"cudaEglFrameTypeArray",                                            {"hipEglFrameTypeArray",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_EGL_FRAME_TYPE_PITCH
  {"cudaEglFrameTypePitch",                                            {"hipEglFrameTypePitch",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1

  // CUeglResourceLocationFlags
  {"cudaEglResourceLocationFlags",                                     {"hipEglResourceLocationFlags",                              "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaEglResourceLocationFlagss enum values
  // CU_EGL_RESOURCE_LOCATION_SYSMEM
  {"cudaEglResourceLocationSysmem",                                    {"hipEglResourceLocationSysmem",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x00
  // CU_EGL_RESOURCE_LOCATION_VIDMEM
  {"cudaEglResourceLocationVidmem",                                    {"hipEglResourceLocationVidmem",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x01

  // CUresult
  {"cudaError",                                                        {"hipError_t",                                               "", CONV_TYPE, API_RUNTIME}},
  {"cudaError_t",                                                      {"hipError_t",                                               "", CONV_TYPE, API_RUNTIME}},
  // cudaError enum values
  // CUDA_SUCCESS = 0
  {"cudaSuccess",                                                      {"hipSuccess",                                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  // no analogue
  {"cudaErrorMissingConfiguration",                                    {"hipErrorMissingConfiguration",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1
  // CUDA_ERROR_OUT_OF_MEMORY = 2
  {"cudaErrorMemoryAllocation",                                        {"hipErrorMemoryAllocation",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 2
  // CUDA_ERROR_NOT_INITIALIZED = 3
  {"cudaErrorInitializationError",                                     {"hipErrorInitializationError",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 3
  // CUDA_ERROR_LAUNCH_FAILED = 719
  {"cudaErrorLaunchFailure",                                           {"hipErrorLaunchFailure",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 4
  // no analogue
  {"cudaErrorPriorLaunchFailure",                                      {"hipErrorPriorLaunchFailure",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 5
  // CUDA_ERROR_LAUNCH_TIMEOUT = 702
  {"cudaErrorLaunchTimeout",                                           {"hipErrorLaunchTimeOut",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 6
  // CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
  {"cudaErrorLaunchOutOfResources",                                    {"hipErrorLaunchOutOfResources",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 7
  // no analogue
  {"cudaErrorInvalidDeviceFunction",                                   {"hipErrorInvalidDeviceFunction",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 8
  // no analogue
  {"cudaErrorInvalidConfiguration",                                    {"hipErrorInvalidConfiguration",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 9
  // CUDA_ERROR_INVALID_DEVICE = 101
  {"cudaErrorInvalidDevice",                                           {"hipErrorInvalidDevice",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 10
  // CUDA_ERROR_INVALID_VALUE = 1
  {"cudaErrorInvalidValue",                                            {"hipErrorInvalidValue",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 11
  // no analogue
  {"cudaErrorInvalidPitchValue",                                       {"hipErrorInvalidPitchValue",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 12
  // no analogue
  {"cudaErrorInvalidSymbol",                                           {"hipErrorInvalidSymbol",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 13
  // CUDA_ERROR_MAP_FAILED = 205
  // TODO: double check the matching
  {"cudaErrorMapBufferObjectFailed",                                   {"hipErrorMapFailed",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 14
  // CUDA_ERROR_UNMAP_FAILED = 206
  // TODO: double check the matching
  {"cudaErrorUnmapBufferObjectFailed",                                 {"hipErrorUnmapFailed",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 15
  // no analogue
  {"cudaErrorInvalidHostPointer",                                      {"hipErrorInvalidHostPointer",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 16
  // no analogue
  {"cudaErrorInvalidDevicePointer",                                    {"hipErrorInvalidDevicePointer",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 17
  // no analogue
  {"cudaErrorInvalidTexture",                                          {"hipErrorInvalidTexture",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 18
  // no analogue
  {"cudaErrorInvalidTextureBinding",                                   {"hipErrorInvalidTextureBinding",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 19
  // no analogue
  {"cudaErrorInvalidChannelDescriptor",                                {"hipErrorInvalidChannelDescriptor",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 20
  // no analogue
  {"cudaErrorInvalidMemcpyDirection",                                  {"hipErrorInvalidMemcpyDirection",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 21
  // no analogue
  {"cudaErrorAddressOfConstant",                                       {"hipErrorAddressOfConstant",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 22
  // no analogue
  {"cudaErrorTextureFetchFailed",                                      {"hipErrorTextureFetchFailed",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 23
  // no analogue
  {"cudaErrorTextureNotBound",                                         {"hipErrorTextureNotBound",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 24
  // no analogue
  {"cudaErrorSynchronizationError",                                    {"hipErrorSynchronizationError",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 25
  // no analogue
  {"cudaErrorInvalidFilterSetting",                                    {"hipErrorInvalidFilterSetting",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 26
  // no analogue
  {"cudaErrorInvalidNormSetting",                                      {"hipErrorInvalidNormSetting",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 27
  // no analogue
  {"cudaErrorMixedDeviceExecution",                                    {"hipErrorMixedDeviceExecution",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 28
  // CUDA_ERROR_DEINITIALIZED = 4
  {"cudaErrorCudartUnloading",                                         {"hipErrorDeinitialized",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 29
  // CUDA_ERROR_UNKNOWN = 999
  {"cudaErrorUnknown",                                                 {"hipErrorUnknown",                                          "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 30
  // Deprecated since CUDA 4.1
  // no analogue
  {"cudaErrorNotYetImplemented",                                       {"hipErrorNotYetImplemented",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 31
  // Deprecated since CUDA 3.1
  // no analogue
  {"cudaErrorMemoryValueTooLarge",                                     {"hipErrorMemoryValueTooLarge",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 32
  // CUDA_ERROR_INVALID_HANDLE = 400
  {"cudaErrorInvalidResourceHandle",                                   {"hipErrorInvalidResourceHandle",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 33
  // CUDA_ERROR_NOT_READY = 600
  {"cudaErrorNotReady",                                                {"hipErrorNotReady",                                         "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 34
  // no analogue
  {"cudaErrorInsufficientDriver",                                      {"hipErrorInsufficientDriver",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 35
  // no analogue
  {"cudaErrorSetOnActiveProcess",                                      {"hipErrorSetOnActiveProcess",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 36
  // no analogue
  {"cudaErrorInvalidSurface",                                          {"hipErrorInvalidSurface",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 37
  // CUDA_ERROR_NO_DEVICE = 100
  {"cudaErrorNoDevice",                                                {"hipErrorNoDevice",                                         "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 38
  // CUDA_ERROR_ECC_UNCORRECTABLE = 214
  {"cudaErrorECCUncorrectable",                                        {"hipErrorECCNotCorrectable",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 39
  // CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
  {"cudaErrorSharedObjectSymbolNotFound",                              {"hipErrorSharedObjectSymbolNotFound",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 40
  // CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
  {"cudaErrorSharedObjectInitFailed",                                  {"hipErrorSharedObjectInitFailed",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 41
  // CUDA_ERROR_UNSUPPORTED_LIMIT = 215
  {"cudaErrorUnsupportedLimit",                                        {"hipErrorUnsupportedLimit",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 42
  // no analogue
  {"cudaErrorDuplicateVariableName",                                   {"hipErrorDuplicateVariableName",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 43
  // no analogue
  {"cudaErrorDuplicateTextureName",                                    {"hipErrorDuplicateTextureName",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 44
  // no analogue
  {"cudaErrorDuplicateSurfaceName",                                    {"hipErrorDuplicateSurfaceName",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 45
  // no analogue
  {"cudaErrorDevicesUnavailable",                                      {"hipErrorDevicesUnavailable",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 46
  // CUDA_ERROR_INVALID_IMAGE = 200
  {"cudaErrorInvalidKernelImage",                                      {"hipErrorInvalidImage",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 47
  // CUDA_ERROR_NO_BINARY_FOR_GPU = 209
  {"cudaErrorNoKernelImageForDevice",                                  {"hipErrorNoBinaryForGpu",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 48
  // no analogue
  {"cudaErrorIncompatibleDriverContext",                               {"hipErrorIncompatibleDriverContext",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 49
  // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
  {"cudaErrorPeerAccessAlreadyEnabled",                                {"hipErrorPeerAccessAlreadyEnabled",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 50
  // CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
  {"cudaErrorPeerAccessNotEnabled",                                    {"hipErrorPeerAccessNotEnabled",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 51
  // no analogue
  {"cudaErrorDeviceAlreadyInUse",                                      {"hipErrorDeviceAlreadyInUse",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 54
  // CUDA_ERROR_PROFILER_DISABLED = 5
  {"cudaErrorProfilerDisabled",                                        {"hipErrorProfilerDisabled",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 55
  // Deprecated since CUDA 5.0
  // CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
  {"cudaErrorProfilerNotInitialized",                                  {"hipErrorProfilerNotInitialized",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 56
  // Deprecated since CUDA 5.0
  // CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
  {"cudaErrorProfilerAlreadyStarted",                                  {"hipErrorProfilerAlreadyStarted",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 57
  // Deprecated since CUDA 5.0
  // CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
  {"cudaErrorProfilerAlreadyStopped",                                  {"hipErrorProfilerAlreadyStopped",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 58
  // CUDA_ERROR_ASSERT = 710
  {"cudaErrorAssert",                                                  {"hipErrorAssert",                                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 59
  // CUDA_ERROR_TOO_MANY_PEERS = 711
  {"cudaErrorTooManyPeers",                                            {"hipErrorTooManyPeers",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 60
  // CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
  {"cudaErrorHostMemoryAlreadyRegistered",                             {"hipErrorHostMemoryAlreadyRegistered",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 61
  // CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
  {"cudaErrorHostMemoryNotRegistered",                                 {"hipErrorHostMemoryNotRegistered",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 62
  // CUDA_ERROR_OPERATING_SYSTEM = 304
  {"cudaErrorOperatingSystem",                                         {"hipErrorOperatingSystem",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 63
  // CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
  {"cudaErrorPeerAccessUnsupported",                                   {"hipErrorPeerAccessUnsupported",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 64
  // no analogue
  {"cudaErrorLaunchMaxDepthExceeded",                                  {"hipErrorLaunchMaxDepthExceeded",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 65
  // no analogue
  {"cudaErrorLaunchFileScopedTex",                                     {"hipErrorLaunchFileScopedTex",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 66
  // no analogue
  {"cudaErrorLaunchFileScopedSurf",                                    {"hipErrorLaunchFileScopedSurf",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 67
  // no analogue
  {"cudaErrorSyncDepthExceeded",                                       {"hipErrorSyncDepthExceeded",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 68
  // no analogue
  {"cudaErrorLaunchPendingCountExceeded",                              {"hipErrorLaunchPendingCountExceeded",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 69
  // CUDA_ERROR_NOT_PERMITTED = 800
  {"cudaErrorNotPermitted",                                            {"hipErrorNotPermitted",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 70
  // CUDA_ERROR_NOT_SUPPORTED = 801
  {"cudaErrorNotSupported",                                            {"hipErrorNotSupported",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 71
  // CUDA_ERROR_HARDWARE_STACK_ERROR = 714
  {"cudaErrorHardwareStackError",                                      {"hipErrorHardwareStackError",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 72
  // CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
  {"cudaErrorIllegalInstruction",                                      {"hipErrorIllegalInstruction",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 73
  // CUDA_ERROR_MISALIGNED_ADDRESS = 716
  {"cudaErrorMisalignedAddress",                                       {"hipErrorMisalignedAddress",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 74
  // CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
  {"cudaErrorInvalidAddressSpace",                                     {"hipErrorInvalidAddressSpace",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 75
  // CUDA_ERROR_INVALID_PC = 718
  {"cudaErrorInvalidPc",                                               {"hipErrorInvalidPc",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 76
  // CUDA_ERROR_ILLEGAL_ADDRESS = 700
  {"cudaErrorIllegalAddress",                                          {"hipErrorIllegalAddress",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 77
  // CUDA_ERROR_INVALID_PTX = 218
  {"cudaErrorInvalidPtx",                                              {"hipErrorInvalidKernelFile",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 78
  // CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
  {"cudaErrorInvalidGraphicsContext",                                  {"hipErrorInvalidGraphicsContext",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 79
  // CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
  {"cudaErrorNvlinkUncorrectable",                                     {"hipErrorNvlinkUncorrectable",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 80
  // no analogue
  {"cudaErrorJitCompilerNotFound",                                     {"hipErrorJitCompilerNotFound",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 81
  // no analogue
  {"cudaErrorCooperativeLaunchTooLarge",                               {"hipErrorCooperativeLaunchTooLarge",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 82
  // CUDA_ERROR_SYSTEM_NOT_READY = 802
  {"cudaErrorSystemNotReady",                                          {"hipErrorSystemNotReady",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 83
  // CUDA_ERROR_ILLEGAL_STATE = 401
  {"cudaErrorIllegalState",                                            {"hipErrorIllegalState",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 84
  // no analogue
  {"cudaErrorStartupFailure",                                          {"hipErrorStartupFailure",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 127
  // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
  {"cudaErrorStreamCaptureUnsupported",                                {"hipErrorStreamCaptureUnsupported",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 900
  // CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
  {"cudaErrorStreamCaptureInvalidated",                                {"hipErrorStreamCaptureInvalidated",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 901
  // CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
  {"cudaErrorStreamCaptureMerge",                                      {"hipErrorStreamCaptureMerge",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 902
  // CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
  {"cudaErrorStreamCaptureUnmatched",                                  {"hipErrorStreamCaptureUnmatched",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 903
  // CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
  {"cudaErrorStreamCaptureUnjoined",                                   {"hipErrorStreamCaptureUnjoined",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 904
  // CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
  {"cudaErrorStreamCaptureIsolation",                                  {"hipErrorStreamCaptureIsolation",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 905
  // CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
  {"cudaErrorStreamCaptureImplicit",                                   {"hipErrorStreamCaptureImplicit",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 906
  // CUDA_ERROR_CAPTURED_EVENT = 907
  {"cudaErrorCapturedEvent",                                           {"hipErrorCapturedEvent",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 907
  // Deprecated since CUDA 4.1
  {"cudaErrorApiFailureBase",                                          {"hipErrorApiFailureBase",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 10000

  // CUexternalMemoryHandleType
  {"cudaExternalMemoryHandleType",                                     {"hipExternalMemoryHandleType",                              "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaExternalMemoryHandleType enum values
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
  {"cudaExternalMemoryHandleTypeOpaqueFd",                             {"hipExternalMemoryHandleTypeOpaqueFD",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
  {"cudaExternalMemoryHandleTypeOpaqueWin32",                          {"hipExternalMemoryHandleTypeOpaqueWin32",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT
  {"cudaExternalMemoryHandleTypeOpaqueWin32Kmt",                       {"hipExternalMemoryHandleTypeOpaqueWin32KMT",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
  {"cudaExternalMemoryHandleTypeD3D12Heap",                            {"hipExternalMemoryHandleTypeD3D12Heap",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 4
  // CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
  {"cudaExternalMemoryHandleTypeD3D12Resource",                        {"hipExternalMemoryHandleTypeD3D12Resource",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 5

  // CUexternalSemaphoreHandleType
  {"cudaExternalSemaphoreHandleType",                                  {"hipExternalSemaphoreHandleType",                           "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaExternalSemaphoreHandleType enum values
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD
  {"cudaExternalSemaphoreHandleTypeOpaqueFd",                          {"hipExternalSemaphoreHandleTypeOpaqueFD",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
  {"cudaExternalSemaphoreHandleTypeOpaqueWin32",                       {"hipExternalSemaphoreHandleTypeOpaqueWin32",                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT
  {"cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt",                    {"hipExternalSemaphoreHandleTypeOpaqueWin32KMT",             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3
  // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
  {"cudaExternalSemaphoreHandleTypeD3D12Fence",                        {"hipExternalSemaphoreHandleTypeD3D12Fence",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 4

  // CUfunction_attribute
  // NOTE: only last, starting from 8, values are presented and are equal to Driver's ones
  {"cudaFuncAttribute",                                                {"hipFuncAttribute",                                         "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaFuncAttribute enum values
  // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
  {"cudaFuncAttributeMaxDynamicSharedMemorySize",                      {"hipFuncAttributeMaxDynamicSharedMemorySize",               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, //  8
  // CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
  {"cudaFuncAttributePreferredSharedMemoryCarveout",                   {"hipFuncAttributePreferredSharedMemoryCarveout",            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, //  9
  // CU_FUNC_ATTRIBUTE_MAX
  {"cudaFuncAttributeMax",                                             {"hipFuncAttributeMax",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 10

  // CUfunc_cache
  {"cudaFuncCache",                                                    {"hipFuncCache_t",                                           "", CONV_TYPE, API_RUNTIME}},
  // cudaFuncCache enum values
  // CU_FUNC_CACHE_PREFER_NONE = 0x00
  {"cudaFuncCachePreferNone",                                          {"hipFuncCachePreferNone",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  // CU_FUNC_CACHE_PREFER_SHARED = 0x01
  {"cudaFuncCachePreferShared",                                        {"hipFuncCachePreferShared",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1
  // CU_FUNC_CACHE_PREFER_L1 = 0x02
  {"cudaFuncCachePreferL1",                                            {"hipFuncCachePreferL1",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 2
  // CU_FUNC_CACHE_PREFER_EQUAL = 0x03
  {"cudaFuncCachePreferEqual",                                         {"hipFuncCachePreferEqual",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 3

  // CUarray_cubemap_face
  {"cudaGraphicsCubeFace",                                             {"hipGraphicsCubeFace",                                      "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaGraphicsCubeFace enum values
  // CU_CUBEMAP_FACE_POSITIVE_X
  {"cudaGraphicsCubeFacePositiveX",                                    {"hipGraphicsCubeFacePositiveX",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x00
  // CU_CUBEMAP_FACE_NEGATIVE_X
  {"cudaGraphicsCubeFaceNegativeX",                                    {"hipGraphicsCubeFaceNegativeX",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x01
  // CU_CUBEMAP_FACE_POSITIVE_Y
  {"cudaGraphicsCubeFacePositiveY",                                    {"hipGraphicsCubeFacePositiveY",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x02
  // CU_CUBEMAP_FACE_NEGATIVE_Y
  {"cudaGraphicsCubeFaceNegativeY",                                    {"hipGraphicsCubeFaceNegativeY",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x03
  // CU_CUBEMAP_FACE_POSITIVE_Z
  {"cudaGraphicsCubeFacePositiveZ",                                    {"hipGraphicsCubeFacePositiveZ",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x04
  // CU_CUBEMAP_FACE_NEGATIVE_Z
  {"cudaGraphicsCubeFaceNegativeZ",                                    {"hipGraphicsCubeFaceNegativeZ",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x05

  // CUgraphicsMapResourceFlags
  {"cudaGraphicsMapFlags",                                             {"hipGraphicsMapFlags",                                      "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaGraphicsMapFlags enum values
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00
  {"cudaGraphicsMapFlagsNone",                                         {"hipGraphicsMapFlagsNone",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01
  {"cudaGraphicsMapFlagsReadOnly",                                     {"hipGraphicsMapFlagsReadOnly",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
  {"cudaGraphicsMapFlagsWriteDiscard",                                 {"hipGraphicsMapFlagsWriteDiscard",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2

  // CUgraphicsRegisterFlags
  {"cudaGraphicsRegisterFlags",                                        {"hipGraphicsRegisterFlags",                                 "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaGraphicsRegisterFlags enum values
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00
  {"cudaGraphicsRegisterFlagsNone",                                    {"hipGraphicsRegisterFlagsNone",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01
  {"cudaGraphicsRegisterFlagsReadOnly",                                {"hipGraphicsRegisterFlagsReadOnly",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02
  {"cudaGraphicsRegisterFlagsWriteDiscard",                            {"hipGraphicsRegisterFlagsWriteDiscard",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x04
  {"cudaGraphicsRegisterFlagsSurfaceLoadStore",                        {"hipGraphicsRegisterFlagsSurfaceLoadStore",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 4
  // CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08
  {"cudaGraphicsRegisterFlagsTextureGather",                           {"hipGraphicsRegisterFlagsTextureGather",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 8

  // CUgraphNodeType
  {"cudaGraphNodeType",                                                {"hipGraphNodeType",                                         "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaGraphNodeType enum values
  // CU_GRAPH_NODE_TYPE_KERNEL = 0
  {"cudaGraphNodeTypeKernel",                                          {"hipGraphNodeTypeKernel",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x00
  // CU_GRAPH_NODE_TYPE_MEMCPY = 1
  {"cudaGraphNodeTypeMemcpy",                                          {"hipGraphNodeTypeMemcpy",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x01
  // CU_GRAPH_NODE_TYPE_MEMSET = 2
  {"cudaGraphNodeTypeMemset",                                          {"hipGraphNodeTypeMemset",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x02
  // CU_GRAPH_NODE_TYPE_HOST = 3
  {"cudaGraphNodeTypeHost",                                            {"hipGraphNodeTypeHost",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x03
  // CU_GRAPH_NODE_TYPE_GRAPH = 4
  {"cudaGraphNodeTypeGraph",                                           {"hipGraphNodeTypeGraph",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x04
  // CU_GRAPH_NODE_TYPE_EMPTY = 5
  {"cudaGraphNodeTypeEmpty",                                           {"hipGraphNodeTypeEmpty",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x05
  // CU_GRAPH_NODE_TYPE_COUNT
  {"cudaGraphNodeTypeCount",                                           {"hipGraphNodeTypeCount",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUlimit
  {"cudaLimit",                                                        {"hipLimit_t",                                               "", CONV_TYPE, API_RUNTIME}},
  // cudaLimit enum values
  // CU_LIMIT_STACK_SIZE
  {"cudaLimitStackSize",                                               {"hipLimitStackSize",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x00
  // CU_LIMIT_PRINTF_FIFO_SIZE
  {"cudaLimitPrintfFifoSize",                                          {"hipLimitPrintfFifoSize",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x01
  // CU_LIMIT_MALLOC_HEAP_SIZE
  {"cudaLimitMallocHeapSize",                                          {"hipLimitMallocHeapSize",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x02
  // CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH
  {"cudaLimitDevRuntimeSyncDepth",                                     {"hipLimitDevRuntimeSyncDepth",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x03
  // CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT
  {"cudaLimitDevRuntimePendingLaunchCount",                            {"hipLimitDevRuntimePendingLaunchCount",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x04
  // CU_LIMIT_MAX_L2_FETCH_GRANULARITY
  {"cudaLimitMaxL2FetchGranularity",                                   {"hipLimitMaxL2FetchGranularity",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x05

  // no analogue
  {"cudaMemcpyKind",                                                   {"hipMemcpyKind",                                            "", CONV_TYPE, API_RUNTIME}},
  // cudaMemcpyKind enum values
  {"cudaMemcpyHostToHost",                                             {"hipMemcpyHostToHost",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  {"cudaMemcpyHostToDevice",                                           {"hipMemcpyHostToDevice",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1
  {"cudaMemcpyDeviceToHost",                                           {"hipMemcpyDeviceToHost",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 2
  {"cudaMemcpyDeviceToDevice",                                         {"hipMemcpyDeviceToDevice",                                  "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 3
  {"cudaMemcpyDefault",                                                {"hipMemcpyDefault",                                         "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 4

  // CUmem_advise
  {"cudaMemoryAdvise",                                                 {"hipMemAdvise",                                             "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaMemoryAdvise enum values
  // CU_MEM_ADVISE_SET_READ_MOSTLY
  {"cudaMemAdviseSetReadMostly",                                       {"hipMemAdviseSetReadMostly",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_MEM_ADVISE_UNSET_READ_MOSTLY
  {"cudaMemAdviseUnsetReadMostly",                                     {"hipMemAdviseUnsetReadMostly",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_MEM_ADVISE_SET_PREFERRED_LOCATION
  {"cudaMemAdviseSetPreferredLocation",                                {"hipMemAdviseSetPreferredLocation",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3
  // CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION
  {"cudaMemAdviseUnsetPreferredLocation",                              {"hipMemAdviseUnsetPreferredLocation",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 4
  // CU_MEM_ADVISE_SET_ACCESSED_BY
  {"cudaMemAdviseSetAccessedBy",                                       {"hipMemAdviseSetAccessedBy",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 5
  // CU_MEM_ADVISE_UNSET_ACCESSED_BY
  {"cudaMemAdviseUnsetAccessedBy",                                     {"hipMemAdviseUnsetAccessedBy",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 6

  // no analogue
  // NOTE: CUmemorytype is partial analogue
  {"cudaMemoryType",                                                   {"hipMemoryType_t",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaMemoryType enum values
  {"cudaMemoryTypeUnregistered",                                       {"hipMemoryTypeUnregistered",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  {"cudaMemoryTypeHost",                                               {"hipMemoryTypeHost",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  {"cudaMemoryTypeDevice",                                             {"hipMemoryTypeDevice",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  {"cudaMemoryTypeManaged",                                            {"hipMemoryTypeManaged",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3

  // CUmem_range_attribute
  {"cudaMemRangeAttribute",                                            {"hipMemRangeAttribute",                                     "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaMemRangeAttribute enum values
  // CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY
  {"cudaMemRangeAttributeReadMostly",                                  {"hipMemRangeAttributeReadMostly",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION
  {"cudaMemRangeAttributePreferredLocation",                           {"hipMemRangeAttributePreferredLocation",                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY
  {"cudaMemRangeAttributeAccessedBy",                                  {"hipMemRangeAttributeAccessedBy",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3
  // CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION
  {"cudaMemRangeAttributeLastPrefetchLocation",                        {"hipMemRangeAttributeLastPrefetchLocation",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 4

  // no analogue
  {"cudaOutputMode",                                                   {"hipOutputMode",                                            "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaOutputMode_t",                                                 {"hipOutputMode",                                            "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaOutputMode enum values
  {"cudaKeyValuePair",                                                 {"hipKeyValuePair",                                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x00
  {"cudaCSV",                                                          {"hipCSV",                                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x01

  // CUresourcetype
  {"cudaResourceType",                                                 {"hipResourceType",                                          "", CONV_TYPE, API_RUNTIME}},
  // cudaResourceType enum values
  // CU_RESOURCE_TYPE_ARRAY
  {"cudaResourceTypeArray",                                            {"hipResourceTypeArray",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x00
  // CU_RESOURCE_TYPE_MIPMAPPED_ARRAY
  {"cudaResourceTypeMipmappedArray",                                   {"hipResourceTypeMipmappedArray",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x01
  // CU_RESOURCE_TYPE_LINEAR
  {"cudaResourceTypeLinear",                                           {"hipResourceTypeLinear",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x02
  // CU_RESOURCE_TYPE_PITCH2D
  {"cudaResourceTypePitch2D",                                          {"hipResourceTypePitch2D",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x03

  // CUresourceViewFormat
  {"cudaResourceViewFormat",                                           {"hipResourceViewFormat",                                    "", CONV_TYPE, API_RUNTIME}},
  // enum cudaResourceViewFormat
  // CU_RES_VIEW_FORMAT_NONE
  {"cudaResViewFormatNone",                                            {"hipResViewFormatNone",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x00
  // CU_RES_VIEW_FORMAT_UINT_1X8
  {"cudaResViewFormatUnsignedChar1",                                   {"hipResViewFormatUnsignedChar1",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x01
  // CU_RES_VIEW_FORMAT_UINT_2X8
  {"cudaResViewFormatUnsignedChar2",                                   {"hipResViewFormatUnsignedChar2",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x02
  // CU_RES_VIEW_FORMAT_UINT_4X8
  {"cudaResViewFormatUnsignedChar4",                                   {"hipResViewFormatUnsignedChar4",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x03
  // CU_RES_VIEW_FORMAT_SINT_1X8
  {"cudaResViewFormatSignedChar1",                                     {"hipResViewFormatSignedChar1",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x04
  // CU_RES_VIEW_FORMAT_SINT_2X8
  {"cudaResViewFormatSignedChar2",                                     {"hipResViewFormatSignedChar2",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x05
  // CU_RES_VIEW_FORMAT_SINT_4X8
  {"cudaResViewFormatSignedChar4",                                     {"hipResViewFormatSignedChar4",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x06
  // CU_RES_VIEW_FORMAT_UINT_1X16
  {"cudaResViewFormatUnsignedShort1",                                  {"hipResViewFormatUnsignedShort1",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x07
  // CU_RES_VIEW_FORMAT_UINT_2X16
  {"cudaResViewFormatUnsignedShort2",                                  {"hipResViewFormatUnsignedShort2",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x08
  // CU_RES_VIEW_FORMAT_UINT_4X16
  {"cudaResViewFormatUnsignedShort4",                                  {"hipResViewFormatUnsignedShort4",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x09
  // CU_RES_VIEW_FORMAT_SINT_1X16
  {"cudaResViewFormatSignedShort1",                                    {"hipResViewFormatSignedShort1",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x0a
  // CU_RES_VIEW_FORMAT_SINT_2X16
  {"cudaResViewFormatSignedShort2",                                    {"hipResViewFormatSignedShort2",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x0b
  // CU_RES_VIEW_FORMAT_SINT_4X16
  {"cudaResViewFormatSignedShort4",                                    {"hipResViewFormatSignedShort4",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x0c
  // CU_RES_VIEW_FORMAT_UINT_1X32
  {"cudaResViewFormatUnsignedInt1",                                    {"hipResViewFormatUnsignedInt1",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x0d
  // CU_RES_VIEW_FORMAT_UINT_2X32
  {"cudaResViewFormatUnsignedInt2",                                    {"hipResViewFormatUnsignedInt2",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x0e
  // CU_RES_VIEW_FORMAT_UINT_4X32
  {"cudaResViewFormatUnsignedInt4",                                    {"hipResViewFormatUnsignedInt4",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x0f
  // CU_RES_VIEW_FORMAT_SINT_1X32
  {"cudaResViewFormatSignedInt1",                                      {"hipResViewFormatSignedInt1",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x10
  // CU_RES_VIEW_FORMAT_SINT_2X32
  {"cudaResViewFormatSignedInt2",                                      {"hipResViewFormatSignedInt2",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x11
  // CU_RES_VIEW_FORMAT_SINT_4X32
  {"cudaResViewFormatSignedInt4",                                      {"hipResViewFormatSignedInt4",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x12
  // CU_RES_VIEW_FORMAT_FLOAT_1X16
  {"cudaResViewFormatHalf1",                                           {"hipResViewFormatHalf1",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x13
  // CU_RES_VIEW_FORMAT_FLOAT_2X16
  {"cudaResViewFormatHalf2",                                           {"hipResViewFormatHalf2",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x14
  // CU_RES_VIEW_FORMAT_FLOAT_4X16
  {"cudaResViewFormatHalf4",                                           {"hipResViewFormatHalf4",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x15
  // CU_RES_VIEW_FORMAT_FLOAT_1X32
  {"cudaResViewFormatFloat1",                                          {"hipResViewFormatFloat1",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x16
  // CU_RES_VIEW_FORMAT_FLOAT_2X32
  {"cudaResViewFormatFloat2",                                          {"hipResViewFormatFloat2",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x17
  // CU_RES_VIEW_FORMAT_FLOAT_4X32
  {"cudaResViewFormatFloat4",                                          {"hipResViewFormatFloat4",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x18
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC1
  {"cudaResViewFormatUnsignedBlockCompressed1",                        {"hipResViewFormatUnsignedBlockCompressed1",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x19
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC2
  {"cudaResViewFormatUnsignedBlockCompressed2",                        {"hipResViewFormatUnsignedBlockCompressed2",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x1a
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC3
  {"cudaResViewFormatUnsignedBlockCompressed3",                        {"hipResViewFormatUnsignedBlockCompressed3",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x1b
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC4
  {"cudaResViewFormatUnsignedBlockCompressed4",                        {"hipResViewFormatUnsignedBlockCompressed4",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x1c
  // CU_RES_VIEW_FORMAT_SIGNED_BC4
  {"cudaResViewFormatSignedBlockCompressed4",                          {"hipResViewFormatSignedBlockCompressed4",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x1d
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC5
  {"cudaResViewFormatUnsignedBlockCompressed5",                        {"hipResViewFormatUnsignedBlockCompressed5",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x1e
  // CU_RES_VIEW_FORMAT_SIGNED_BC5
  {"cudaResViewFormatSignedBlockCompressed5",                          {"hipResViewFormatSignedBlockCompressed5",                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x1f
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC6H
  {"cudaResViewFormatUnsignedBlockCompressed6H",                       {"hipResViewFormatUnsignedBlockCompressed6H",                "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x20
  // CU_RES_VIEW_FORMAT_SIGNED_BC6H
  {"cudaResViewFormatSignedBlockCompressed6H",                         {"hipResViewFormatSignedBlockCompressed6H",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x21
  // CU_RES_VIEW_FORMAT_UNSIGNED_BC7
  {"cudaResViewFormatUnsignedBlockCompressed7",                        {"hipResViewFormatUnsignedBlockCompressed7",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0x22

  // CUshared_carveout
  {"cudaSharedCarveout",                                               {"hipSharedCarveout",                                        "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaSharedCarveout enum values
  // CU_SHAREDMEM_CARVEOUT_DEFAULT
  {"cudaSharedmemCarveoutDefault",                                     {"hipSharedmemCarveoutDefault",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // -1
  // CU_SHAREDMEM_CARVEOUT_MAX_SHARED
  {"cudaSharedmemCarveoutMaxShared",                                   {"hipSharedmemCarveoutMaxShared",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 100
  // CU_SHAREDMEM_CARVEOUT_MAX_L1
  {"cudaSharedmemCarveoutMaxShared",                                   {"hipSharedmemCarveoutMaxL1",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0

  // CUsharedconfig
  {"cudaSharedMemConfig",                                              {"hipSharedMemConfig",                                       "", CONV_TYPE, API_RUNTIME}},
  // cudaSharedMemConfig enum values
  // CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0x00
  {"cudaSharedMemBankSizeDefault",                                     {"hipSharedMemBankSizeDefault",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  // CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 0x01
  {"cudaSharedMemBankSizeFourByte",                                    {"hipSharedMemBankSizeFourByte",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1
  // CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 0x02
  {"cudaSharedMemBankSizeEightByte",                                   {"hipSharedMemBankSizeEightByte",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 2

  // CUstreamCaptureStatus
  {"cudaStreamCaptureStatus",                                          {"hipStreamCaptureStatus",                                   "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaStreamCaptureStatus enum values
  // CU_STREAM_CAPTURE_STATUS_NONE
  {"cudaStreamCaptureStatusNone",                                      {"hipStreamCaptureStatusNone",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_STREAM_CAPTURE_STATUS_ACTIVE
  {"cudaStreamCaptureStatusActive",                                    {"hipStreamCaptureStatusActive",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_STREAM_CAPTURE_STATUS_INVALIDATED
  {"cudaStreamCaptureStatusInvalidated",                               {"hipStreamCaptureStatusInvalidated",                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2

  // no analogue
  {"cudaSurfaceBoundaryMode",                                          {"hipSurfaceBoundaryMode",                                   "", CONV_TYPE, API_RUNTIME}},
  // cudaSurfaceBoundaryMode enum values
  {"cudaBoundaryModeZero",                                             {"hipBoundaryModeZero",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  {"cudaBoundaryModeClamp",                                            {"hipBoundaryModeClamp",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1
  {"cudaBoundaryModeTrap",                                             {"hipBoundaryModeTrap",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 2

  // no analogue
  {"cudaSurfaceFormatMode",                                            {"hipSurfaceFormatMode",                                     "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // enum cudaSurfaceFormatMode
  {"cudaFormatModeForced",                                             {"hipFormatModeForced",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  {"cudaFormatModeAuto",                                               {"hipFormatModeAuto",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1

  // no analogue
  {"cudaTextureAddressMode",                                           {"hipTextureAddressMode",                                    "", CONV_TYPE, API_RUNTIME}},
  // cudaTextureAddressMode enum values
  {"cudaAddressModeWrap",                                              {"hipAddressModeWrap",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  {"cudaAddressModeClamp",                                             {"hipAddressModeClamp",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1
  {"cudaAddressModeMirror",                                            {"hipAddressModeMirror",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 2
  {"cudaAddressModeBorder",                                            {"hipAddressModeBorder",                                     "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 3

  // CUfilter_mode
  {"cudaTextureFilterMode",                                            {"hipTextureFilterMode",                                     "", CONV_TYPE, API_RUNTIME}},
  // cudaTextureFilterMode enum values
  // CU_TR_FILTER_MODE_POINT
  {"cudaFilterModePoint",                                              {"hipFilterModePoint",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  // CU_TR_FILTER_MODE_LINEAR
  {"cudaFilterModeLinear",                                             {"hipFilterModeLinear",                                      "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1

  // no analogue
  {"cudaTextureReadMode",                                              {"hipTextureReadMode",                                       "", CONV_TYPE, API_RUNTIME}},
  // cudaTextureReadMode enum values
  {"cudaReadModeElementType",                                          {"hipReadModeElementType",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  {"cudaReadModeNormalizedFloat",                                      {"hipReadModeNormalizedFloat",                               "", CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1

  // CUGLDeviceList
  {"cudaGLDeviceList",                                                 {"hipGLDeviceList",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaGLDeviceList enum values
  // CU_GL_DEVICE_LIST_ALL = 0x01
  {"cudaGLDeviceListAll",                                              {"hipGLDeviceListAll",                                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_GL_DEVICE_LIST_CURRENT_FRAME = 0x02
  {"cudaGLDeviceListCurrentFrame",                                     {"hipGLDeviceListCurrentFrame",                              "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_GL_DEVICE_LIST_NEXT_FRAME = 0x03
  {"cudaGLDeviceListNextFrame",                                        {"hipGLDeviceListNextFrame",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3

  // CUGLmap_flags
  {"cudaGLMapFlags",                                                   {"hipGLMapFlags",                                            "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaGLMapFlags enum values
  // CU_GL_MAP_RESOURCE_FLAGS_NONE = 0x00
  {"cudaGLMapFlagsNone",                                               {"hipGLMapFlagsNone",                                        "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01
  {"cudaGLMapFlagsReadOnly",                                           {"hipGLMapFlagsReadOnly",                                    "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02
  {"cudaGLMapFlagsWriteDiscard",                                       {"hipGLMapFlagsWriteDiscard",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2

  // CUd3d9DeviceList
  {"cudaD3D9DeviceList",                                               {"hipD3D9DeviceList",                                        "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // CUd3d9DeviceList enum values
  // CU_D3D9_DEVICE_LIST_ALL = 0x01
  {"cudaD3D9DeviceListAll",                                            {"HIP_D3D9_DEVICE_LIST_ALL",                                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_D3D9_DEVICE_LIST_CURRENT_FRAME = 0x02
  {"cudaD3D9DeviceListCurrentFrame",                                   {"HIP_D3D9_DEVICE_LIST_CURRENT_FRAME",                       "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_D3D9_DEVICE_LIST_NEXT_FRAME = 0x03
  {"cudaD3D9DeviceListNextFrame",                                      {"HIP_D3D9_DEVICE_LIST_NEXT_FRAME",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3

  // CUd3d9map_flags
  {"cudaD3D9MapFlags",                                                 {"hipD3D9MapFlags",                                          "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaD3D9MapFlags enum values
  // CU_D3D9_MAPRESOURCE_FLAGS_NONE = 0x00
  {"cudaD3D9MapFlagsNone",                                             {"HIP_D3D9_MAPRESOURCE_FLAGS_NONE",                          "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_D3D9_MAPRESOURCE_FLAGS_READONLY = 0x01
  {"cudaD3D9MapFlagsReadOnly",                                         {"HIP_D3D9_MAPRESOURCE_FLAGS_READONLY",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD = 0x02
  {"cudaD3D9MapFlagsWriteDiscard",                                     {"HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD",                  "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2

  // CUd3d9Register_flags
  {"cudaD3D9RegisterFlags",                                            {"hipD3D9RegisterFlags",                                     "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaD3D9RegisterFlags enum values
  // CU_D3D9_REGISTER_FLAGS_NONE = 0x00
  {"cudaD3D9RegisterFlagsNone",                                        {"HIP_D3D9_REGISTER_FLAGS_NONE",                             "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_D3D9_REGISTER_FLAGS_ARRAY = 0x01
  {"cudaD3D9RegisterFlagsArray",                                       {"HIP_D3D9_REGISTER_FLAGS_ARRAY",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1

  // CUd3d10DeviceList
  {"cudaD3D10DeviceList",                                              {"hipd3d10DeviceList",                                       "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaD3D10DeviceList enum values
  // CU_D3D10_DEVICE_LIST_ALL = 0x01
  {"cudaD3D10DeviceListAll",                                           {"HIP_D3D10_DEVICE_LIST_ALL",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_D3D10_DEVICE_LIST_CURRENT_FRAME = 0x02
  {"cudaD3D10DeviceListCurrentFrame",                                  {"HIP_D3D10_DEVICE_LIST_CURRENT_FRAME",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_D3D10_DEVICE_LIST_NEXT_FRAME = 0x03
  {"cudaD3D10DeviceListNextFrame",                                     {"HIP_D3D10_DEVICE_LIST_NEXT_FRAME",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3

  // CUd3d10map_flags
  {"cudaD3D10MapFlags",                                                {"hipD3D10MapFlags",                                         "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaD3D10MapFlags enum values
  // CU_D3D10_MAPRESOURCE_FLAGS_NONE = 0x00
  {"cudaD3D10MapFlagsNone",                                            {"HIP_D3D10_MAPRESOURCE_FLAGS_NONE",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_D3D10_MAPRESOURCE_FLAGS_READONLY = 0x01
  {"cudaD3D10MapFlagsReadOnly",                                        {"HIP_D3D10_MAPRESOURCE_FLAGS_READONLY",                     "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD = 0x02
  {"cudaD3D10MapFlagsWriteDiscard",                                    {"HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD",                 "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2

  // CUd3d10Register_flags
  {"cudaD3D10RegisterFlags",                                           {"hipD3D10RegisterFlags",                                    "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaD3D10RegisterFlags enum values
  // CU_D3D10_REGISTER_FLAGS_NONE = 0x00
  {"cudaD3D10RegisterFlagsNone",                                       {"HIP_D3D10_REGISTER_FLAGS_NONE",                            "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 0
  // CU_D3D10_REGISTER_FLAGS_ARRAY = 0x01
  {"cudaD3D10RegisterFlagsArray",                                      {"HIP_D3D10_REGISTER_FLAGS_ARRAY",                           "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1

  // CUd3d11DeviceList
  {"cudaD3D11DeviceList",                                              {"hipd3d11DeviceList",                                       "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // cudaD3D11DeviceList enum values
  // CU_D3D11_DEVICE_LIST_ALL = 0x01
  {"cudaD3D11DeviceListAll",                                           {"HIP_D3D11_DEVICE_LIST_ALL",                                "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 1
  // CU_D3D11_DEVICE_LIST_CURRENT_FRAME = 0x02
  {"cudaD3D11DeviceListCurrentFrame",                                  {"HIP_D3D11_DEVICE_LIST_CURRENT_FRAME",                      "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 2
  // CU_D3D11_DEVICE_LIST_NEXT_FRAME = 0x03
  {"cudaD3D11DeviceListNextFrame",                                     {"HIP_D3D11_DEVICE_LIST_NEXT_FRAME",                         "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 3

  // no analogue
  {"libraryPropertyType",                                              {"hipLibraryPropertyType_t",                                 "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"libraryPropertyType_t",                                            {"hipLibraryPropertyType_t",                                 "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"MAJOR_VERSION",                                                    {"hipLibraryMajorVersion",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"MINOR_VERSION",                                                    {"hipLibraryMinorVersion",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}},
  // no analogue
  {"PATCH_LEVEL",                                                      {"hipLibraryPatchVersion",                                   "", CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}},

  // 4. Typedefs

  // CUhostFn
  {"cudaHostFn_t",                                                     {"hipHostFn",                                                "", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUstreamCallback
  {"cudaStreamCallback_t",                                             {"hipStreamCallback_t",                                      "", CONV_TYPE, API_RUNTIME}},

  // CUsurfObject
  {"cudaSurfaceObject_t",                                              {"hipSurfaceObject_t",                                       "", CONV_TYPE, API_RUNTIME}},

  // CUtexObject
  {"cudaTextureObject_t",                                              {"hipTextureObject_t",                                       "", CONV_TYPE, API_RUNTIME}},

  // 5. Defines

  // no analogue
  {"CUDA_EGL_MAX_PLANES",                                              {"HIP_EGL_MAX_PLANES",                                       "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 3
  // CU_IPC_HANDLE_SIZE
  {"CUDA_IPC_HANDLE_SIZE",                                             {"HIP_IPC_HANDLE_SIZE",                                      "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 64
  // no analogue
  {"cudaArrayDefault",                                                 {"hipArrayDefault",                                          "", CONV_DEFINE, API_RUNTIME}}, // 0x00
  // CUDA_ARRAY3D_LAYERED
  {"cudaArrayLayered",                                                 {"hipArrayLayered",                                          "", CONV_DEFINE, API_RUNTIME}}, // 0x01
  // CUDA_ARRAY3D_SURFACE_LDST
  {"cudaArraySurfaceLoadStore",                                        {"hipArraySurfaceLoadStore",                                 "", CONV_DEFINE, API_RUNTIME}}, // 0x02
  // CUDA_ARRAY3D_CUBEMAP
  {"cudaArrayCubemap",                                                 {"hipArrayCubemap",                                          "", CONV_DEFINE, API_RUNTIME}}, // 0x04
  // CUDA_ARRAY3D_TEXTURE_GATHER
  {"cudaArrayTextureGather",                                           {"hipArrayTextureGather",                                    "", CONV_DEFINE, API_RUNTIME}}, // 0x08
  // CUDA_ARRAY3D_COLOR_ATTACHMENT
  {"cudaArrayColorAttachment",                                         {"hipArrayColorAttachment",                                  "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x20
  // CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC
  {"cudaCooperativeLaunchMultiDeviceNoPreSync",                        {"hipCooperativeLaunchMultiDeviceNoPreSync",                 "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x01
  // CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC
  {"cudaCooperativeLaunchMultiDeviceNoPostSync",                       {"hipCooperativeLaunchMultiDeviceNoPostSync",                "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x02
  // CU_DEVICE_CPU ((CUdevice)-1)
  {"cudaCpuDeviceId",                                                  {"hipCpuDeviceId",                                           "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // ((int)-1)
  // CU_DEVICE_INVALID ((CUdevice)-2)
  {"cudaInvalidDeviceId",                                              {"hipInvalidDeviceId",                                       "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // ((int)-2)
  // CU_CTX_BLOCKING_SYNC
  // NOTE: Deprecated since CUDA 4.0 and replaced with cudaDeviceScheduleBlockingSync
  {"cudaDeviceBlockingSync",                                           {"hipDeviceScheduleBlockingSync",                            "", CONV_DEFINE, API_RUNTIME}}, // 0x04
  // CU_CTX_LMEM_RESIZE_TO_MAX
  {"cudaDeviceLmemResizeToMax",                                        {"hipDeviceLmemResizeToMax",                                 "", CONV_DEFINE, API_RUNTIME}}, // 0x10
  // CU_CTX_MAP_HOST
  {"cudaDeviceMapHost",                                                {"hipDeviceMapHost",                                         "", CONV_DEFINE, API_RUNTIME}}, // 0x08
  // CU_CTX_FLAGS_MASK
  {"cudaDeviceMask",                                                   {"hipDeviceMask",                                            "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x1f
  // no analogue
  {"cudaDevicePropDontCare",                                           {"hipDevicePropDontCare",                                    "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}},
  // CU_CTX_SCHED_AUTO
  {"cudaDeviceScheduleAuto",                                           {"hipDeviceScheduleAuto",                                    "", CONV_DEFINE, API_RUNTIME}}, // 0x00
  // CU_CTX_SCHED_SPIN
  {"cudaDeviceScheduleSpin",                                           {"hipDeviceScheduleSpin",                                    "", CONV_DEFINE, API_RUNTIME}}, // 0x01
  // CU_CTX_SCHED_YIELD
  {"cudaDeviceScheduleYield",                                          {"hipDeviceScheduleYield",                                   "", CONV_DEFINE, API_RUNTIME}}, // 0x02
  // CU_CTX_SCHED_BLOCKING_SYNC
  {"cudaDeviceScheduleBlockingSync",                                   {"hipDeviceScheduleBlockingSync",                            "", CONV_DEFINE, API_RUNTIME}}, // 0x04
  // CU_CTX_SCHED_MASK
  {"cudaDeviceScheduleMask",                                           {"hipDeviceScheduleMask",                                    "", CONV_DEFINE, API_RUNTIME}}, // 0x07
  // CU_EVENT_DEFAULT
  {"cudaEventDefault",                                                 {"hipEventDefault",                                          "", CONV_DEFINE, API_RUNTIME}}, // 0x00
  // CU_EVENT_BLOCKING_SYNC
  {"cudaEventBlockingSync",                                            {"hipEventBlockingSync",                                     "", CONV_DEFINE, API_RUNTIME}}, // 0x01
  // CU_EVENT_DISABLE_TIMING
  {"cudaEventDisableTiming",                                           {"hipEventDisableTiming",                                    "", CONV_DEFINE, API_RUNTIME}}, // 0x02
  // CU_EVENT_INTERPROCESS
  {"cudaEventInterprocess",                                            {"hipEventInterprocess",                                     "", CONV_DEFINE, API_RUNTIME}}, // 0x04
  // CUDA_EXTERNAL_MEMORY_DEDICATED
  {"cudaExternalMemoryDedicated",                                      {"hipExternalMemoryDedicated",                               "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x01
  // no analogue
  {"cudaHostAllocDefault",                                             {"hipHostMallocDefault",                                     "", CONV_DEFINE, API_RUNTIME}}, // 0x00
  // CU_MEMHOSTALLOC_PORTABLE
  {"cudaHostAllocPortable",                                            {"hipHostMallocPortable",                                    "", CONV_DEFINE, API_RUNTIME}}, // 0x01
  // CU_MEMHOSTALLOC_DEVICEMAP
  {"cudaHostAllocMapped",                                              {"hipHostMallocMapped",                                      "", CONV_DEFINE, API_RUNTIME}}, // 0x02
  // CU_MEMHOSTALLOC_WRITECOMBINED
  {"cudaHostAllocWriteCombined",                                       {"hipHostAllocWriteCombined",                                "", CONV_DEFINE, API_RUNTIME}}, // 0x04
  // no analogue
  {"cudaHostRegisterDefault",                                          {"hipHostRegisterDefault",                                   "", CONV_DEFINE, API_RUNTIME}}, // 0x00
  // CU_MEMHOSTREGISTER_PORTABLE
  {"cudaHostRegisterPortable",                                         {"hipHostRegisterPortable",                                  "", CONV_DEFINE, API_RUNTIME}}, // 0x01
  // CU_MEMHOSTREGISTER_DEVICEMAP
  {"cudaHostRegisterMapped",                                           {"hipHostRegisterMapped",                                    "", CONV_DEFINE, API_RUNTIME}}, // 0x02
  // CU_MEMHOSTREGISTER_IOMEMORY
  {"cudaHostRegisterIoMemory",                                         {"hipHostRegisterIoMemory",                                  "", CONV_DEFINE, API_RUNTIME}}, // 0x04
  // CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
  {"cudaIpcMemLazyEnablePeerAccess",                                   {"hipIpcMemLazyEnablePeerAccess",                            "", CONV_DEFINE, API_RUNTIME}}, // 0x01
  // CU_MEM_ATTACH_GLOBAL
  {"cudaMemAttachGlobal",                                              {"hipMemAttachGlobal",                                       "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x01
  // CU_MEM_ATTACH_HOST
  {"cudaMemAttachHost",                                                {"hipMemAttachHost",                                         "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x02
  // CU_MEM_ATTACH_SINGLE
  {"cudaMemAttachSingle",                                              {"hipMemAttachSingle",                                       "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x04
  // no analogue
  {"cudaTextureType1D",                                                {"hipTextureType1D",                                         "", CONV_DEFINE, API_RUNTIME}}, // 0x01
  // no analogue
  {"cudaTextureType2D",                                                {"hipTextureType2D",                                         "", CONV_DEFINE, API_RUNTIME}}, // 0x02
  // no analogue
  {"cudaTextureType3D",                                                {"hipTextureType3D",                                         "", CONV_DEFINE, API_RUNTIME}}, // 0x03
  // no analogue
  {"cudaTextureTypeCubemap",                                           {"hipTextureTypeCubemap",                                    "", CONV_DEFINE, API_RUNTIME}}, // 0x0C
  // no analogue
  {"cudaTextureType1DLayered",                                         {"hipTextureType1DLayered",                                  "", CONV_DEFINE, API_RUNTIME}}, // 0xF1
  // no analogue
  {"cudaTextureType2DLayered",                                         {"hipTextureType2DLayered",                                  "", CONV_DEFINE, API_RUNTIME}}, // 0xF2
  // no analogue
  {"cudaTextureTypeCubemapLayered",                                    {"hipTextureTypeCubemapLayered",                             "", CONV_DEFINE, API_RUNTIME}}, // 0xFC
  // CU_OCCUPANCY_DEFAULT
  {"cudaOccupancyDefault",                                             {"hipOccupancyDefault",                                      "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x00
  // CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE
  {"cudaOccupancyDisableCachingOverride",                              {"hipOccupancyDisableCachingOverride",                       "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // 0x01
  // CU_STREAM_DEFAULT
  {"cudaStreamDefault",                                                {"hipStreamDefault",                                         "", CONV_DEFINE, API_RUNTIME}}, // 0x00
  // CU_STREAM_NON_BLOCKING
  {"cudaStreamNonBlocking",                                            {"hipStreamNonBlocking",                                     "", CONV_DEFINE, API_RUNTIME}}, // 0x01
  // CU_STREAM_LEGACY ((CUstream)0x1)
  {"cudaStreamLegacy",                                                 {"hipStreamLegacy",                                          "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // ((cudaStream_t)0x1)
  // CU_STREAM_PER_THREAD ((CUstream)0x2)
  {"cudaStreamPerThread",                                              {"hipStreamPerThread",                                       "", CONV_DEFINE, API_RUNTIME, HIP_UNSUPPORTED}}, // ((cudaStream_t)0x2)
};

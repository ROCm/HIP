#include "CUDA2HIP.h"


// Map of all functions
const std::map<llvm::StringRef, hipCounter> CUDA_DRIVER_FUNCTION_MAP{

  ///////////////////////////// CUDA DRIVER API /////////////////////////////

  // Error Handling
  {"cuGetErrorName",   {"hipGetErrorName___",   CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED}},    // cudaGetErrorName (hipGetErrorName) has different signature
  {"cuGetErrorString", {"hipGetErrorString___", CONV_ERROR, API_DRIVER, HIP_UNSUPPORTED}},    // cudaGetErrorString (hipGetErrorString) has different signature

  // Init
  {"cuInit", {"hipInit", CONV_INIT, API_DRIVER}},

  // Driver
  {"cuDriverGetVersion", {"hipDriverGetVersion", CONV_VERSION, API_DRIVER}},

  // Context Management
  {"cuCtxCreate_v2",              {"hipCtxCreate",                 CONV_CONTEXT, API_DRIVER}},
  {"cuCtxDestroy_v2",             {"hipCtxDestroy",                CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetApiVersion",          {"hipCtxGetApiVersion",          CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetCacheConfig",         {"hipCtxGetCacheConfig",         CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetCurrent",             {"hipCtxGetCurrent",             CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetDevice",              {"hipCtxGetDevice",              CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetFlags",               {"hipCtxGetFlags",               CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetLimit",               {"hipCtxGetLimit",               CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuCtxGetSharedMemConfig",     {"hipCtxGetSharedMemConfig",     CONV_CONTEXT, API_DRIVER}},
  {"cuCtxGetStreamPriorityRange", {"hipCtxGetStreamPriorityRange", CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuCtxPopCurrent_v2",          {"hipCtxPopCurrent",             CONV_CONTEXT, API_DRIVER}},
  {"cuCtxPushCurrent_v2",         {"hipCtxPushCurrent",            CONV_CONTEXT, API_DRIVER}},
  {"cuCtxSetCacheConfig",         {"hipCtxSetCacheConfig",         CONV_CONTEXT, API_DRIVER}},
  {"cuCtxSetCurrent",             {"hipCtxSetCurrent",             CONV_CONTEXT, API_DRIVER}},
  {"cuCtxSetLimit",               {"hipCtxSetLimit",               CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuCtxSetSharedMemConfig",     {"hipCtxSetSharedMemConfig",     CONV_CONTEXT, API_DRIVER}},
  {"cuCtxSynchronize",            {"hipCtxSynchronize",            CONV_CONTEXT, API_DRIVER}},
  // Context Management [DEPRECATED]
  {"cuCtxAttach",                 {"hipCtxAttach",                 CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuCtxDetach",                 {"hipCtxDetach",                 CONV_CONTEXT, API_DRIVER, HIP_UNSUPPORTED}},

  // Primary Context Management
  {"cuDevicePrimaryCtxGetState", {"hipDevicePrimaryCtxGetState", CONV_CONTEXT, API_DRIVER}},
  {"cuDevicePrimaryCtxRelease",  {"hipDevicePrimaryCtxRelease",  CONV_CONTEXT, API_DRIVER}},
  {"cuDevicePrimaryCtxReset",    {"hipDevicePrimaryCtxReset",    CONV_CONTEXT, API_DRIVER}},
  {"cuDevicePrimaryCtxRetain",   {"hipDevicePrimaryCtxRetain",   CONV_CONTEXT, API_DRIVER}},
  {"cuDevicePrimaryCtxSetFlags", {"hipDevicePrimaryCtxSetFlags", CONV_CONTEXT, API_DRIVER}},

  // 1. Device Management
  {"cuDeviceGet",           {"hipGetDevice",           CONV_DEVICE, API_DRIVER}},
  {"cuDeviceGetName",       {"hipDeviceGetName",       CONV_DEVICE, API_DRIVER}},
  {"cuDeviceGetCount",      {"hipGetDeviceCount",      CONV_DEVICE, API_DRIVER}},
  {"cuDeviceGetAttribute",  {"hipDeviceGetAttribute",  CONV_DEVICE, API_DRIVER}},
  {"cuDeviceGetPCIBusId",   {"hipDeviceGetPCIBusId",   CONV_DEVICE, API_DRIVER}},
  {"cuDeviceGetByPCIBusId", {"hipDeviceGetByPCIBusId", CONV_DEVICE, API_DRIVER}},
  {"cuDeviceTotalMem_v2",   {"hipDeviceTotalMem",      CONV_DEVICE, API_DRIVER}},
  {"cuDeviceGetLuid",       {"hipDeviceGetLuid",       CONV_DEVICE, API_DRIVER, HIP_UNSUPPORTED}},

  // 12. Peer Context Memory Access
  {"cuCtxEnablePeerAccess",   {"hipCtxEnablePeerAccess",   CONV_PEER, API_DRIVER}},
  {"cuCtxDisablePeerAccess",  {"hipCtxDisablePeerAccess",  CONV_PEER, API_DRIVER}},
  {"cuDeviceCanAccessPeer",   {"hipDeviceCanAccessPeer",   CONV_PEER, API_DRIVER}},
  {"cuDeviceGetP2PAttribute", {"hipDeviceGetP2PAttribute", CONV_PEER, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaDeviceGetP2PAttribute)

  // Device Management [DEPRECATED]
  {"cuDeviceComputeCapability", {"hipDeviceComputeCapability", CONV_DEVICE, API_DRIVER}},
  {"cuDeviceGetProperties",     {"hipGetDeviceProperties",     CONV_DEVICE, API_DRIVER}},

  // Module Management
  {"cuLinkAddData",         {"hipLinkAddData",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkAddFile",         {"hipLinkAddFile",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkComplete",        {"hipLinkComplete",        CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkCreate",          {"hipLinkCreate",          CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLinkDestroy",         {"hipLinkDestroy",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuModuleGetFunction",   {"hipModuleGetFunction",   CONV_MODULE, API_DRIVER}},
  {"cuModuleGetGlobal_v2",  {"hipModuleGetGlobal",     CONV_MODULE, API_DRIVER}},
  {"cuModuleGetSurfRef",    {"hipModuleGetSurfRef",    CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuModuleGetTexRef",     {"hipModuleGetTexRef",     CONV_MODULE, API_DRIVER}},
  {"cuModuleLoad",          {"hipModuleLoad",          CONV_MODULE, API_DRIVER}},
  {"cuModuleLoadData",      {"hipModuleLoadData",      CONV_MODULE, API_DRIVER}},
  {"cuModuleLoadDataEx",    {"hipModuleLoadDataEx",    CONV_MODULE, API_DRIVER}},
  {"cuModuleLoadFatBinary", {"hipModuleLoadFatBinary", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuModuleUnload",        {"hipModuleUnload",        CONV_MODULE, API_DRIVER}},

  // Event functions
  {"cuEventCreate",           {"hipEventCreate",        CONV_EVENT, API_DRIVER}},
  {"cuEventDestroy_v2",       {"hipEventDestroy",       CONV_EVENT, API_DRIVER}},
  {"cuEventElapsedTime",      {"hipEventElapsedTime",   CONV_EVENT, API_DRIVER}},
  {"cuEventQuery",            {"hipEventQuery",         CONV_EVENT, API_DRIVER}},
  {"cuEventRecord",           {"hipEventRecord",        CONV_EVENT, API_DRIVER}},
  {"cuEventSynchronize",      {"hipEventSynchronize",   CONV_EVENT, API_DRIVER}},

  // External Resource Interoperability
  {"cuSignalExternalSemaphoresAsync",          {"hipSignalExternalSemaphoresAsync",          CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuWaitExternalSemaphoresAsync",            {"hipWaitExternalSemaphoresAsync",            CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuImportExternalMemory",                   {"hipImportExternalMemory",                   CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuExternalMemoryGetMappedBuffer",          {"hipExternalMemoryGetMappedBuffer",          CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuExternalMemoryGetMappedMipmappedArray",  {"hipExternalMemoryGetMappedMipmappedArray",  CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuDestroyExternalMemory",                  {"hipDestroyExternalMemory",                  CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuImportExternalSemaphore",                {"hipImportExternalSemaphore",                CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuDestroyExternalSemaphore",               {"hipDestroyExternalSemaphore",               CONV_EXT_RES, API_DRIVER, HIP_UNSUPPORTED}},

  // Execution Control
  {"cuFuncGetAttribute",       {"hipFuncGetAttribute",       CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuFuncSetCacheConfig",     {"hipFuncSetCacheConfig",     CONV_EXECUTION, API_DRIVER}},
  {"cuFuncSetSharedMemConfig", {"hipFuncSetSharedMemConfig", CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLaunchKernel",           {"hipModuleLaunchKernel",     CONV_EXECUTION, API_DRIVER}},
  {"cuLaunchHostFunc",         {"hipLaunchHostFunc",         CONV_EXECUTION, API_DRIVER, HIP_UNSUPPORTED}},

  // Execution Control [DEPRECATED]
  {"cuFuncSetBlockShape", {"hipFuncSetBlockShape", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuFuncSetSharedSize", {"hipFuncSetSharedSize", CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLaunch",            {"hipLaunch",            CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaLaunch)
  {"cuLaunchGrid",        {"hipLaunchGrid",        CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuLaunchGridAsync",   {"hipLaunchGridAsync",   CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuParamSetf",         {"hipParamSetf",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuParamSeti",         {"hipParamSeti",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuParamSetSize",      {"hipParamSetSize",      CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuParamSetSize",      {"hipParamSetSize",      CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuParamSetv",         {"hipParamSetv",         CONV_MODULE, API_DRIVER, HIP_UNSUPPORTED}},

  // Graph Management
  {"cuGraphCreate",                 {"hipGraphCreate",                 CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphLaunch",                 {"hipGraphLaunch",                 CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphAddKernelNode",          {"hipGraphAddKernelNode",          CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphKernelNodeGetParams",    {"hipGraphKernelNodeGetParams",    CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphKernelNodeSetParams",    {"hipGraphKernelNodeSetParams",    CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphAddMemcpyNode",          {"hipGraphAddMemcpyNode",          CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphMemcpyNodeGetParams",    {"hipGraphMemcpyNodeGetParams",    CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphMemcpyNodeSetParams",    {"hipGraphMemcpyNodeSetParams",    CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphAddMemsetNode",          {"hipGraphAddMemsetNode",          CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphMemsetNodeGetParams",    {"hipGraphMemsetNodeGetParams",    CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphMemsetNodeSetParams",    {"hipGraphMemsetNodeSetParams",    CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphAddHostNode",            {"hipGraphAddHostNode",            CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphHostNodeGetParams",      {"hipGraphHostNodeGetParams",      CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphHostNodeSetParams",      {"hipGraphHostNodeSetParams",      CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphAddChildGraphNode",      {"hipGraphAddChildGraphNode",      CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphChildGraphNodeGetGraph", {"hipGraphChildGraphNodeGetGraph", CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphAddEmptyNode",           {"hipGraphAddEmptyNode",           CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphClone",                  {"hipGraphClone",                  CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphNodeFindInClone",        {"hipGraphNodeFindInClone",        CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphNodeGetType",            {"hipGraphNodeGetType",            CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphGetNodes",               {"hipGraphGetNodes",               CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphGetRootNodes",           {"hipGraphGetRootNodes",           CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphGetEdges",               {"hipGraphGetEdges",               CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphNodeGetDependencies",    {"hipGraphNodeGetDependencies",    CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphNodeGetDependentNodes",  {"hipGraphNodeGetDependentNodes",  CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphAddDependencies",        {"hipGraphAddDependencies",        CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphRemoveDependencies",     {"hipGraphRemoveDependencies",     CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphDestroyNode",            {"hipGraphDestroyNode",            CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphInstantiate",            {"hipGraphInstantiate",            CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphExecDestroy",            {"hipGraphExecDestroy",            CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuGraphDestroy",                {"hipGraphDestroy",                CONV_GRAPH, API_DRIVER, HIP_UNSUPPORTED}},

  // Occupancy
  {"cuOccupancyMaxActiveBlocksPerMultiprocessor",          {"hipOccupancyMaxActiveBlocksPerMultiprocessor",          CONV_OCCUPANCY, API_DRIVER}},    // API_Runtime ANALOGUE (cudaOccupancyMaxActiveBlocksPerMultiprocessor)
  {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", {"hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags", CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
  {"cuOccupancyMaxPotentialBlockSize",                     {"hipOccupancyMaxPotentialBlockSize",                     CONV_OCCUPANCY, API_DRIVER}},    // API_Runtime ANALOGUE (cudaOccupancyMaxPotentialBlockSize)
  {"cuOccupancyMaxPotentialBlockSizeWithFlags",            {"hipOccupancyMaxPotentialBlockSizeWithFlags",            CONV_OCCUPANCY, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaOccupancyMaxPotentialBlockSizeWithFlags)

  // Streams
  {"cuStreamAddCallback",        {"hipStreamAddCallback",        CONV_STREAM, API_DRIVER}},
  {"cuStreamAttachMemAsync",     {"hipStreamAttachMemAsync",     CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuStreamCreate",             {"hipStreamCreate__",           CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaStreamCreate due to different signatures
  {"cuStreamCreateWithPriority", {"hipStreamCreateWithPriority", CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuStreamDestroy_v2",         {"hipStreamDestroy",            CONV_STREAM, API_DRIVER}},
  {"cuStreamGetFlags",           {"hipStreamGetFlags",           CONV_STREAM, API_DRIVER}},
  {"cuStreamGetPriority",        {"hipStreamGetPriority",        CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuStreamQuery",              {"hipStreamQuery",              CONV_STREAM, API_DRIVER}},
  {"cuStreamSynchronize",        {"hipStreamSynchronize",        CONV_STREAM, API_DRIVER}},
  {"cuStreamWaitEvent",          {"hipStreamWaitEvent",          CONV_STREAM, API_DRIVER}},
  {"cuStreamWaitValue32",        {"hipStreamWaitValue32",        CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuStreamWaitValue64",        {"hipStreamWaitValue64",        CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuStreamWriteValue32",       {"hipStreamWriteValue32",       CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuStreamWriteValue64",       {"hipStreamWriteValue64",       CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuStreamBatchMemOp",         {"hipStreamBatchMemOp",         CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuStreamBeginCapture",       {"hipStreamBeginCapture",       CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuStreamEndCapture",         {"hipStreamEndCapture",         CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuStreamIsCapturing",        {"hipStreamIsCapturing",        CONV_STREAM, API_DRIVER, HIP_UNSUPPORTED}},

  // Memory management
  {"cuArray3DCreate",           {"hipArray3DCreate",           CONV_MEMORY, API_DRIVER}},
  {"cuArray3DGetDescriptor",    {"hipArray3DGetDescriptor",    CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuArrayCreate",             {"hipArrayCreate",             CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuArrayDestroy",            {"hipArrayDestroy",            CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuArrayGetDescriptor",      {"hipArrayGetDescriptor",      CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuIpcCloseMemHandle",       {"hipIpcCloseMemHandle",       CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuIpcGetEventHandle",       {"hipIpcGetEventHandle",       CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuIpcGetMemHandle",         {"hipIpcGetMemHandle",         CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuIpcOpenEventHandle",      {"hipIpcOpenEventHandle",      CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuIpcOpenMemHandle",        {"hipIpcOpenMemHandle",        CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemAlloc_v2",             {"hipMalloc",                  CONV_MEMORY, API_DRIVER}},
  {"cuMemAllocHost",            {"hipMemAllocHost",            CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemAllocManaged",         {"hipMemAllocManaged",         CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemAllocPitch",           {"hipMemAllocPitch__",         CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemAllocPitch due to different signatures
  {"cuMemcpy",                  {"hipMemcpy__",                CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy due to different signatures
  {"cuMemcpy2D",                {"hipMemcpy2D__",              CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy2D due to different signatures
  {"cuMemcpy2DAsync",           {"hipMemcpy2DAsync__",         CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy2DAsync due to different signatures
  {"cuMemcpy2DUnaligned",       {"hipMemcpy2DUnaligned",       CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpy3D",                {"hipMemcpy3D__",              CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy3D due to different signatures
  {"cuMemcpy3DAsync",           {"hipMemcpy3DAsync__",         CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy3DAsync due to different signatures
  {"cuMemcpy3DPeer",            {"hipMemcpy3DPeer__",          CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy3DPeer due to different signatures
  {"cuMemcpy3DPeerAsync",       {"hipMemcpy3DPeerAsync__",     CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpy3DPeerAsync due to different signatures
  {"cuMemcpyAsync",             {"hipMemcpyAsync__",           CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpyAsync due to different signatures
  {"cuMemcpyAtoA",              {"hipMemcpyAtoA",              CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoD",              {"hipMemcpyAtoD",              CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoH",              {"hipMemcpyAtoH",              CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyAtoHAsync",         {"hipMemcpyAtoHAsync",         CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyDtoA",              {"hipMemcpyDtoA",              CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyDtoD_v2",           {"hipMemcpyDtoD",              CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyDtoDAsync_v2",      {"hipMemcpyDtoDAsync",         CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyDtoH_v2",           {"hipMemcpyDtoH",              CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyDtoHAsync_v2",      {"hipMemcpyDtoHAsync",         CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyHtoA",              {"hipMemcpyHtoA",              CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyHtoAAsync",         {"hipMemcpyHtoAAsync",         CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemcpyHtoD_v2",           {"hipMemcpyHtoD",              CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyHtoDAsync_v2",      {"hipMemcpyHtoDAsync",         CONV_MEMORY, API_DRIVER}},
  {"cuMemcpyPeerAsync",         {"hipMemcpyPeerAsync__",       CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpyPeerAsync due to different signatures
  {"cuMemcpyPeer",              {"hipMemcpyPeer__",            CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaMemcpyPeer due to different signatures
  {"cuMemFree_v2",              {"hipFree",                    CONV_MEMORY, API_DRIVER}},
  {"cuMemFreeHost",             {"hipHostFree",                CONV_MEMORY, API_DRIVER}},
  {"cuMemGetAddressRange",      {"hipMemGetAddressRange",      CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemGetInfo_v2",           {"hipMemGetInfo",              CONV_MEMORY, API_DRIVER}},
  {"cuMemHostAlloc",            {"hipHostMalloc",              CONV_MEMORY, API_DRIVER}},    // API_Runtime ANALOGUE (cudaHostAlloc)
  {"cuMemHostGetDevicePointer", {"hipMemHostGetDevicePointer", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemHostGetFlags",         {"hipMemHostGetFlags",         CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemHostRegister_v2",      {"hipHostRegister",            CONV_MEMORY, API_DRIVER}},    // API_Runtime ANALOGUE (cudaHostAlloc)
  {"cuMemHostUnregister",       {"hipHostUnregister",          CONV_MEMORY, API_DRIVER}},    // API_Runtime ANALOGUE (cudaHostUnregister)
  {"cuMemsetD16_v2",            {"hipMemsetD16",               CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD16Async",          {"hipMemsetD16Async",          CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD2D16_v2",          {"hipMemsetD2D16",             CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD2D16Async",        {"hipMemsetD2D16Async",        CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD2D32_v2",          {"hipMemsetD2D32",             CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD2D32Async",        {"hipMemsetD2D32Async",        CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD2D8_v2",           {"hipMemsetD2D8",              CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD2D8Async",         {"hipMemsetD2D8Async",         CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD32_v2",            {"hipMemset",                  CONV_MEMORY, API_DRIVER}},    // API_Runtime ANALOGUE (cudaMemset)
  {"cuMemsetD32Async",          {"hipMemsetAsync",             CONV_MEMORY, API_DRIVER}},    // API_Runtime ANALOGUE (cudaMemsetAsync)
  {"cuMemsetD8_v2",             {"hipMemsetD8",                CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMemsetD8Async",           {"hipMemsetD8Async",           CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMipmappedArrayCreate",    {"hipMipmappedArrayCreate",    CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMipmappedArrayDestroy",   {"hipMipmappedArrayDestroy",   CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuMipmappedArrayGetLevel",  {"hipMipmappedArrayGetLevel",  CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},

  // Unified Addressing
  {"cuMemPrefetchAsync",      {"hipMemPrefetchAsync__",    CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // // no API_Runtime ANALOGUE (cudaMemPrefetchAsync has different signature)
  {"cuMemAdvise",             {"hipMemAdvise",             CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // // API_Runtime ANALOGUE (cudaMemAdvise)
  {"cuMemRangeGetAttribute",  {"hipMemRangeGetAttribute",  CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // // API_Runtime ANALOGUE (cudaMemRangeGetAttribute)
  {"cuMemRangeGetAttributes", {"hipMemRangeGetAttributes", CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},    // // API_Runtime ANALOGUE (cudaMemRangeGetAttributes)
  {"cuPointerGetAttribute",   {"hipPointerGetAttribute",   CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuPointerGetAttributes",  {"hipPointerGetAttributes",  CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuPointerSetAttribute",   {"hipPointerSetAttribute",   CONV_MEMORY, API_DRIVER, HIP_UNSUPPORTED}},

  // Texture Reference Mngmnt

  {"cuTexRefGetAddress",          {"hipTexRefGetAddress",          CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetAddressMode",      {"hipTexRefGetAddressMode",      CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetArray",            {"hipTexRefGetArray",            CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetBorderColor",      {"hipTexRefGetBorderColor",      CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},    // // no API_Runtime ANALOGUE
  {"cuTexRefGetFilterMode",       {"hipTexRefGetFilterMode",       CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetFlags",            {"hipTexRefGetFlags",            CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetFormat",           {"hipTexRefGetFormat",           CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMaxAnisotropy",    {"hipTexRefGetMaxAnisotropy",    CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMipmapFilterMode", {"hipTexRefGetMipmapFilterMode", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMipmapLevelBias",  {"hipTexRefGetMipmapLevelBias",  CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMipmapLevelClamp", {"hipTexRefGetMipmapLevelClamp", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefGetMipmappedArray",   {"hipTexRefGetMipmappedArray",   CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetAddress",          {"hipTexRefSetAddress",          CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetAddress2D",        {"hipTexRefSetAddress2D",        CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetAddressMode",      {"hipTexRefSetAddressMode",      CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetArray",            {"hipTexRefSetArray",            CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetBorderColor",      {"hipTexRefSetBorderColor",      CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},    // // no API_Runtime ANALOGUE
  {"cuTexRefSetFilterMode",       {"hipTexRefSetFilterMode",       CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetFlags",            {"hipTexRefSetFlags",            CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetFormat",           {"hipTexRefSetFormat",           CONV_TEXTURE, API_DRIVER}},
  {"cuTexRefSetMaxAnisotropy",    {"hipTexRefSetMaxAnisotropy",    CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetMipmapFilterMode", {"hipTexRefSetMipmapFilterMode", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetMipmapLevelBias",  {"hipTexRefSetMipmapLevelBias",  CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetMipmapLevelClamp", {"hipTexRefSetMipmapLevelClamp", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefSetMipmappedArray",   {"hipTexRefSetMipmappedArray",   CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},

  // Texture Reference Mngmnt [DEPRECATED]
  {"cuTexRefCreate",                 {"hipTexRefCreate",                 CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexRefDestroy",                {"hipTexRefDestroy",                CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},

  // Surface Reference Mngmnt
  {"cuSurfRefGetArray",              {"hipSurfRefGetArray",              CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuSurfRefSetArray",              {"hipSurfRefSetArray",              CONV_SURFACE, API_DRIVER, HIP_UNSUPPORTED}},

  // Texture Object Mngmnt
  {"cuTexObjectCreate",              {"hipTexObjectCreate",              CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexObjectDestroy",             {"hipTexObjectDestroy",             CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexObjectGetResourceDesc",     {"hipTexObjectGetResourceDesc",     CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexObjectGetResourceViewDesc", {"hipTexObjectGetResourceViewDesc", CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuTexObjectGetTextureDesc",      {"hipTexObjectGetTextureDesc",      CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},

  // Surface Object Mngmnt
  {"cuSurfObjectCreate",             {"hipSurfObjectCreate",             CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuSurfObjectDestroy",            {"hipSurfObjectDestroy",            CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},
  {"cuSurfObjectGetResourceDesc",    {"hipSurfObjectGetResourceDesc",    CONV_TEXTURE, API_DRIVER, HIP_UNSUPPORTED}},

  // Graphics Interoperability
  {"cuGraphicsMapResources",                    {"hipGraphicsMapResources",                    CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsMapResources)
  {"cuGraphicsResourceGetMappedMipmappedArray", {"hipGraphicsResourceGetMappedMipmappedArray", CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsResourceGetMappedMipmappedArray)
  {"cuGraphicsResourceGetMappedPointer",        {"hipGraphicsResourceGetMappedPointer",        CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsResourceGetMappedPointer)
  {"cuGraphicsResourceSetMapFlags",             {"hipGraphicsResourceSetMapFlags",             CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsResourceSetMapFlags)
  {"cuGraphicsSubResourceGetMappedArray",       {"hipGraphicsSubResourceGetMappedArray",       CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsSubResourceGetMappedArray)
  {"cuGraphicsUnmapResources",                  {"hipGraphicsUnmapResources",                  CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsUnmapResources)
  {"cuGraphicsUnregisterResource",              {"hipGraphicsUnregisterResource",              CONV_GRAPHICS, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsUnregisterResource)

  // Profiler
  {"cuProfilerInitialize", {"hipProfilerInitialize", CONV_PROFILER, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaProfilerInitialize)
  {"cuProfilerStart",      {"hipProfilerStart",      CONV_PROFILER, API_DRIVER}},    // API_Runtime ANALOGUE (cudaProfilerStart)
  {"cuProfilerStop",       {"hipProfilerStop",       CONV_PROFILER, API_DRIVER}},    // API_Runtime ANALOGUE (cudaProfilerStop)

  {"cuGLGetDevices",                  {"hipGLGetDevices",                  CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLGetDevices)
  {"cuGraphicsGLRegisterBuffer",      {"hipGraphicsGLRegisterBuffer",      CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsGLRegisterBuffer)
  {"cuGraphicsGLRegisterImage",       {"hipGraphicsGLRegisterImage",       CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsGLRegisterImage)
  {"cuWGLGetDevice",                  {"hipWGLGetDevice",                  CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaWGLGetDevice)

  {"cuGLCtxCreate",                          {"hipGLCtxCreate",                          CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuGLInit",                               {"hipGLInit",                               CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuGLMapBufferObject",                    {"hipGLMapBufferObject",                    CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaGLMapBufferObject due to different signatures
  {"cuGLMapBufferObjectAsync",               {"hipGLMapBufferObjectAsync",               CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // Not equal to cudaGLMapBufferObjectAsync due to different signatures
  {"cuGLRegisterBufferObject",               {"hipGLRegisterBufferObject",               CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLRegisterBufferObject)
  {"cuGLSetBufferObjectMapFlags",            {"hipGLSetBufferObjectMapFlags",            CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLSetBufferObjectMapFlags)
  {"cuGLUnmapBufferObject",                  {"hipGLUnmapBufferObject",                  CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLUnmapBufferObject)
  {"cuGLUnmapBufferObjectAsync",             {"hipGLUnmapBufferObjectAsync",             CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLUnmapBufferObjectAsync)
  {"cuGLUnregisterBufferObject",             {"hipGLUnregisterBufferObject",             CONV_OPENGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGLUnregisterBufferObject)

  {"cuD3D9CtxCreate",                   {"hipD3D9CtxCreate",                   CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuD3D9CtxCreateOnDevice",           {"hipD3D9CtxCreateOnDevice",           CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuD3D9GetDevice",                   {"hipD3D9GetDevice",                   CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9GetDevice)
  {"cuD3D9GetDevices",                  {"hipD3D9GetDevices",                  CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9GetDevices)
  {"cuD3D9GetDirect3DDevice",           {"hipD3D9GetDirect3DDevice",           CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9GetDirect3DDevice)
  {"cuGraphicsD3D9RegisterResource",    {"hipGraphicsD3D9RegisterResource",    CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsD3D9RegisterResource)

  {"cuD3D9MapResources",                 {"hipD3D9MapResources",                 CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9MapResources)
  {"cuD3D9RegisterResource",             {"hipD3D9RegisterResource",             CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9RegisterResource)
  {"cuD3D9ResourceGetMappedArray",       {"hipD3D9ResourceGetMappedArray",       CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedArray)
  {"cuD3D9ResourceGetMappedPitch",       {"hipD3D9ResourceGetMappedPitch",       CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedPitch)
  {"cuD3D9ResourceGetMappedPointer",     {"hipD3D9ResourceGetMappedPointer",     CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedPointer)
  {"cuD3D9ResourceGetMappedSize",        {"hipD3D9ResourceGetMappedSize",        CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetMappedSize)
  {"cuD3D9ResourceGetSurfaceDimensions", {"hipD3D9ResourceGetSurfaceDimensions", CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceGetSurfaceDimensions)
  {"cuD3D9ResourceSetMapFlags",          {"hipD3D9ResourceSetMapFlags",          CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9ResourceSetMapFlags)
  {"cuD3D9UnmapResources",               {"hipD3D9UnmapResources",               CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9UnmapResources)
  {"cuD3D9UnregisterResource",           {"hipD3D9UnregisterResource",           CONV_D3D9, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D9UnregisterResource)

  // Direct3D 10 Interoperability
  {"cuD3D10GetDevice",                   {"hipD3D10GetDevice",                   CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10GetDevice)
  {"cuD3D10GetDevices",                  {"hipD3D10GetDevices",                  CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10GetDevices)
  {"cuGraphicsD3D10RegisterResource",    {"hipGraphicsD3D10RegisterResource",    CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsD3D10RegisterResource)

  // Direct3D 10 Interoperability [DEPRECATED]
  {"cuD3D10CtxCreate",                    {"hipD3D10CtxCreate",                    CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuD3D10CtxCreateOnDevice",            {"hipD3D10CtxCreateOnDevice",            CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuD3D10GetDirect3DDevice",            {"hipD3D10GetDirect3DDevice",            CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10GetDirect3DDevice)
  {"cuD3D10MapResources",                 {"hipD3D10MapResources",                 CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10MapResources)
  {"cuD3D10RegisterResource",             {"hipD3D10RegisterResource",             CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10RegisterResource)
  {"cuD3D10ResourceGetMappedArray",       {"hipD3D10ResourceGetMappedArray",       CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedArray)
  {"cuD3D10ResourceGetMappedPitch",       {"hipD3D10ResourceGetMappedPitch",       CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedPitch)
  {"cuD3D10ResourceGetMappedPointer",     {"hipD3D10ResourceGetMappedPointer",     CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedPointer)
  {"cuD3D10ResourceGetMappedSize",        {"hipD3D10ResourceGetMappedSize",        CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetMappedSize)
  {"cuD3D10ResourceGetSurfaceDimensions", {"hipD3D10ResourceGetSurfaceDimensions", CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceGetSurfaceDimensions)
  {"cuD310ResourceSetMapFlags",           {"hipD3D10ResourceSetMapFlags",          CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10ResourceSetMapFlags)
  {"cuD3D10UnmapResources",               {"hipD3D10UnmapResources",               CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10UnmapResources)
  {"cuD3D10UnregisterResource",           {"hipD3D10UnregisterResource",           CONV_D3D10, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D10UnregisterResource)

  // Direct3D 11 Interoperability
  {"cuD3D11GetDevice",                   {"hipD3D11GetDevice",                   CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D11GetDevice)
  {"cuD3D11GetDevices",                  {"hipD3D11GetDevices",                  CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D11GetDevices)
  {"cuGraphicsD3D11RegisterResource",    {"hipGraphicsD3D11RegisterResource",    CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsD3D11RegisterResource)

  // Direct3D 11 Interoperability [DEPRECATED]
  {"cuD3D11CtxCreate",         {"hipD3D11CtxCreate",         CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuD3D11CtxCreateOnDevice", {"hipD3D11CtxCreateOnDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuD3D11GetDirect3DDevice", {"hipD3D11GetDirect3DDevice", CONV_D3D11, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaD3D11GetDirect3DDevice)

  // VDPAU Interoperability
  {"cuGraphicsVDPAURegisterOutputSurface", {"hipGraphicsVDPAURegisterOutputSurface", CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsVDPAURegisterOutputSurface)
  {"cuGraphicsVDPAURegisterVideoSurface",  {"hipGraphicsVDPAURegisterVideoSurface",  CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsVDPAURegisterVideoSurface)
  {"cuVDPAUGetDevice",                     {"hipVDPAUGetDevice",                     CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaVDPAUGetDevice)
  {"cuVDPAUCtxCreate",                     {"hipVDPAUCtxCreate",                     CONV_VDPAU, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE

  // EGL Interoperability
  {"cuEGLStreamConsumerAcquireFrame",     {"hipEGLStreamConsumerAcquireFrame",     CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamConsumerAcquireFrame)
  {"cuEGLStreamConsumerConnect",          {"hipEGLStreamConsumerConnect",          CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamConsumerConnect)
  {"cuEGLStreamConsumerConnectWithFlags", {"hipEGLStreamConsumerConnectWithFlags", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamConsumerConnectWithFlags)
  {"cuEGLStreamConsumerDisconnect",       {"hipEGLStreamConsumerDisconnect",       CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // no API_Runtime ANALOGUE
  {"cuEGLStreamConsumerReleaseFrame",     {"hipEGLStreamConsumerReleaseFrame",     CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamConsumerReleaseFrame)
  {"cuEGLStreamProducerConnect",          {"hipEGLStreamProducerConnect",          CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamProducerConnect)
  {"cuEGLStreamProducerDisconnect",       {"hipEGLStreamProducerDisconnect",       CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamProducerDisconnect)
  {"cuEGLStreamProducerPresentFrame",     {"hipEGLStreamProducerPresentFrame",     CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamProducerPresentFrame)
  {"cuEGLStreamProducerReturnFrame",      {"hipEGLStreamProducerReturnFrame",      CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaEGLStreamProducerReturnFrame)
  {"cuGraphicsEGLRegisterImage",          {"hipGraphicsEGLRegisterImage",          CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsEGLRegisterImage)
  {"cuGraphicsResourceGetMappedEglFrame", {"hipGraphicsResourceGetMappedEglFrame", CONV_EGL, API_DRIVER, HIP_UNSUPPORTED}},    // API_Runtime ANALOGUE (cudaGraphicsResourceGetMappedEglFrame)


////////////////////////////// cuComplex API //////////////////////////////
  {"cuFloatComplex",                               {"hipFloatComplex",                                     CONV_TYPE, API_COMPLEX}},
  {"cuDoubleComplex",                              {"hipDoubleComplex",                                    CONV_TYPE, API_COMPLEX}},
  {"cuComplex",                                    {"hipComplex",                                          CONV_TYPE, API_COMPLEX}},

  {"cuCrealf",                                     {"hipCrealf",                                           CONV_COMPLEX, API_COMPLEX}},
  {"cuCimagf",                                     {"hipCimagf",                                           CONV_COMPLEX, API_COMPLEX}},
  {"make_cuFloatComplex",                          {"make_hipFloatComplex",                                CONV_COMPLEX, API_COMPLEX}},
  {"cuConjf",                                      {"hipConjf",                                            CONV_COMPLEX, API_COMPLEX}},
  {"cuCaddf",                                      {"hipCaddf",                                            CONV_COMPLEX, API_COMPLEX}},
  {"cuCsubf",                                      {"hipCsubf",                                            CONV_COMPLEX, API_COMPLEX}},
  {"cuCmulf",                                      {"hipCmulf",                                            CONV_COMPLEX, API_COMPLEX}},
  {"cuCdivf",                                      {"hipCdivf",                                            CONV_COMPLEX, API_COMPLEX}},
  {"cuCabsf",                                      {"hipCabsf",                                            CONV_COMPLEX, API_COMPLEX}},
  {"cuCreal",                                      {"hipCreal",                                            CONV_COMPLEX, API_COMPLEX}},
  {"cuCimag",                                      {"hipCimag",                                            CONV_COMPLEX, API_COMPLEX}},
  {"make_cuDoubleComplex",                         {"make_hipDoubleComplex",                               CONV_COMPLEX, API_COMPLEX}},
  {"cuConj",                                       {"hipConj",                                             CONV_COMPLEX, API_COMPLEX}},
  {"cuCadd",                                       {"hipCadd",                                             CONV_COMPLEX, API_COMPLEX}},
  {"cuCsub",                                       {"hipCsub",                                             CONV_COMPLEX, API_COMPLEX}},
  {"cuCmul",                                       {"hipCmul",                                             CONV_COMPLEX, API_COMPLEX}},
  {"cuCdiv",                                       {"hipCdiv",                                             CONV_COMPLEX, API_COMPLEX}},
  {"cuCabs",                                       {"hipCabs",                                             CONV_COMPLEX, API_COMPLEX}},
  {"make_cuComplex",                               {"make_hipComplex",                                     CONV_COMPLEX, API_COMPLEX}},
  {"cuComplexFloatToDouble",                       {"hipComplexFloatToDouble",                             CONV_COMPLEX, API_COMPLEX}},
  {"cuComplexDoubleToFloat",                       {"hipComplexDoubleToFloat",                             CONV_COMPLEX, API_COMPLEX}},
  {"cuCfmaf",                                      {"hipCfmaf",                                            CONV_COMPLEX, API_COMPLEX}},
  {"cuCfma",                                       {"hipCfma",                                             CONV_COMPLEX, API_COMPLEX}},
};
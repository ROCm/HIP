// automatically generated sources
#ifndef _HIP_PROF_STR_H
#define _HIP_PROF_STR_H

// Dummy API primitives
#define INIT_NONE_CB_ARGS_DATA(cb_data) {};
#define INIT_hipModuleGetFunctionEx_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDestroySurfaceObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDestroyTextureObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddress_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTexture2D_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipCreateTextureObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTextureToMipmappedArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTextureToArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFormat_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyHtoH_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureReference_CB_ARGS_DATA(cb_data) {};
#define INIT_hipHccGetAcceleratorView_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddress2D_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectResourceViewDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipUnbindTexture_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFilterMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureAlignmentOffset_CB_ARGS_DATA(cb_data) {};
#define INIT_hipCreateSurfaceObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetChannelDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetAddressMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipExtModuleLaunchKernel_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectResourceDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddressMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFlags_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTexture_CB_ARGS_DATA(cb_data) {};
#define INIT_hipHccGetAccelerator_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefGetAddress_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectTextureDesc_CB_ARGS_DATA(cb_data) {};

// HIP API callbacks ID enumaration
enum hip_api_id_t {
  HIP_API_ID_hipStreamCreateWithPriority = 0,
  HIP_API_ID_hipMemcpyToSymbolAsync = 1,
  HIP_API_ID_hipMallocPitch = 2,
  HIP_API_ID_hipMalloc = 3,
  HIP_API_ID_hipMemsetD16 = 4,
  HIP_API_ID_hipDeviceGetName = 5,
  HIP_API_ID_hipEventRecord = 6,
  HIP_API_ID_hipCtxSynchronize = 7,
  HIP_API_ID_hipSetDevice = 8,
  HIP_API_ID_hipCtxGetApiVersion = 9,
  HIP_API_ID_hipSetupArgument = 10,
  HIP_API_ID_hipMemcpyFromSymbolAsync = 11,
  HIP_API_ID_hipExtGetLinkTypeAndHopCount = 12,
  HIP_API_ID_hipMemcpyDtoD = 13,
  HIP_API_ID_hipHostFree = 14,
  HIP_API_ID_hipMemcpy2DToArray = 15,
  HIP_API_ID_hipMemsetD8Async = 16,
  HIP_API_ID_hipCtxGetCacheConfig = 17,
  HIP_API_ID_hipStreamWaitEvent = 18,
  HIP_API_ID_hipDeviceGetStreamPriorityRange = 19,
  HIP_API_ID_hipModuleLoad = 20,
  HIP_API_ID_hipDevicePrimaryCtxSetFlags = 21,
  HIP_API_ID_hipLaunchCooperativeKernel = 22,
  HIP_API_ID_hipLaunchCooperativeKernelMultiDevice = 23,
  HIP_API_ID_hipMemcpyAsync = 24,
  HIP_API_ID_hipMalloc3DArray = 25,
  HIP_API_ID_hipStreamCreate = 26,
  HIP_API_ID_hipCtxGetCurrent = 27,
  HIP_API_ID_hipDevicePrimaryCtxGetState = 28,
  HIP_API_ID_hipEventQuery = 29,
  HIP_API_ID_hipEventCreate = 30,
  HIP_API_ID_hipMemGetAddressRange = 31,
  HIP_API_ID_hipMemcpyFromSymbol = 32,
  HIP_API_ID_hipArrayCreate = 33,
  HIP_API_ID_hipStreamGetFlags = 34,
  HIP_API_ID_hipMallocArray = 35,
  HIP_API_ID_hipCtxGetSharedMemConfig = 36,
  HIP_API_ID_hipMemPtrGetInfo = 37,
  HIP_API_ID_hipFuncGetAttribute = 38,
  HIP_API_ID_hipCtxGetFlags = 39,
  HIP_API_ID_hipStreamDestroy = 40,
  HIP_API_ID___hipPushCallConfiguration = 41,
  HIP_API_ID_hipMemset3DAsync = 42,
  HIP_API_ID_hipMemcpy3D = 43,
  HIP_API_ID_hipInit = 44,
  HIP_API_ID_hipMemcpyAtoH = 45,
  HIP_API_ID_hipStreamGetPriority = 46,
  HIP_API_ID_hipMemset2D = 47,
  HIP_API_ID_hipMemset2DAsync = 48,
  HIP_API_ID_hipDeviceCanAccessPeer = 49,
  HIP_API_ID_hipDeviceEnablePeerAccess = 50,
  HIP_API_ID_hipLaunchKernel = 51,
  HIP_API_ID_hipMemsetD16Async = 52,
  HIP_API_ID_hipModuleUnload = 53,
  HIP_API_ID_hipHostUnregister = 54,
  HIP_API_ID_hipProfilerStop = 55,
  HIP_API_ID_hipLaunchByPtr = 56,
  HIP_API_ID_hipStreamSynchronize = 57,
  HIP_API_ID_hipDeviceSetCacheConfig = 58,
  HIP_API_ID_hipGetErrorName = 59,
  HIP_API_ID_hipMemcpyHtoD = 60,
  HIP_API_ID_hipModuleGetGlobal = 61,
  HIP_API_ID_hipMemcpyHtoA = 62,
  HIP_API_ID_hipCtxCreate = 63,
  HIP_API_ID_hipMemcpy2D = 64,
  HIP_API_ID_hipIpcCloseMemHandle = 65,
  HIP_API_ID_hipChooseDevice = 66,
  HIP_API_ID_hipDeviceSetSharedMemConfig = 67,
  HIP_API_ID_hipDeviceComputeCapability = 68,
  HIP_API_ID_hipDeviceGet = 69,
  HIP_API_ID_hipProfilerStart = 70,
  HIP_API_ID_hipCtxSetCacheConfig = 71,
  HIP_API_ID_hipFuncSetCacheConfig = 72,
  HIP_API_ID_hipModuleGetTexRef = 73,
  HIP_API_ID_hipMemcpyPeerAsync = 74,
  HIP_API_ID_hipMemcpyWithStream = 75,
  HIP_API_ID_hipDevicePrimaryCtxReset = 76,
  HIP_API_ID_hipMemcpy3DAsync = 77,
  HIP_API_ID_hipEventDestroy = 78,
  HIP_API_ID_hipCtxPopCurrent = 79,
  HIP_API_ID_hipGetSymbolAddress = 80,
  HIP_API_ID_hipHostGetFlags = 81,
  HIP_API_ID_hipHostMalloc = 82,
  HIP_API_ID_hipDriverGetVersion = 83,
  HIP_API_ID_hipMemGetInfo = 84,
  HIP_API_ID_hipDeviceReset = 85,
  HIP_API_ID_hipMemset = 86,
  HIP_API_ID_hipMemsetD8 = 87,
  HIP_API_ID_hipMemcpyParam2DAsync = 88,
  HIP_API_ID_hipHostRegister = 89,
  HIP_API_ID_hipCtxSetSharedMemConfig = 90,
  HIP_API_ID_hipArray3DCreate = 91,
  HIP_API_ID_hipIpcOpenMemHandle = 92,
  HIP_API_ID_hipGetLastError = 93,
  HIP_API_ID_hipCtxDestroy = 94,
  HIP_API_ID_hipDeviceGetSharedMemConfig = 95,
  HIP_API_ID_hipMemcpy2DFromArray = 96,
  HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = 97,
  HIP_API_ID_hipSetDeviceFlags = 98,
  HIP_API_ID_hipHccModuleLaunchKernel = 99,
  HIP_API_ID_hipFree = 100,
  HIP_API_ID_hipOccupancyMaxPotentialBlockSize = 101,
  HIP_API_ID_hipDeviceGetAttribute = 102,
  HIP_API_ID_hipMemcpyDtoH = 103,
  HIP_API_ID_hipCtxDisablePeerAccess = 104,
  HIP_API_ID_hipMallocManaged = 105,
  HIP_API_ID_hipDeviceGetByPCIBusId = 106,
  HIP_API_ID_hipIpcGetMemHandle = 107,
  HIP_API_ID_hipMemcpyHtoDAsync = 108,
  HIP_API_ID_hipCtxGetDevice = 109,
  HIP_API_ID_hipMemset3D = 110,
  HIP_API_ID_hipModuleLoadData = 111,
  HIP_API_ID_hipDeviceTotalMem = 112,
  HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor = 113,
  HIP_API_ID_hipCtxSetCurrent = 114,
  HIP_API_ID_hipGetErrorString = 115,
  HIP_API_ID_hipDevicePrimaryCtxRetain = 116,
  HIP_API_ID_hipDeviceDisablePeerAccess = 117,
  HIP_API_ID_hipStreamCreateWithFlags = 118,
  HIP_API_ID_hipMemcpyFromArray = 119,
  HIP_API_ID_hipMemcpy2DAsync = 120,
  HIP_API_ID_hipFuncGetAttributes = 121,
  HIP_API_ID_hipGetSymbolSize = 122,
  HIP_API_ID_hipEventCreateWithFlags = 123,
  HIP_API_ID_hipStreamQuery = 124,
  HIP_API_ID_hipDeviceGetPCIBusId = 125,
  HIP_API_ID_hipMemcpy = 126,
  HIP_API_ID_hipPeekAtLastError = 127,
  HIP_API_ID_hipExtLaunchMultiKernelMultiDevice = 128,
  HIP_API_ID_hipStreamAddCallback = 129,
  HIP_API_ID_hipMemcpyToArray = 130,
  HIP_API_ID_hipMemsetD32 = 131,
  HIP_API_ID_hipDeviceSynchronize = 132,
  HIP_API_ID_hipDeviceGetCacheConfig = 133,
  HIP_API_ID_hipMalloc3D = 134,
  HIP_API_ID_hipPointerGetAttributes = 135,
  HIP_API_ID_hipMemsetAsync = 136,
  HIP_API_ID_hipMemcpyToSymbol = 137,
  HIP_API_ID_hipCtxPushCurrent = 138,
  HIP_API_ID_hipMemcpyPeer = 139,
  HIP_API_ID_hipEventSynchronize = 140,
  HIP_API_ID_hipMemcpyDtoDAsync = 141,
  HIP_API_ID_hipExtMallocWithFlags = 142,
  HIP_API_ID_hipCtxEnablePeerAccess = 143,
  HIP_API_ID_hipMemcpyDtoHAsync = 144,
  HIP_API_ID_hipModuleLaunchKernel = 145,
  HIP_API_ID_hipMemAllocPitch = 146,
  HIP_API_ID_hipMemcpy2DFromArrayAsync = 147,
  HIP_API_ID_hipDeviceGetLimit = 148,
  HIP_API_ID_hipModuleLoadDataEx = 149,
  HIP_API_ID_hipRuntimeGetVersion = 150,
  HIP_API_ID___hipPopCallConfiguration = 151,
  HIP_API_ID_hipGetDeviceProperties = 152,
  HIP_API_ID_hipFreeArray = 153,
  HIP_API_ID_hipEventElapsedTime = 154,
  HIP_API_ID_hipDevicePrimaryCtxRelease = 155,
  HIP_API_ID_hipHostGetDevicePointer = 156,
  HIP_API_ID_hipMemcpyParam2D = 157,
  HIP_API_ID_hipModuleGetFunction = 158,
  HIP_API_ID_hipMemsetD32Async = 159,
  HIP_API_ID_hipGetDevice = 160,
  HIP_API_ID_hipGetDeviceCount = 161,
  HIP_API_ID_NUMBER = 162,

  HIP_API_ID_NONE = HIP_API_ID_NUMBER,
  HIP_API_ID_hipModuleGetFunctionEx = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDestroySurfaceObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDestroyTextureObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddress = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTexture2D = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipCreateTextureObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTextureToMipmappedArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTextureToArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFormat = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyHtoH = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureReference = HIP_API_ID_NUMBER,
  HIP_API_ID_hipHccGetAcceleratorView = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddress2D = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectResourceViewDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipUnbindTexture = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFilterMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureAlignmentOffset = HIP_API_ID_NUMBER,
  HIP_API_ID_hipCreateSurfaceObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetChannelDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetAddressMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipExtModuleLaunchKernel = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectResourceDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddressMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFlags = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTexture = HIP_API_ID_NUMBER,
  HIP_API_ID_hipHccGetAccelerator = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefGetAddress = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectTextureDesc = HIP_API_ID_NUMBER,
};

// Return HIP API string
static inline const char* hip_api_name(const uint32_t id) {
  switch(id) {
    case HIP_API_ID_hipStreamCreateWithPriority: return "hipStreamCreateWithPriority";
    case HIP_API_ID_hipMemcpyToSymbolAsync: return "hipMemcpyToSymbolAsync";
    case HIP_API_ID_hipMallocPitch: return "hipMallocPitch";
    case HIP_API_ID_hipMalloc: return "hipMalloc";
    case HIP_API_ID_hipMemsetD16: return "hipMemsetD16";
    case HIP_API_ID_hipDeviceGetName: return "hipDeviceGetName";
    case HIP_API_ID_hipEventRecord: return "hipEventRecord";
    case HIP_API_ID_hipCtxSynchronize: return "hipCtxSynchronize";
    case HIP_API_ID_hipSetDevice: return "hipSetDevice";
    case HIP_API_ID_hipCtxGetApiVersion: return "hipCtxGetApiVersion";
    case HIP_API_ID_hipSetupArgument: return "hipSetupArgument";
    case HIP_API_ID_hipMemcpyFromSymbolAsync: return "hipMemcpyFromSymbolAsync";
    case HIP_API_ID_hipExtGetLinkTypeAndHopCount: return "hipExtGetLinkTypeAndHopCount";
    case HIP_API_ID_hipMemcpyDtoD: return "hipMemcpyDtoD";
    case HIP_API_ID_hipHostFree: return "hipHostFree";
    case HIP_API_ID_hipMemcpy2DToArray: return "hipMemcpy2DToArray";
    case HIP_API_ID_hipMemsetD8Async: return "hipMemsetD8Async";
    case HIP_API_ID_hipCtxGetCacheConfig: return "hipCtxGetCacheConfig";
    case HIP_API_ID_hipStreamWaitEvent: return "hipStreamWaitEvent";
    case HIP_API_ID_hipDeviceGetStreamPriorityRange: return "hipDeviceGetStreamPriorityRange";
    case HIP_API_ID_hipModuleLoad: return "hipModuleLoad";
    case HIP_API_ID_hipDevicePrimaryCtxSetFlags: return "hipDevicePrimaryCtxSetFlags";
    case HIP_API_ID_hipLaunchCooperativeKernel: return "hipLaunchCooperativeKernel";
    case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice: return "hipLaunchCooperativeKernelMultiDevice";
    case HIP_API_ID_hipMemcpyAsync: return "hipMemcpyAsync";
    case HIP_API_ID_hipMalloc3DArray: return "hipMalloc3DArray";
    case HIP_API_ID_hipStreamCreate: return "hipStreamCreate";
    case HIP_API_ID_hipCtxGetCurrent: return "hipCtxGetCurrent";
    case HIP_API_ID_hipDevicePrimaryCtxGetState: return "hipDevicePrimaryCtxGetState";
    case HIP_API_ID_hipEventQuery: return "hipEventQuery";
    case HIP_API_ID_hipEventCreate: return "hipEventCreate";
    case HIP_API_ID_hipMemGetAddressRange: return "hipMemGetAddressRange";
    case HIP_API_ID_hipMemcpyFromSymbol: return "hipMemcpyFromSymbol";
    case HIP_API_ID_hipArrayCreate: return "hipArrayCreate";
    case HIP_API_ID_hipStreamGetFlags: return "hipStreamGetFlags";
    case HIP_API_ID_hipMallocArray: return "hipMallocArray";
    case HIP_API_ID_hipCtxGetSharedMemConfig: return "hipCtxGetSharedMemConfig";
    case HIP_API_ID_hipMemPtrGetInfo: return "hipMemPtrGetInfo";
    case HIP_API_ID_hipFuncGetAttribute: return "hipFuncGetAttribute";
    case HIP_API_ID_hipCtxGetFlags: return "hipCtxGetFlags";
    case HIP_API_ID_hipStreamDestroy: return "hipStreamDestroy";
    case HIP_API_ID___hipPushCallConfiguration: return "__hipPushCallConfiguration";
    case HIP_API_ID_hipMemset3DAsync: return "hipMemset3DAsync";
    case HIP_API_ID_hipMemcpy3D: return "hipMemcpy3D";
    case HIP_API_ID_hipInit: return "hipInit";
    case HIP_API_ID_hipMemcpyAtoH: return "hipMemcpyAtoH";
    case HIP_API_ID_hipStreamGetPriority: return "hipStreamGetPriority";
    case HIP_API_ID_hipMemset2D: return "hipMemset2D";
    case HIP_API_ID_hipMemset2DAsync: return "hipMemset2DAsync";
    case HIP_API_ID_hipDeviceCanAccessPeer: return "hipDeviceCanAccessPeer";
    case HIP_API_ID_hipDeviceEnablePeerAccess: return "hipDeviceEnablePeerAccess";
    case HIP_API_ID_hipLaunchKernel: return "hipLaunchKernel";
    case HIP_API_ID_hipMemsetD16Async: return "hipMemsetD16Async";
    case HIP_API_ID_hipModuleUnload: return "hipModuleUnload";
    case HIP_API_ID_hipHostUnregister: return "hipHostUnregister";
    case HIP_API_ID_hipProfilerStop: return "hipProfilerStop";
    case HIP_API_ID_hipLaunchByPtr: return "hipLaunchByPtr";
    case HIP_API_ID_hipStreamSynchronize: return "hipStreamSynchronize";
    case HIP_API_ID_hipDeviceSetCacheConfig: return "hipDeviceSetCacheConfig";
    case HIP_API_ID_hipGetErrorName: return "hipGetErrorName";
    case HIP_API_ID_hipMemcpyHtoD: return "hipMemcpyHtoD";
    case HIP_API_ID_hipModuleGetGlobal: return "hipModuleGetGlobal";
    case HIP_API_ID_hipMemcpyHtoA: return "hipMemcpyHtoA";
    case HIP_API_ID_hipCtxCreate: return "hipCtxCreate";
    case HIP_API_ID_hipMemcpy2D: return "hipMemcpy2D";
    case HIP_API_ID_hipIpcCloseMemHandle: return "hipIpcCloseMemHandle";
    case HIP_API_ID_hipChooseDevice: return "hipChooseDevice";
    case HIP_API_ID_hipDeviceSetSharedMemConfig: return "hipDeviceSetSharedMemConfig";
    case HIP_API_ID_hipDeviceComputeCapability: return "hipDeviceComputeCapability";
    case HIP_API_ID_hipDeviceGet: return "hipDeviceGet";
    case HIP_API_ID_hipProfilerStart: return "hipProfilerStart";
    case HIP_API_ID_hipCtxSetCacheConfig: return "hipCtxSetCacheConfig";
    case HIP_API_ID_hipFuncSetCacheConfig: return "hipFuncSetCacheConfig";
    case HIP_API_ID_hipModuleGetTexRef: return "hipModuleGetTexRef";
    case HIP_API_ID_hipMemcpyPeerAsync: return "hipMemcpyPeerAsync";
    case HIP_API_ID_hipMemcpyWithStream: return "hipMemcpyWithStream";
    case HIP_API_ID_hipDevicePrimaryCtxReset: return "hipDevicePrimaryCtxReset";
    case HIP_API_ID_hipMemcpy3DAsync: return "hipMemcpy3DAsync";
    case HIP_API_ID_hipEventDestroy: return "hipEventDestroy";
    case HIP_API_ID_hipCtxPopCurrent: return "hipCtxPopCurrent";
    case HIP_API_ID_hipGetSymbolAddress: return "hipGetSymbolAddress";
    case HIP_API_ID_hipHostGetFlags: return "hipHostGetFlags";
    case HIP_API_ID_hipHostMalloc: return "hipHostMalloc";
    case HIP_API_ID_hipDriverGetVersion: return "hipDriverGetVersion";
    case HIP_API_ID_hipMemGetInfo: return "hipMemGetInfo";
    case HIP_API_ID_hipDeviceReset: return "hipDeviceReset";
    case HIP_API_ID_hipMemset: return "hipMemset";
    case HIP_API_ID_hipMemsetD8: return "hipMemsetD8";
    case HIP_API_ID_hipMemcpyParam2DAsync: return "hipMemcpyParam2DAsync";
    case HIP_API_ID_hipHostRegister: return "hipHostRegister";
    case HIP_API_ID_hipCtxSetSharedMemConfig: return "hipCtxSetSharedMemConfig";
    case HIP_API_ID_hipArray3DCreate: return "hipArray3DCreate";
    case HIP_API_ID_hipIpcOpenMemHandle: return "hipIpcOpenMemHandle";
    case HIP_API_ID_hipGetLastError: return "hipGetLastError";
    case HIP_API_ID_hipCtxDestroy: return "hipCtxDestroy";
    case HIP_API_ID_hipDeviceGetSharedMemConfig: return "hipDeviceGetSharedMemConfig";
    case HIP_API_ID_hipMemcpy2DFromArray: return "hipMemcpy2DFromArray";
    case HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags: return "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags";
    case HIP_API_ID_hipSetDeviceFlags: return "hipSetDeviceFlags";
    case HIP_API_ID_hipHccModuleLaunchKernel: return "hipHccModuleLaunchKernel";
    case HIP_API_ID_hipFree: return "hipFree";
    case HIP_API_ID_hipOccupancyMaxPotentialBlockSize: return "hipOccupancyMaxPotentialBlockSize";
    case HIP_API_ID_hipDeviceGetAttribute: return "hipDeviceGetAttribute";
    case HIP_API_ID_hipMemcpyDtoH: return "hipMemcpyDtoH";
    case HIP_API_ID_hipCtxDisablePeerAccess: return "hipCtxDisablePeerAccess";
    case HIP_API_ID_hipMallocManaged: return "hipMallocManaged";
    case HIP_API_ID_hipDeviceGetByPCIBusId: return "hipDeviceGetByPCIBusId";
    case HIP_API_ID_hipIpcGetMemHandle: return "hipIpcGetMemHandle";
    case HIP_API_ID_hipMemcpyHtoDAsync: return "hipMemcpyHtoDAsync";
    case HIP_API_ID_hipCtxGetDevice: return "hipCtxGetDevice";
    case HIP_API_ID_hipMemset3D: return "hipMemset3D";
    case HIP_API_ID_hipModuleLoadData: return "hipModuleLoadData";
    case HIP_API_ID_hipDeviceTotalMem: return "hipDeviceTotalMem";
    case HIP_API_ID_hipOccupancyMaxActiveBlocksPerMultiprocessor: return "hipOccupancyMaxActiveBlocksPerMultiprocessor";
    case HIP_API_ID_hipCtxSetCurrent: return "hipCtxSetCurrent";
    case HIP_API_ID_hipGetErrorString: return "hipGetErrorString";
    case HIP_API_ID_hipDevicePrimaryCtxRetain: return "hipDevicePrimaryCtxRetain";
    case HIP_API_ID_hipDeviceDisablePeerAccess: return "hipDeviceDisablePeerAccess";
    case HIP_API_ID_hipStreamCreateWithFlags: return "hipStreamCreateWithFlags";
    case HIP_API_ID_hipMemcpyFromArray: return "hipMemcpyFromArray";
    case HIP_API_ID_hipMemcpy2DAsync: return "hipMemcpy2DAsync";
    case HIP_API_ID_hipFuncGetAttributes: return "hipFuncGetAttributes";
    case HIP_API_ID_hipGetSymbolSize: return "hipGetSymbolSize";
    case HIP_API_ID_hipEventCreateWithFlags: return "hipEventCreateWithFlags";
    case HIP_API_ID_hipStreamQuery: return "hipStreamQuery";
    case HIP_API_ID_hipDeviceGetPCIBusId: return "hipDeviceGetPCIBusId";
    case HIP_API_ID_hipMemcpy: return "hipMemcpy";
    case HIP_API_ID_hipPeekAtLastError: return "hipPeekAtLastError";
    case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice: return "hipExtLaunchMultiKernelMultiDevice";
    case HIP_API_ID_hipStreamAddCallback: return "hipStreamAddCallback";
    case HIP_API_ID_hipMemcpyToArray: return "hipMemcpyToArray";
    case HIP_API_ID_hipMemsetD32: return "hipMemsetD32";
    case HIP_API_ID_hipDeviceSynchronize: return "hipDeviceSynchronize";
    case HIP_API_ID_hipDeviceGetCacheConfig: return "hipDeviceGetCacheConfig";
    case HIP_API_ID_hipMalloc3D: return "hipMalloc3D";
    case HIP_API_ID_hipPointerGetAttributes: return "hipPointerGetAttributes";
    case HIP_API_ID_hipMemsetAsync: return "hipMemsetAsync";
    case HIP_API_ID_hipMemcpyToSymbol: return "hipMemcpyToSymbol";
    case HIP_API_ID_hipCtxPushCurrent: return "hipCtxPushCurrent";
    case HIP_API_ID_hipMemcpyPeer: return "hipMemcpyPeer";
    case HIP_API_ID_hipEventSynchronize: return "hipEventSynchronize";
    case HIP_API_ID_hipMemcpyDtoDAsync: return "hipMemcpyDtoDAsync";
    case HIP_API_ID_hipExtMallocWithFlags: return "hipExtMallocWithFlags";
    case HIP_API_ID_hipCtxEnablePeerAccess: return "hipCtxEnablePeerAccess";
    case HIP_API_ID_hipMemcpyDtoHAsync: return "hipMemcpyDtoHAsync";
    case HIP_API_ID_hipModuleLaunchKernel: return "hipModuleLaunchKernel";
    case HIP_API_ID_hipMemAllocPitch: return "hipMemAllocPitch";
    case HIP_API_ID_hipMemcpy2DFromArrayAsync: return "hipMemcpy2DFromArrayAsync";
    case HIP_API_ID_hipDeviceGetLimit: return "hipDeviceGetLimit";
    case HIP_API_ID_hipModuleLoadDataEx: return "hipModuleLoadDataEx";
    case HIP_API_ID_hipRuntimeGetVersion: return "hipRuntimeGetVersion";
    case HIP_API_ID___hipPopCallConfiguration: return "__hipPopCallConfiguration";
    case HIP_API_ID_hipGetDeviceProperties: return "hipGetDeviceProperties";
    case HIP_API_ID_hipFreeArray: return "hipFreeArray";
    case HIP_API_ID_hipEventElapsedTime: return "hipEventElapsedTime";
    case HIP_API_ID_hipDevicePrimaryCtxRelease: return "hipDevicePrimaryCtxRelease";
    case HIP_API_ID_hipHostGetDevicePointer: return "hipHostGetDevicePointer";
    case HIP_API_ID_hipMemcpyParam2D: return "hipMemcpyParam2D";
    case HIP_API_ID_hipModuleGetFunction: return "hipModuleGetFunction";
    case HIP_API_ID_hipMemsetD32Async: return "hipMemsetD32Async";
    case HIP_API_ID_hipGetDevice: return "hipGetDevice";
    case HIP_API_ID_hipGetDeviceCount: return "hipGetDeviceCount";
  };
  return "unknown";
};

// HIP API callbacks data structure
typedef struct hip_api_data_t {
  uint64_t correlation_id;
  uint32_t phase;
  union {
    struct {
      hipStream_t* stream;
      unsigned int flags;
      int priority;
    } hipStreamCreateWithPriority;
    struct {
      const void* symbolName;
      const void* src;
      size_t sizeBytes;
      size_t offset;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpyToSymbolAsync;
    struct {
      void** ptr;
      size_t* pitch;
      size_t width;
      size_t height;
    } hipMallocPitch;
    struct {
      void** ptr;
      size_t size;
    } hipMalloc;
    struct {
      hipDeviceptr_t dest;
      unsigned short value;
      size_t count;
    } hipMemsetD16;
    struct {
      char* name;
      int len;
      hipDevice_t device;
    } hipDeviceGetName;
    struct {
      hipEvent_t event;
      hipStream_t stream;
    } hipEventRecord;
    struct {
      int deviceId;
    } hipSetDevice;
    struct {
      hipCtx_t ctx;
      int* apiVersion;
    } hipCtxGetApiVersion;
    struct {
      const void* arg;
      size_t size;
      size_t offset;
    } hipSetupArgument;
    struct {
      void* dst;
      const void* symbolName;
      size_t sizeBytes;
      size_t offset;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpyFromSymbolAsync;
    struct {
      int device1;
      int device2;
      unsigned int* linktype;
      unsigned int* hopcount;
    } hipExtGetLinkTypeAndHopCount;
    struct {
      hipDeviceptr_t dst;
      hipDeviceptr_t src;
      size_t sizeBytes;
    } hipMemcpyDtoD;
    struct {
      void* ptr;
    } hipHostFree;
    struct {
      hipArray* dst;
      size_t wOffset;
      size_t hOffset;
      const void* src;
      size_t spitch;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
    } hipMemcpy2DToArray;
    struct {
      hipDeviceptr_t dest;
      unsigned char value;
      size_t count;
      hipStream_t stream;
    } hipMemsetD8Async;
    struct {
      hipFuncCache_t* cacheConfig;
    } hipCtxGetCacheConfig;
    struct {
      hipStream_t stream;
      hipEvent_t event;
      unsigned int flags;
    } hipStreamWaitEvent;
    struct {
      int* leastPriority;
      int* greatestPriority;
    } hipDeviceGetStreamPriorityRange;
    struct {
      hipModule_t* module;
      const char* fname;
    } hipModuleLoad;
    struct {
      hipDevice_t dev;
      unsigned int flags;
    } hipDevicePrimaryCtxSetFlags;
    struct {
      const void* f;
      dim3 gridDim;
      dim3 blockDimX;
      void** kernelParams;
      unsigned int sharedMemBytes;
      hipStream_t stream;
    } hipLaunchCooperativeKernel;
    struct {
      hipLaunchParams* launchParamsList;
      int numDevices;
      unsigned int flags;
    } hipLaunchCooperativeKernelMultiDevice;
    struct {
      void* dst;
      const void* src;
      size_t sizeBytes;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpyAsync;
    struct {
      hipArray** array;
      const hipChannelFormatDesc* desc;
      hipExtent extent;
      unsigned int flags;
    } hipMalloc3DArray;
    struct {
      hipStream_t* stream;
    } hipStreamCreate;
    struct {
      hipCtx_t* ctx;
    } hipCtxGetCurrent;
    struct {
      hipDevice_t dev;
      unsigned int* flags;
      int* active;
    } hipDevicePrimaryCtxGetState;
    struct {
      hipEvent_t event;
    } hipEventQuery;
    struct {
      hipEvent_t* event;
    } hipEventCreate;
    struct {
      hipDeviceptr_t* pbase;
      size_t* psize;
      hipDeviceptr_t dptr;
    } hipMemGetAddressRange;
    struct {
      void* dst;
      const void* symbolName;
      size_t sizeBytes;
      size_t offset;
      hipMemcpyKind kind;
    } hipMemcpyFromSymbol;
    struct {
      hipArray** pHandle;
      const HIP_ARRAY_DESCRIPTOR* pAllocateArray;
    } hipArrayCreate;
    struct {
      hipStream_t stream;
      unsigned int* flags;
    } hipStreamGetFlags;
    struct {
      hipArray** array;
      const hipChannelFormatDesc* desc;
      size_t width;
      size_t height;
      unsigned int flags;
    } hipMallocArray;
    struct {
      hipSharedMemConfig* pConfig;
    } hipCtxGetSharedMemConfig;
    struct {
      void* ptr;
      size_t* size;
    } hipMemPtrGetInfo;
    struct {
      int* value;
      hipFunction_attribute attrib;
      hipFunction_t hfunc;
    } hipFuncGetAttribute;
    struct {
      unsigned int* flags;
    } hipCtxGetFlags;
    struct {
      hipStream_t stream;
    } hipStreamDestroy;
    struct {
      dim3 gridDim;
      dim3 blockDim;
      size_t sharedMem;
      hipStream_t stream;
    } __hipPushCallConfiguration;
    struct {
      hipPitchedPtr pitchedDevPtr;
      int value;
      hipExtent extent;
      hipStream_t stream;
    } hipMemset3DAsync;
    struct {
      const hipMemcpy3DParms* p;
    } hipMemcpy3D;
    struct {
      unsigned int flags;
    } hipInit;
    struct {
      void* dst;
      hipArray* srcArray;
      size_t srcOffset;
      size_t count;
    } hipMemcpyAtoH;
    struct {
      hipStream_t stream;
      int* priority;
    } hipStreamGetPriority;
    struct {
      void* dst;
      size_t pitch;
      int value;
      size_t width;
      size_t height;
    } hipMemset2D;
    struct {
      void* dst;
      size_t pitch;
      int value;
      size_t width;
      size_t height;
      hipStream_t stream;
    } hipMemset2DAsync;
    struct {
      int* canAccessPeer;
      int deviceId;
      int peerDeviceId;
    } hipDeviceCanAccessPeer;
    struct {
      int peerDeviceId;
      unsigned int flags;
    } hipDeviceEnablePeerAccess;
    struct {
      const void* function_address;
      dim3 numBlocks;
      dim3 dimBlocks;
      void** args;
      size_t sharedMemBytes;
      hipStream_t stream;
    } hipLaunchKernel;
    struct {
      hipDeviceptr_t dest;
      unsigned short value;
      size_t count;
      hipStream_t stream;
    } hipMemsetD16Async;
    struct {
      hipModule_t module;
    } hipModuleUnload;
    struct {
      void* hostPtr;
    } hipHostUnregister;
    struct {
      const void* func;
    } hipLaunchByPtr;
    struct {
      hipStream_t stream;
    } hipStreamSynchronize;
    struct {
      hipFuncCache_t cacheConfig;
    } hipDeviceSetCacheConfig;
    struct {
      hipError_t hip_error;
    } hipGetErrorName;
    struct {
      hipDeviceptr_t dst;
      void* src;
      size_t sizeBytes;
    } hipMemcpyHtoD;
    struct {
      hipDeviceptr_t* dptr;
      size_t* bytes;
      hipModule_t hmod;
      const char* name;
    } hipModuleGetGlobal;
    struct {
      hipArray* dstArray;
      size_t dstOffset;
      const void* srcHost;
      size_t count;
    } hipMemcpyHtoA;
    struct {
      hipCtx_t* ctx;
      unsigned int flags;
      hipDevice_t device;
    } hipCtxCreate;
    struct {
      void* dst;
      size_t dpitch;
      const void* src;
      size_t spitch;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
    } hipMemcpy2D;
    struct {
      void* devPtr;
    } hipIpcCloseMemHandle;
    struct {
      int* device;
      const hipDeviceProp_t* prop;
    } hipChooseDevice;
    struct {
      hipSharedMemConfig config;
    } hipDeviceSetSharedMemConfig;
    struct {
      int* major;
      int* minor;
      hipDevice_t device;
    } hipDeviceComputeCapability;
    struct {
      hipDevice_t* device;
      int ordinal;
    } hipDeviceGet;
    struct {
      hipFuncCache_t cacheConfig;
    } hipCtxSetCacheConfig;
    struct {
      const void* func;
      hipFuncCache_t config;
    } hipFuncSetCacheConfig;
    struct {
      textureReference** texRef;
      hipModule_t hmod;
      const char* name;
    } hipModuleGetTexRef;
    struct {
      void* dst;
      int dstDeviceId;
      const void* src;
      int srcDevice;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemcpyPeerAsync;
    struct {
      void* dst;
      const void* src;
      size_t sizeBytes;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpyWithStream;
    struct {
      hipDevice_t dev;
    } hipDevicePrimaryCtxReset;
    struct {
      const hipMemcpy3DParms* p;
      hipStream_t stream;
    } hipMemcpy3DAsync;
    struct {
      hipEvent_t event;
    } hipEventDestroy;
    struct {
      hipCtx_t* ctx;
    } hipCtxPopCurrent;
    struct {
      void** devPtr;
      const void* symbolName;
    } hipGetSymbolAddress;
    struct {
      unsigned int* flagsPtr;
      void* hostPtr;
    } hipHostGetFlags;
    struct {
      void** ptr;
      size_t size;
      unsigned int flags;
    } hipHostMalloc;
    struct {
      int* driverVersion;
    } hipDriverGetVersion;
    struct {
      size_t* free;
      size_t* total;
    } hipMemGetInfo;
    struct {
      void* dst;
      int value;
      size_t sizeBytes;
    } hipMemset;
    struct {
      hipDeviceptr_t dest;
      unsigned char value;
      size_t count;
    } hipMemsetD8;
    struct {
      const hip_Memcpy2D* pCopy;
      hipStream_t stream;
    } hipMemcpyParam2DAsync;
    struct {
      void* hostPtr;
      size_t sizeBytes;
      unsigned int flags;
    } hipHostRegister;
    struct {
      hipSharedMemConfig config;
    } hipCtxSetSharedMemConfig;
    struct {
      hipArray** array;
      const HIP_ARRAY3D_DESCRIPTOR* pAllocateArray;
    } hipArray3DCreate;
    struct {
      void** devPtr;
      hipIpcMemHandle_t handle;
      unsigned int flags;
    } hipIpcOpenMemHandle;
    struct {
      hipCtx_t ctx;
    } hipCtxDestroy;
    struct {
      hipSharedMemConfig* pConfig;
    } hipDeviceGetSharedMemConfig;
    struct {
      void* dst;
      size_t dpitch;
      hipArray_const_t src;
      size_t wOffset;
      size_t hOffset;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
    } hipMemcpy2DFromArray;
    struct {
      unsigned int* numBlocks;
      hipFunction_t f;
      unsigned int blockSize;
      size_t dynSharedMemPerBlk;
      unsigned int flags;
    } hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags;
    struct {
      unsigned int flags;
    } hipSetDeviceFlags;
    struct {
      hipFunction_t f;
      unsigned int globalWorkSizeX;
      unsigned int globalWorkSizeY;
      unsigned int globalWorkSizeZ;
      unsigned int localWorkSizeX;
      unsigned int localWorkSizeY;
      unsigned int localWorkSizeZ;
      size_t sharedMemBytes;
      hipStream_t hStream;
      void** kernelParams;
      void** extra;
      hipEvent_t startEvent;
      hipEvent_t stopEvent;
    } hipHccModuleLaunchKernel;
    struct {
      void* ptr;
    } hipFree;
    struct {
      unsigned int* gridSize;
      unsigned int* blockSize;
      hipFunction_t f;
      size_t dynSharedMemPerBlk;
      unsigned int blockSizeLimit;
    } hipOccupancyMaxPotentialBlockSize;
    struct {
      int* pi;
      hipDeviceAttribute_t attr;
      int deviceId;
    } hipDeviceGetAttribute;
    struct {
      void* dst;
      hipDeviceptr_t src;
      size_t sizeBytes;
    } hipMemcpyDtoH;
    struct {
      hipCtx_t peerCtx;
    } hipCtxDisablePeerAccess;
    struct {
      void** devPtr;
      size_t size;
      unsigned int flags;
    } hipMallocManaged;
    struct {
      int* device;
      const char* pciBusId;
    } hipDeviceGetByPCIBusId;
    struct {
      hipIpcMemHandle_t* handle;
      void* devPtr;
    } hipIpcGetMemHandle;
    struct {
      hipDeviceptr_t dst;
      void* src;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemcpyHtoDAsync;
    struct {
      hipDevice_t* device;
    } hipCtxGetDevice;
    struct {
      hipPitchedPtr pitchedDevPtr;
      int value;
      hipExtent extent;
    } hipMemset3D;
    struct {
      hipModule_t* module;
      const void* image;
    } hipModuleLoadData;
    struct {
      size_t* bytes;
      hipDevice_t device;
    } hipDeviceTotalMem;
    struct {
      unsigned int* numBlocks;
      hipFunction_t f;
      unsigned int blockSize;
      size_t dynSharedMemPerBlk;
    } hipOccupancyMaxActiveBlocksPerMultiprocessor;
    struct {
      hipCtx_t ctx;
    } hipCtxSetCurrent;
    struct {
      hipError_t hipError;
    } hipGetErrorString;
    struct {
      hipCtx_t* pctx;
      hipDevice_t dev;
    } hipDevicePrimaryCtxRetain;
    struct {
      int peerDeviceId;
    } hipDeviceDisablePeerAccess;
    struct {
      hipStream_t* stream;
      unsigned int flags;
    } hipStreamCreateWithFlags;
    struct {
      void* dst;
      hipArray_const_t srcArray;
      size_t wOffset;
      size_t hOffset;
      size_t count;
      hipMemcpyKind kind;
    } hipMemcpyFromArray;
    struct {
      void* dst;
      size_t dpitch;
      const void* src;
      size_t spitch;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpy2DAsync;
    struct {
      hipFuncAttributes* attr;
      const void* func;
    } hipFuncGetAttributes;
    struct {
      size_t* size;
      const void* symbolName;
    } hipGetSymbolSize;
    struct {
      hipEvent_t* event;
      unsigned int flags;
    } hipEventCreateWithFlags;
    struct {
      hipStream_t stream;
    } hipStreamQuery;
    struct {
      char* pciBusId;
      int len;
      int device;
    } hipDeviceGetPCIBusId;
    struct {
      void* dst;
      const void* src;
      size_t sizeBytes;
      hipMemcpyKind kind;
    } hipMemcpy;
    struct {
      hipLaunchParams* launchParamsList;
      int numDevices;
      unsigned int flags;
    } hipExtLaunchMultiKernelMultiDevice;
    struct {
      hipStream_t stream;
      hipStreamCallback_t callback;
      void* userData;
      unsigned int flags;
    } hipStreamAddCallback;
    struct {
      hipArray* dst;
      size_t wOffset;
      size_t hOffset;
      const void* src;
      size_t count;
      hipMemcpyKind kind;
    } hipMemcpyToArray;
    struct {
      hipDeviceptr_t dest;
      int value;
      size_t count;
    } hipMemsetD32;
    struct {
      hipFuncCache_t* cacheConfig;
    } hipDeviceGetCacheConfig;
    struct {
      hipPitchedPtr* pitchedDevPtr;
      hipExtent extent;
    } hipMalloc3D;
    struct {
      hipPointerAttribute_t* attributes;
      const void* ptr;
    } hipPointerGetAttributes;
    struct {
      void* dst;
      int value;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemsetAsync;
    struct {
      const void* symbolName;
      const void* src;
      size_t sizeBytes;
      size_t offset;
      hipMemcpyKind kind;
    } hipMemcpyToSymbol;
    struct {
      hipCtx_t ctx;
    } hipCtxPushCurrent;
    struct {
      void* dst;
      int dstDeviceId;
      const void* src;
      int srcDeviceId;
      size_t sizeBytes;
    } hipMemcpyPeer;
    struct {
      hipEvent_t event;
    } hipEventSynchronize;
    struct {
      hipDeviceptr_t dst;
      hipDeviceptr_t src;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemcpyDtoDAsync;
    struct {
      void** ptr;
      size_t sizeBytes;
      unsigned int flags;
    } hipExtMallocWithFlags;
    struct {
      hipCtx_t peerCtx;
      unsigned int flags;
    } hipCtxEnablePeerAccess;
    struct {
      void* dst;
      hipDeviceptr_t src;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemcpyDtoHAsync;
    struct {
      hipFunction_t f;
      unsigned int gridDimX;
      unsigned int gridDimY;
      unsigned int gridDimZ;
      unsigned int blockDimX;
      unsigned int blockDimY;
      unsigned int blockDimZ;
      unsigned int sharedMemBytes;
      hipStream_t stream;
      void** kernelParams;
      void** extra;
    } hipModuleLaunchKernel;
    struct {
      hipDeviceptr_t* dptr;
      size_t* pitch;
      size_t widthInBytes;
      size_t height;
      unsigned int elementSizeBytes;
    } hipMemAllocPitch;
    struct {
      void* dst;
      size_t dpitch;
      hipArray_const_t src;
      size_t wOffset;
      size_t hOffset;
      size_t width;
      size_t height;
      hipMemcpyKind kind;
      hipStream_t stream;
    } hipMemcpy2DFromArrayAsync;
    struct {
      size_t* pValue;
      enum hipLimit_t limit;
    } hipDeviceGetLimit;
    struct {
      hipModule_t* module;
      const void* image;
      unsigned int numOptions;
      hipJitOption* options;
      void** optionValues;
    } hipModuleLoadDataEx;
    struct {
      int* runtimeVersion;
    } hipRuntimeGetVersion;
    struct {
      dim3 *gridDim;
      dim3 *blockDim;
      size_t *sharedMem;
      hipStream_t *stream;
    } __hipPopCallConfiguration;
    struct {
      hipDeviceProp_t* prop;
      int deviceId;
    } hipGetDeviceProperties;
    struct {
      hipArray* array;
    } hipFreeArray;
    struct {
      float* ms;
      hipEvent_t start;
      hipEvent_t stop;
    } hipEventElapsedTime;
    struct {
      hipDevice_t dev;
    } hipDevicePrimaryCtxRelease;
    struct {
      void** devPtr;
      void* hstPtr;
      unsigned int flags;
    } hipHostGetDevicePointer;
    struct {
      const hip_Memcpy2D* pCopy;
    } hipMemcpyParam2D;
    struct {
      hipFunction_t* function;
      hipModule_t module;
      const char* kname;
    } hipModuleGetFunction;
    struct {
      hipDeviceptr_t dst;
      int value;
      size_t count;
      hipStream_t stream;
    } hipMemsetD32Async;
    struct {
      int* deviceId;
    } hipGetDevice;
    struct {
      int* count;
    } hipGetDeviceCount;
  } args;
} hip_api_data_t;

// HIP API callbacks args data filling macros
// hipStreamCreateWithPriority[('hipStream_t*', 'stream'), ('unsigned int', 'flags'), ('int', 'priority')]
#define INIT_hipStreamCreateWithPriority_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreateWithPriority.stream = stream; \
  cb_data.args.hipStreamCreateWithPriority.flags = flags; \
  cb_data.args.hipStreamCreateWithPriority.priority = priority; \
};
// hipMemcpyToSymbolAsync[('const void*', 'symbolName'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyToSymbolAsync_CB_ARGS_DATA(cb_data) { \
};
// hipMallocPitch[('void**', 'ptr'), ('size_t*', 'pitch'), ('size_t', 'width'), ('size_t', 'height')]
#define INIT_hipMallocPitch_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocPitch.ptr = ptr; \
  cb_data.args.hipMallocPitch.pitch = pitch; \
  cb_data.args.hipMallocPitch.width = width; \
  cb_data.args.hipMallocPitch.height = height; \
};
// hipMalloc[('void**', 'ptr'), ('size_t', 'size')]
#define INIT_hipMalloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc.ptr = ptr; \
  cb_data.args.hipMalloc.size = sizeBytes; \
};
// hipMemsetD16[('hipDeviceptr_t', 'dest'), ('unsigned short', 'value'), ('size_t', 'count')]
#define INIT_hipMemsetD16_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD16.dest = dst; \
  cb_data.args.hipMemsetD16.value = value; \
  cb_data.args.hipMemsetD16.count = count; \
};
// hipDeviceGetName[('char*', 'name'), ('int', 'len'), ('hipDevice_t', 'device')]
#define INIT_hipDeviceGetName_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetName.name = name; \
  cb_data.args.hipDeviceGetName.len = len; \
  cb_data.args.hipDeviceGetName.device = device; \
};
// hipEventRecord[('hipEvent_t', 'event'), ('hipStream_t', 'stream')]
#define INIT_hipEventRecord_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventRecord.event = event; \
  cb_data.args.hipEventRecord.stream = stream; \
};
// hipCtxSynchronize[]
#define INIT_hipCtxSynchronize_CB_ARGS_DATA(cb_data) { \
};
// hipSetDevice[('int', 'deviceId')]
#define INIT_hipSetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetDevice.deviceId = deviceId; \
};
// hipCtxGetApiVersion[('hipCtx_t', 'ctx'), ('int*', 'apiVersion')]
#define INIT_hipCtxGetApiVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetApiVersion.ctx = ctx; \
  cb_data.args.hipCtxGetApiVersion.apiVersion = apiVersion; \
};
// hipSetupArgument[('const void*', 'arg'), ('size_t', 'size'), ('size_t', 'offset')]
#define INIT_hipSetupArgument_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetupArgument.arg = arg; \
  cb_data.args.hipSetupArgument.size = size; \
  cb_data.args.hipSetupArgument.offset = offset; \
};
// hipMemcpyFromSymbolAsync[('void*', 'dst'), ('const void*', 'symbolName'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyFromSymbolAsync_CB_ARGS_DATA(cb_data) { \
};
// hipExtGetLinkTypeAndHopCount[('int', 'device1'), ('int', 'device2'), ('unsigned int*', 'linktype'), ('unsigned int*', 'hopcount')]
#define INIT_hipExtGetLinkTypeAndHopCount_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipExtGetLinkTypeAndHopCount.device1 = device1; \
  cb_data.args.hipExtGetLinkTypeAndHopCount.device2 = device2; \
  cb_data.args.hipExtGetLinkTypeAndHopCount.linktype = linktype; \
  cb_data.args.hipExtGetLinkTypeAndHopCount.hopcount = hopcount; \
};
// hipMemcpyDtoD[('hipDeviceptr_t', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes')]
#define INIT_hipMemcpyDtoD_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoD.dst = dst; \
  cb_data.args.hipMemcpyDtoD.src = src; \
  cb_data.args.hipMemcpyDtoD.sizeBytes = sizeBytes; \
};
// hipHostFree[('void*', 'ptr')]
#define INIT_hipHostFree_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostFree.ptr = ptr; \
};
// hipMemcpy2DToArray[('hipArray*', 'dst'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('const void*', 'src'), ('size_t', 'spitch'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpy2DToArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DToArray.dst = dst; \
  cb_data.args.hipMemcpy2DToArray.wOffset = wOffset; \
  cb_data.args.hipMemcpy2DToArray.hOffset = hOffset; \
  cb_data.args.hipMemcpy2DToArray.src = src; \
  cb_data.args.hipMemcpy2DToArray.spitch = spitch; \
  cb_data.args.hipMemcpy2DToArray.width = width; \
  cb_data.args.hipMemcpy2DToArray.height = height; \
  cb_data.args.hipMemcpy2DToArray.kind = kind; \
};
// hipMemsetD8Async[('hipDeviceptr_t', 'dest'), ('unsigned char', 'value'), ('size_t', 'count'), ('hipStream_t', 'stream')]
#define INIT_hipMemsetD8Async_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD8Async.dest = dst; \
  cb_data.args.hipMemsetD8Async.value = value; \
  cb_data.args.hipMemsetD8Async.count = count; \
  cb_data.args.hipMemsetD8Async.stream = stream; \
};
// hipCtxGetCacheConfig[('hipFuncCache_t*', 'cacheConfig')]
#define INIT_hipCtxGetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetCacheConfig.cacheConfig = cacheConfig; \
};
// hipStreamWaitEvent[('hipStream_t', 'stream'), ('hipEvent_t', 'event'), ('unsigned int', 'flags')]
#define INIT_hipStreamWaitEvent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamWaitEvent.stream = stream; \
  cb_data.args.hipStreamWaitEvent.event = event; \
  cb_data.args.hipStreamWaitEvent.flags = flags; \
};
// hipDeviceGetStreamPriorityRange[('int*', 'leastPriority'), ('int*', 'greatestPriority')]
#define INIT_hipDeviceGetStreamPriorityRange_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetStreamPriorityRange.leastPriority = leastPriority; \
  cb_data.args.hipDeviceGetStreamPriorityRange.greatestPriority = greatestPriority; \
};
// hipModuleLoad[('hipModule_t*', 'module'), ('const char*', 'fname')]
#define INIT_hipModuleLoad_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoad.module = module; \
  cb_data.args.hipModuleLoad.fname = fname; \
};
// hipDevicePrimaryCtxSetFlags[('hipDevice_t', 'dev'), ('unsigned int', 'flags')]
#define INIT_hipDevicePrimaryCtxSetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxSetFlags.dev = dev; \
  cb_data.args.hipDevicePrimaryCtxSetFlags.flags = flags; \
};
// hipLaunchCooperativeKernel[('const void*', 'f'), ('dim3', 'gridDim'), ('dim3', 'blockDimX'), ('void**', 'kernelParams'), ('unsigned int', 'sharedMemBytes'), ('hipStream_t', 'stream')]
#define INIT_hipLaunchCooperativeKernel_CB_ARGS_DATA(cb_data) { \
};
// hipLaunchCooperativeKernelMultiDevice[('hipLaunchParams*', 'launchParamsList'), ('int', 'numDevices'), ('unsigned int', 'flags')]
#define INIT_hipLaunchCooperativeKernelMultiDevice_CB_ARGS_DATA(cb_data) { \
};
// hipMemcpyAsync[('void*', 'dst'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyAsync.dst = dst; \
  cb_data.args.hipMemcpyAsync.src = src; \
  cb_data.args.hipMemcpyAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyAsync.kind = kind; \
  cb_data.args.hipMemcpyAsync.stream = stream; \
};
// hipMalloc3DArray[('hipArray**', 'array'), ('const hipChannelFormatDesc*', 'desc'), ('hipExtent', 'extent'), ('unsigned int', 'flags')]
#define INIT_hipMalloc3DArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc3DArray.array = array; \
  cb_data.args.hipMalloc3DArray.desc = desc; \
  cb_data.args.hipMalloc3DArray.extent = extent; \
  cb_data.args.hipMalloc3DArray.flags = flags; \
};
// hipStreamCreate[('hipStream_t*', 'stream')]
#define INIT_hipStreamCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreate.stream = stream; \
};
// hipCtxGetCurrent[('hipCtx_t*', 'ctx')]
#define INIT_hipCtxGetCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetCurrent.ctx = ctx; \
};
// hipDevicePrimaryCtxGetState[('hipDevice_t', 'dev'), ('unsigned int*', 'flags'), ('int*', 'active')]
#define INIT_hipDevicePrimaryCtxGetState_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxGetState.dev = dev; \
  cb_data.args.hipDevicePrimaryCtxGetState.flags = flags; \
  cb_data.args.hipDevicePrimaryCtxGetState.active = active; \
};
// hipEventQuery[('hipEvent_t', 'event')]
#define INIT_hipEventQuery_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventQuery.event = event; \
};
// hipEventCreate[('hipEvent_t*', 'event')]
#define INIT_hipEventCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventCreate.event = event; \
};
// hipMemGetAddressRange[('hipDeviceptr_t*', 'pbase'), ('size_t*', 'psize'), ('hipDeviceptr_t', 'dptr')]
#define INIT_hipMemGetAddressRange_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemGetAddressRange.pbase = pbase; \
  cb_data.args.hipMemGetAddressRange.psize = psize; \
  cb_data.args.hipMemGetAddressRange.dptr = dptr; \
};
// hipMemcpyFromSymbol[('void*', 'dst'), ('const void*', 'symbolName'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpyFromSymbol_CB_ARGS_DATA(cb_data) { \
};
// hipArrayCreate[('hipArray**', 'pHandle'), ('const HIP_ARRAY_DESCRIPTOR*', 'pAllocateArray')]
#define INIT_hipArrayCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipArrayCreate.pHandle = array; \
  cb_data.args.hipArrayCreate.pAllocateArray = pAllocateArray; \
};
// hipStreamGetFlags[('hipStream_t', 'stream'), ('unsigned int*', 'flags')]
#define INIT_hipStreamGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamGetFlags.stream = stream; \
  cb_data.args.hipStreamGetFlags.flags = flags; \
};
// hipMallocArray[('hipArray**', 'array'), ('const hipChannelFormatDesc*', 'desc'), ('size_t', 'width'), ('size_t', 'height'), ('unsigned int', 'flags')]
#define INIT_hipMallocArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocArray.array = array; \
  cb_data.args.hipMallocArray.desc = desc; \
  cb_data.args.hipMallocArray.width = width; \
  cb_data.args.hipMallocArray.height = height; \
  cb_data.args.hipMallocArray.flags = flags; \
};
// hipCtxGetSharedMemConfig[('hipSharedMemConfig*', 'pConfig')]
#define INIT_hipCtxGetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetSharedMemConfig.pConfig = pConfig; \
};
// hipMemPtrGetInfo[('void*', 'ptr'), ('size_t*', 'size')]
#define INIT_hipMemPtrGetInfo_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemPtrGetInfo.ptr = ptr; \
  cb_data.args.hipMemPtrGetInfo.size = size; \
};
// hipFuncGetAttribute[('int*', 'value'), ('hipFunction_attribute', 'attrib'), ('hipFunction_t', 'hfunc')]
#define INIT_hipFuncGetAttribute_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncGetAttribute.value = value; \
  cb_data.args.hipFuncGetAttribute.attrib = attrib; \
  cb_data.args.hipFuncGetAttribute.hfunc = hfunc; \
};
// hipCtxGetFlags[('unsigned int*', 'flags')]
#define INIT_hipCtxGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetFlags.flags = flags; \
};
// hipStreamDestroy[('hipStream_t', 'stream')]
#define INIT_hipStreamDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamDestroy.stream = stream; \
};
// __hipPushCallConfiguration[('dim3', 'gridDim'), ('dim3', 'blockDim'), ('size_t', 'sharedMem'), ('hipStream_t', 'stream')]
#define INIT___hipPushCallConfiguration_CB_ARGS_DATA(cb_data) { \
};
// hipMemset3DAsync[('hipPitchedPtr', 'pitchedDevPtr'), ('int', 'value'), ('hipExtent', 'extent'), ('hipStream_t', 'stream')]
#define INIT_hipMemset3DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset3DAsync.pitchedDevPtr = pitchedDevPtr; \
  cb_data.args.hipMemset3DAsync.value = value; \
  cb_data.args.hipMemset3DAsync.extent = extent; \
  cb_data.args.hipMemset3DAsync.stream = stream; \
};
// hipMemcpy3D[('const hipMemcpy3DParms*', 'p')]
#define INIT_hipMemcpy3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy3D.p = p; \
};
// hipInit[('unsigned int', 'flags')]
#define INIT_hipInit_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipInit.flags = flags; \
};
// hipMemcpyAtoH[('void*', 'dst'), ('hipArray*', 'srcArray'), ('size_t', 'srcOffset'), ('size_t', 'count')]
#define INIT_hipMemcpyAtoH_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyAtoH.dst = dst; \
  cb_data.args.hipMemcpyAtoH.srcArray = srcArray; \
  cb_data.args.hipMemcpyAtoH.srcOffset = srcOffset; \
  cb_data.args.hipMemcpyAtoH.count = count; \
};
// hipStreamGetPriority[('hipStream_t', 'stream'), ('int*', 'priority')]
#define INIT_hipStreamGetPriority_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamGetPriority.stream = stream; \
  cb_data.args.hipStreamGetPriority.priority = priority; \
};
// hipMemset2D[('void*', 'dst'), ('size_t', 'pitch'), ('int', 'value'), ('size_t', 'width'), ('size_t', 'height')]
#define INIT_hipMemset2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset2D.dst = dst; \
  cb_data.args.hipMemset2D.pitch = pitch; \
  cb_data.args.hipMemset2D.value = value; \
  cb_data.args.hipMemset2D.width = width; \
  cb_data.args.hipMemset2D.height = height; \
};
// hipMemset2DAsync[('void*', 'dst'), ('size_t', 'pitch'), ('int', 'value'), ('size_t', 'width'), ('size_t', 'height'), ('hipStream_t', 'stream')]
#define INIT_hipMemset2DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset2DAsync.dst = dst; \
  cb_data.args.hipMemset2DAsync.pitch = pitch; \
  cb_data.args.hipMemset2DAsync.value = value; \
  cb_data.args.hipMemset2DAsync.width = width; \
  cb_data.args.hipMemset2DAsync.height = height; \
  cb_data.args.hipMemset2DAsync.stream = stream; \
};
// hipDeviceCanAccessPeer[('int*', 'canAccessPeer'), ('int', 'deviceId'), ('int', 'peerDeviceId')]
#define INIT_hipDeviceCanAccessPeer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceCanAccessPeer.canAccessPeer = canAccessPeer; \
  cb_data.args.hipDeviceCanAccessPeer.deviceId = deviceId; \
  cb_data.args.hipDeviceCanAccessPeer.peerDeviceId = peerDeviceId; \
};
// hipDeviceEnablePeerAccess[('int', 'peerDeviceId'), ('unsigned int', 'flags')]
#define INIT_hipDeviceEnablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceEnablePeerAccess.peerDeviceId = peerDeviceId; \
  cb_data.args.hipDeviceEnablePeerAccess.flags = flags; \
};
// hipLaunchKernel[('const void*', 'function_address'), ('dim3', 'numBlocks'), ('dim3', 'dimBlocks'), ('void**', 'args'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'stream')]
#define INIT_hipLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipLaunchKernel.function_address = func_addr; \
  cb_data.args.hipLaunchKernel.numBlocks = numBlocks; \
  cb_data.args.hipLaunchKernel.dimBlocks = dimBlocks; \
  cb_data.args.hipLaunchKernel.args = args; \
  cb_data.args.hipLaunchKernel.sharedMemBytes = sharedMemBytes; \
  cb_data.args.hipLaunchKernel.stream = stream; \
};
// hipMemsetD16Async[('hipDeviceptr_t', 'dest'), ('unsigned short', 'value'), ('size_t', 'count'), ('hipStream_t', 'stream')]
#define INIT_hipMemsetD16Async_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD16Async.dest = dst; \
  cb_data.args.hipMemsetD16Async.value = value; \
  cb_data.args.hipMemsetD16Async.count = count; \
  cb_data.args.hipMemsetD16Async.stream = stream; \
};
// hipModuleUnload[('hipModule_t', 'module')]
#define INIT_hipModuleUnload_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleUnload.module = hmod; \
};
// hipHostUnregister[('void*', 'hostPtr')]
#define INIT_hipHostUnregister_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostUnregister.hostPtr = hostPtr; \
};
// hipProfilerStop[]
#define INIT_hipProfilerStop_CB_ARGS_DATA(cb_data) { \
};
// hipLaunchByPtr[('const void*', 'func')]
#define INIT_hipLaunchByPtr_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipLaunchByPtr.func = hostFunction; \
};
// hipStreamSynchronize[('hipStream_t', 'stream')]
#define INIT_hipStreamSynchronize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamSynchronize.stream = stream; \
};
// hipDeviceSetCacheConfig[('hipFuncCache_t', 'cacheConfig')]
#define INIT_hipDeviceSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceSetCacheConfig.cacheConfig = cacheConfig; \
};
// hipGetErrorName[('hipError_t', 'hip_error')]
#define INIT_hipGetErrorName_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetErrorName.hip_error = hip_error; \
};
// hipMemcpyHtoD[('hipDeviceptr_t', 'dst'), ('void*', 'src'), ('size_t', 'sizeBytes')]
#define INIT_hipMemcpyHtoD_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoD.dst = dst; \
  cb_data.args.hipMemcpyHtoD.src = src; \
  cb_data.args.hipMemcpyHtoD.sizeBytes = sizeBytes; \
};
// hipModuleGetGlobal[('hipDeviceptr_t*', 'dptr'), ('size_t*', 'bytes'), ('hipModule_t', 'hmod'), ('const char*', 'name')]
#define INIT_hipModuleGetGlobal_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetGlobal.dptr = dptr; \
  cb_data.args.hipModuleGetGlobal.bytes = bytes; \
  cb_data.args.hipModuleGetGlobal.hmod = hmod; \
  cb_data.args.hipModuleGetGlobal.name = name; \
};
// hipMemcpyHtoA[('hipArray*', 'dstArray'), ('size_t', 'dstOffset'), ('const void*', 'srcHost'), ('size_t', 'count')]
#define INIT_hipMemcpyHtoA_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoA.dstArray = dstArray; \
  cb_data.args.hipMemcpyHtoA.dstOffset = dstOffset; \
  cb_data.args.hipMemcpyHtoA.srcHost = srcHost; \
  cb_data.args.hipMemcpyHtoA.count = count; \
};
// hipCtxCreate[('hipCtx_t*', 'ctx'), ('unsigned int', 'flags'), ('hipDevice_t', 'device')]
#define INIT_hipCtxCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxCreate.ctx = ctx; \
  cb_data.args.hipCtxCreate.flags = flags; \
  cb_data.args.hipCtxCreate.device = device; \
};
// hipMemcpy2D[('void*', 'dst'), ('size_t', 'dpitch'), ('const void*', 'src'), ('size_t', 'spitch'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpy2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2D.dst = dst; \
  cb_data.args.hipMemcpy2D.dpitch = dpitch; \
  cb_data.args.hipMemcpy2D.src = src; \
  cb_data.args.hipMemcpy2D.spitch = spitch; \
  cb_data.args.hipMemcpy2D.width = width; \
  cb_data.args.hipMemcpy2D.height = height; \
  cb_data.args.hipMemcpy2D.kind = kind; \
};
// hipIpcCloseMemHandle[('void*', 'devPtr')]
#define INIT_hipIpcCloseMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcCloseMemHandle.devPtr = devPtr; \
};
// hipChooseDevice[('int*', 'device'), ('const hipDeviceProp_t*', 'prop')]
#define INIT_hipChooseDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipChooseDevice.device = device; \
  cb_data.args.hipChooseDevice.prop = prop; \
};
// hipDeviceSetSharedMemConfig[('hipSharedMemConfig', 'config')]
#define INIT_hipDeviceSetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceSetSharedMemConfig.config = config; \
};
// hipDeviceComputeCapability[('int*', 'major'), ('int*', 'minor'), ('hipDevice_t', 'device')]
#define INIT_hipDeviceComputeCapability_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceComputeCapability.major = major; \
  cb_data.args.hipDeviceComputeCapability.minor = minor; \
  cb_data.args.hipDeviceComputeCapability.device = device; \
};
// hipDeviceGet[('hipDevice_t*', 'device'), ('int', 'ordinal')]
#define INIT_hipDeviceGet_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGet.device = device; \
  cb_data.args.hipDeviceGet.ordinal = deviceId; \
};
// hipProfilerStart[]
#define INIT_hipProfilerStart_CB_ARGS_DATA(cb_data) { \
};
// hipCtxSetCacheConfig[('hipFuncCache_t', 'cacheConfig')]
#define INIT_hipCtxSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetCacheConfig.cacheConfig = cacheConfig; \
};
// hipFuncSetCacheConfig[('const void*', 'func'), ('hipFuncCache_t', 'config')]
#define INIT_hipFuncSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncSetCacheConfig.func = func; \
  cb_data.args.hipFuncSetCacheConfig.config = cacheConfig; \
};
// hipModuleGetTexRef[('textureReference**', 'texRef'), ('hipModule_t', 'hmod'), ('const char*', 'name')]
#define INIT_hipModuleGetTexRef_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetTexRef.texRef = texRef; \
  cb_data.args.hipModuleGetTexRef.hmod = hmod; \
  cb_data.args.hipModuleGetTexRef.name = name; \
};
// hipMemcpyPeerAsync[('void*', 'dst'), ('int', 'dstDeviceId'), ('const void*', 'src'), ('int', 'srcDevice'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyPeerAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyPeerAsync.dst = dst; \
  cb_data.args.hipMemcpyPeerAsync.dstDeviceId = dstDevice; \
  cb_data.args.hipMemcpyPeerAsync.src = src; \
  cb_data.args.hipMemcpyPeerAsync.srcDevice = srcDevice; \
  cb_data.args.hipMemcpyPeerAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyPeerAsync.stream = stream; \
};
// hipMemcpyWithStream[('void*', 'dst'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyWithStream_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyWithStream.dst = dst; \
  cb_data.args.hipMemcpyWithStream.src = src; \
  cb_data.args.hipMemcpyWithStream.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyWithStream.kind = kind; \
  cb_data.args.hipMemcpyWithStream.stream = stream; \
};
// hipDevicePrimaryCtxReset[('hipDevice_t', 'dev')]
#define INIT_hipDevicePrimaryCtxReset_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxReset.dev = dev; \
};
// hipMemcpy3DAsync[('const hipMemcpy3DParms*', 'p'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpy3DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy3DAsync.p = p; \
  cb_data.args.hipMemcpy3DAsync.stream = stream; \
};
// hipEventDestroy[('hipEvent_t', 'event')]
#define INIT_hipEventDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventDestroy.event = event; \
};
// hipCtxPopCurrent[('hipCtx_t*', 'ctx')]
#define INIT_hipCtxPopCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxPopCurrent.ctx = ctx; \
};
// hipGetSymbolAddress[('void**', 'devPtr'), ('const void*', 'symbolName')]
#define INIT_hipGetSymbolAddress_CB_ARGS_DATA(cb_data) { \
};
// hipHostGetFlags[('unsigned int*', 'flagsPtr'), ('void*', 'hostPtr')]
#define INIT_hipHostGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostGetFlags.flagsPtr = flagsPtr; \
  cb_data.args.hipHostGetFlags.hostPtr = hostPtr; \
};
// hipHostMalloc[('void**', 'ptr'), ('size_t', 'size'), ('unsigned int', 'flags')]
#define INIT_hipHostMalloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostMalloc.ptr = ptr; \
  cb_data.args.hipHostMalloc.size = sizeBytes; \
  cb_data.args.hipHostMalloc.flags = flags; \
};
// hipDriverGetVersion[('int*', 'driverVersion')]
#define INIT_hipDriverGetVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDriverGetVersion.driverVersion = driverVersion; \
};
// hipMemGetInfo[('size_t*', 'free'), ('size_t*', 'total')]
#define INIT_hipMemGetInfo_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemGetInfo.free = free; \
  cb_data.args.hipMemGetInfo.total = total; \
};
// hipDeviceReset[]
#define INIT_hipDeviceReset_CB_ARGS_DATA(cb_data) { \
};
// hipMemset[('void*', 'dst'), ('int', 'value'), ('size_t', 'sizeBytes')]
#define INIT_hipMemset_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset.dst = dst; \
  cb_data.args.hipMemset.value = value; \
  cb_data.args.hipMemset.sizeBytes = sizeBytes; \
};
// hipMemsetD8[('hipDeviceptr_t', 'dest'), ('unsigned char', 'value'), ('size_t', 'count')]
#define INIT_hipMemsetD8_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD8.dest = dst; \
  cb_data.args.hipMemsetD8.value = value; \
  cb_data.args.hipMemsetD8.count = count; \
};
// hipMemcpyParam2DAsync[('const hip_Memcpy2D*', 'pCopy'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyParam2DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyParam2DAsync.pCopy = pCopy; \
  cb_data.args.hipMemcpyParam2DAsync.stream = stream; \
};
// hipHostRegister[('void*', 'hostPtr'), ('size_t', 'sizeBytes'), ('unsigned int', 'flags')]
#define INIT_hipHostRegister_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostRegister.hostPtr = hostPtr; \
  cb_data.args.hipHostRegister.sizeBytes = sizeBytes; \
  cb_data.args.hipHostRegister.flags = flags; \
};
// hipCtxSetSharedMemConfig[('hipSharedMemConfig', 'config')]
#define INIT_hipCtxSetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetSharedMemConfig.config = config; \
};
// hipArray3DCreate[('hipArray**', 'array'), ('const HIP_ARRAY3D_DESCRIPTOR*', 'pAllocateArray')]
#define INIT_hipArray3DCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipArray3DCreate.array = array; \
  cb_data.args.hipArray3DCreate.pAllocateArray = pAllocateArray; \
};
// hipIpcOpenMemHandle[('void**', 'devPtr'), ('hipIpcMemHandle_t', 'handle'), ('unsigned int', 'flags')]
#define INIT_hipIpcOpenMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcOpenMemHandle.devPtr = devPtr; \
  cb_data.args.hipIpcOpenMemHandle.handle = handle; \
  cb_data.args.hipIpcOpenMemHandle.flags = flags; \
};
// hipGetLastError[]
#define INIT_hipGetLastError_CB_ARGS_DATA(cb_data) { \
};
// hipCtxDestroy[('hipCtx_t', 'ctx')]
#define INIT_hipCtxDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxDestroy.ctx = ctx; \
};
// hipDeviceGetSharedMemConfig[('hipSharedMemConfig*', 'pConfig')]
#define INIT_hipDeviceGetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetSharedMemConfig.pConfig = pConfig; \
};
// hipMemcpy2DFromArray[('void*', 'dst'), ('size_t', 'dpitch'), ('hipArray_const_t', 'src'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpy2DFromArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DFromArray.dst = dst; \
  cb_data.args.hipMemcpy2DFromArray.dpitch = dpitch; \
  cb_data.args.hipMemcpy2DFromArray.src = src; \
  cb_data.args.hipMemcpy2DFromArray.wOffset = wOffset; \
  cb_data.args.hipMemcpy2DFromArray.hOffset = hOffset; \
  cb_data.args.hipMemcpy2DFromArray.width = width; \
  cb_data.args.hipMemcpy2DFromArray.height = height; \
  cb_data.args.hipMemcpy2DFromArray.kind = kind; \
};
// hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags[('unsigned int*', 'numBlocks'), ('hipFunction_t', 'f'), ('unsigned int', 'blockSize'), ('size_t', 'dynSharedMemPerBlk'), ('unsigned int', 'flags')]
#define INIT_hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.numBlocks = numBlocks; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.f = f; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.blockSize = blockSize; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.dynSharedMemPerBlk = dynSharedMemPerBlk; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.flags = flags; \
};
// hipSetDeviceFlags[('unsigned int', 'flags')]
#define INIT_hipSetDeviceFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetDeviceFlags.flags = flags; \
};
// hipHccModuleLaunchKernel[('hipFunction_t', 'f'), ('unsigned int', 'globalWorkSizeX'), ('unsigned int', 'globalWorkSizeY'), ('unsigned int', 'globalWorkSizeZ'), ('unsigned int', 'localWorkSizeX'), ('unsigned int', 'localWorkSizeY'), ('unsigned int', 'localWorkSizeZ'), ('size_t', 'sharedMemBytes'), ('hipStream_t', 'hStream'), ('void**', 'kernelParams'), ('void**', 'extra'), ('hipEvent_t', 'startEvent'), ('hipEvent_t', 'stopEvent')]
#define INIT_hipHccModuleLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHccModuleLaunchKernel.f = f; \
  cb_data.args.hipHccModuleLaunchKernel.globalWorkSizeX = globalWorkSizeX; \
  cb_data.args.hipHccModuleLaunchKernel.globalWorkSizeY = globalWorkSizeY; \
  cb_data.args.hipHccModuleLaunchKernel.globalWorkSizeZ = globalWorkSizeZ; \
  cb_data.args.hipHccModuleLaunchKernel.localWorkSizeX = localWorkSizeX; \
  cb_data.args.hipHccModuleLaunchKernel.localWorkSizeY = localWorkSizeY; \
  cb_data.args.hipHccModuleLaunchKernel.localWorkSizeZ = localWorkSizeZ; \
  cb_data.args.hipHccModuleLaunchKernel.sharedMemBytes = sharedMemBytes; \
  cb_data.args.hipHccModuleLaunchKernel.hStream = hStream; \
  cb_data.args.hipHccModuleLaunchKernel.kernelParams = kernelParams; \
  cb_data.args.hipHccModuleLaunchKernel.extra = extra; \
  cb_data.args.hipHccModuleLaunchKernel.startEvent = startEvent; \
  cb_data.args.hipHccModuleLaunchKernel.stopEvent = stopEvent; \
};
// hipFree[('void*', 'ptr')]
#define INIT_hipFree_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFree.ptr = ptr; \
};
// hipOccupancyMaxPotentialBlockSize[('unsigned int*', 'gridSize'), ('unsigned int*', 'blockSize'), ('hipFunction_t', 'f'), ('size_t', 'dynSharedMemPerBlk'), ('unsigned int', 'blockSizeLimit')]
#define INIT_hipOccupancyMaxPotentialBlockSize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.gridSize = gridSize; \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.blockSize = blockSize; \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.f = f; \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.dynSharedMemPerBlk = dynSharedMemPerBlk; \
  cb_data.args.hipOccupancyMaxPotentialBlockSize.blockSizeLimit = blockSizeLimit; \
};
// hipDeviceGetAttribute[('int*', 'pi'), ('hipDeviceAttribute_t', 'attr'), ('int', 'deviceId')]
#define INIT_hipDeviceGetAttribute_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetAttribute.pi = pi; \
  cb_data.args.hipDeviceGetAttribute.attr = attr; \
  cb_data.args.hipDeviceGetAttribute.deviceId = device; \
};
// hipMemcpyDtoH[('void*', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes')]
#define INIT_hipMemcpyDtoH_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoH.dst = dst; \
  cb_data.args.hipMemcpyDtoH.src = src; \
  cb_data.args.hipMemcpyDtoH.sizeBytes = sizeBytes; \
};
// hipCtxDisablePeerAccess[('hipCtx_t', 'peerCtx')]
#define INIT_hipCtxDisablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxDisablePeerAccess.peerCtx = peerCtx; \
};
// hipMallocManaged[('void**', 'devPtr'), ('size_t', 'size'), ('unsigned int', 'flags')]
#define INIT_hipMallocManaged_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocManaged.devPtr = devPtr; \
  cb_data.args.hipMallocManaged.size = size; \
  cb_data.args.hipMallocManaged.flags = flags; \
};
// hipDeviceGetByPCIBusId[('int*', 'device'), ('const char*', 'pciBusId')]
#define INIT_hipDeviceGetByPCIBusId_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetByPCIBusId.device = device; \
  cb_data.args.hipDeviceGetByPCIBusId.pciBusId = pciBusId; \
};
// hipIpcGetMemHandle[('hipIpcMemHandle_t*', 'handle'), ('void*', 'devPtr')]
#define INIT_hipIpcGetMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcGetMemHandle.handle = handle; \
  cb_data.args.hipIpcGetMemHandle.devPtr = devPtr; \
};
// hipMemcpyHtoDAsync[('hipDeviceptr_t', 'dst'), ('void*', 'src'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyHtoDAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoDAsync.dst = dst; \
  cb_data.args.hipMemcpyHtoDAsync.src = src; \
  cb_data.args.hipMemcpyHtoDAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyHtoDAsync.stream = stream; \
};
// hipCtxGetDevice[('hipDevice_t*', 'device')]
#define INIT_hipCtxGetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetDevice.device = device; \
};
// hipMemset3D[('hipPitchedPtr', 'pitchedDevPtr'), ('int', 'value'), ('hipExtent', 'extent')]
#define INIT_hipMemset3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset3D.pitchedDevPtr = pitchedDevPtr; \
  cb_data.args.hipMemset3D.value = value; \
  cb_data.args.hipMemset3D.extent = extent; \
};
// hipModuleLoadData[('hipModule_t*', 'module'), ('const void*', 'image')]
#define INIT_hipModuleLoadData_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoadData.module = module; \
  cb_data.args.hipModuleLoadData.image = image; \
};
// hipDeviceTotalMem[('size_t*', 'bytes'), ('hipDevice_t', 'device')]
#define INIT_hipDeviceTotalMem_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceTotalMem.bytes = bytes; \
  cb_data.args.hipDeviceTotalMem.device = device; \
};
// hipOccupancyMaxActiveBlocksPerMultiprocessor[('unsigned int*', 'numBlocks'), ('hipFunction_t', 'f'), ('unsigned int', 'blockSize'), ('size_t', 'dynSharedMemPerBlk')]
#define INIT_hipOccupancyMaxActiveBlocksPerMultiprocessor_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessor.numBlocks = numBlocks; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessor.f = f; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessor.blockSize = blockSize; \
  cb_data.args.hipOccupancyMaxActiveBlocksPerMultiprocessor.dynSharedMemPerBlk = dynSharedMemPerBlk; \
};
// hipCtxSetCurrent[('hipCtx_t', 'ctx')]
#define INIT_hipCtxSetCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetCurrent.ctx = ctx; \
};
// hipGetErrorString[('hipError_t', 'hipError')]
#define INIT_hipGetErrorString_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetErrorString.hipError = hip_error; \
};
// hipDevicePrimaryCtxRetain[('hipCtx_t*', 'pctx'), ('hipDevice_t', 'dev')]
#define INIT_hipDevicePrimaryCtxRetain_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxRetain.pctx = pctx; \
  cb_data.args.hipDevicePrimaryCtxRetain.dev = dev; \
};
// hipDeviceDisablePeerAccess[('int', 'peerDeviceId')]
#define INIT_hipDeviceDisablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceDisablePeerAccess.peerDeviceId = peerDeviceId; \
};
// hipStreamCreateWithFlags[('hipStream_t*', 'stream'), ('unsigned int', 'flags')]
#define INIT_hipStreamCreateWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreateWithFlags.stream = stream; \
  cb_data.args.hipStreamCreateWithFlags.flags = flags; \
};
// hipMemcpyFromArray[('void*', 'dst'), ('hipArray_const_t', 'srcArray'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('size_t', 'count'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpyFromArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyFromArray.dst = dst; \
  cb_data.args.hipMemcpyFromArray.srcArray = srcArray; \
  cb_data.args.hipMemcpyFromArray.wOffset = wOffset; \
  cb_data.args.hipMemcpyFromArray.hOffset = hOffset; \
  cb_data.args.hipMemcpyFromArray.count = count; \
  cb_data.args.hipMemcpyFromArray.kind = kind; \
};
// hipMemcpy2DAsync[('void*', 'dst'), ('size_t', 'dpitch'), ('const void*', 'src'), ('size_t', 'spitch'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpy2DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DAsync.dst = dst; \
  cb_data.args.hipMemcpy2DAsync.dpitch = dpitch; \
  cb_data.args.hipMemcpy2DAsync.src = src; \
  cb_data.args.hipMemcpy2DAsync.spitch = spitch; \
  cb_data.args.hipMemcpy2DAsync.width = width; \
  cb_data.args.hipMemcpy2DAsync.height = height; \
  cb_data.args.hipMemcpy2DAsync.kind = kind; \
  cb_data.args.hipMemcpy2DAsync.stream = stream; \
};
// hipFuncGetAttributes[('hipFuncAttributes*', 'attr'), ('const void*', 'func')]
#define INIT_hipFuncGetAttributes_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncGetAttributes.attr = attr; \
  cb_data.args.hipFuncGetAttributes.func = func; \
};
// hipGetSymbolSize[('size_t*', 'size'), ('const void*', 'symbolName')]
#define INIT_hipGetSymbolSize_CB_ARGS_DATA(cb_data) { \
};
// hipEventCreateWithFlags[('hipEvent_t*', 'event'), ('unsigned int', 'flags')]
#define INIT_hipEventCreateWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventCreateWithFlags.event = event; \
  cb_data.args.hipEventCreateWithFlags.flags = flags; \
};
// hipStreamQuery[('hipStream_t', 'stream')]
#define INIT_hipStreamQuery_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamQuery.stream = stream; \
};
// hipDeviceGetPCIBusId[('char*', 'pciBusId'), ('int', 'len'), ('int', 'device')]
#define INIT_hipDeviceGetPCIBusId_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetPCIBusId.pciBusId = pciBusId; \
  cb_data.args.hipDeviceGetPCIBusId.len = len; \
  cb_data.args.hipDeviceGetPCIBusId.device = device; \
};
// hipMemcpy[('void*', 'dst'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy.dst = dst; \
  cb_data.args.hipMemcpy.src = src; \
  cb_data.args.hipMemcpy.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpy.kind = kind; \
};
// hipPeekAtLastError[]
#define INIT_hipPeekAtLastError_CB_ARGS_DATA(cb_data) { \
};
// hipExtLaunchMultiKernelMultiDevice[('hipLaunchParams*', 'launchParamsList'), ('int', 'numDevices'), ('unsigned int', 'flags')]
#define INIT_hipExtLaunchMultiKernelMultiDevice_CB_ARGS_DATA(cb_data) { \
};
// hipStreamAddCallback[('hipStream_t', 'stream'), ('hipStreamCallback_t', 'callback'), ('void*', 'userData'), ('unsigned int', 'flags')]
#define INIT_hipStreamAddCallback_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamAddCallback.stream = stream; \
  cb_data.args.hipStreamAddCallback.callback = callback; \
  cb_data.args.hipStreamAddCallback.userData = userData; \
  cb_data.args.hipStreamAddCallback.flags = flags; \
};
// hipMemcpyToArray[('hipArray*', 'dst'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('const void*', 'src'), ('size_t', 'count'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpyToArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyToArray.dst = dst; \
  cb_data.args.hipMemcpyToArray.wOffset = wOffset; \
  cb_data.args.hipMemcpyToArray.hOffset = hOffset; \
  cb_data.args.hipMemcpyToArray.src = src; \
  cb_data.args.hipMemcpyToArray.count = count; \
  cb_data.args.hipMemcpyToArray.kind = kind; \
};
// hipMemsetD32[('hipDeviceptr_t', 'dest'), ('int', 'value'), ('size_t', 'count')]
#define INIT_hipMemsetD32_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD32.dest = dst; \
  cb_data.args.hipMemsetD32.value = value; \
  cb_data.args.hipMemsetD32.count = count; \
};
// hipDeviceSynchronize[]
#define INIT_hipDeviceSynchronize_CB_ARGS_DATA(cb_data) { \
};
// hipDeviceGetCacheConfig[('hipFuncCache_t*', 'cacheConfig')]
#define INIT_hipDeviceGetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetCacheConfig.cacheConfig = cacheConfig; \
};
// hipMalloc3D[('hipPitchedPtr*', 'pitchedDevPtr'), ('hipExtent', 'extent')]
#define INIT_hipMalloc3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc3D.pitchedDevPtr = pitchedDevPtr; \
  cb_data.args.hipMalloc3D.extent = extent; \
};
// hipPointerGetAttributes[('hipPointerAttribute_t*', 'attributes'), ('const void*', 'ptr')]
#define INIT_hipPointerGetAttributes_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipPointerGetAttributes.attributes = attributes; \
  cb_data.args.hipPointerGetAttributes.ptr = ptr; \
};
// hipMemsetAsync[('void*', 'dst'), ('int', 'value'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemsetAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetAsync.dst = dst; \
  cb_data.args.hipMemsetAsync.value = value; \
  cb_data.args.hipMemsetAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemsetAsync.stream = stream; \
};
// hipMemcpyToSymbol[('const void*', 'symbolName'), ('const void*', 'src'), ('size_t', 'sizeBytes'), ('size_t', 'offset'), ('hipMemcpyKind', 'kind')]
#define INIT_hipMemcpyToSymbol_CB_ARGS_DATA(cb_data) { \
};
// hipCtxPushCurrent[('hipCtx_t', 'ctx')]
#define INIT_hipCtxPushCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxPushCurrent.ctx = ctx; \
};
// hipMemcpyPeer[('void*', 'dst'), ('int', 'dstDeviceId'), ('const void*', 'src'), ('int', 'srcDeviceId'), ('size_t', 'sizeBytes')]
#define INIT_hipMemcpyPeer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyPeer.dst = dst; \
  cb_data.args.hipMemcpyPeer.dstDeviceId = dstDevice; \
  cb_data.args.hipMemcpyPeer.src = src; \
  cb_data.args.hipMemcpyPeer.srcDeviceId = srcDevice; \
  cb_data.args.hipMemcpyPeer.sizeBytes = sizeBytes; \
};
// hipEventSynchronize[('hipEvent_t', 'event')]
#define INIT_hipEventSynchronize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventSynchronize.event = event; \
};
// hipMemcpyDtoDAsync[('hipDeviceptr_t', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyDtoDAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoDAsync.dst = dst; \
  cb_data.args.hipMemcpyDtoDAsync.src = src; \
  cb_data.args.hipMemcpyDtoDAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyDtoDAsync.stream = stream; \
};
// hipExtMallocWithFlags[('void**', 'ptr'), ('size_t', 'sizeBytes'), ('unsigned int', 'flags')]
#define INIT_hipExtMallocWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipExtMallocWithFlags.ptr = ptr; \
  cb_data.args.hipExtMallocWithFlags.sizeBytes = sizeBytes; \
  cb_data.args.hipExtMallocWithFlags.flags = flags; \
};
// hipCtxEnablePeerAccess[('hipCtx_t', 'peerCtx'), ('unsigned int', 'flags')]
#define INIT_hipCtxEnablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxEnablePeerAccess.peerCtx = peerCtx; \
  cb_data.args.hipCtxEnablePeerAccess.flags = flags; \
};
// hipMemcpyDtoHAsync[('void*', 'dst'), ('hipDeviceptr_t', 'src'), ('size_t', 'sizeBytes'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpyDtoHAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoHAsync.dst = dst; \
  cb_data.args.hipMemcpyDtoHAsync.src = src; \
  cb_data.args.hipMemcpyDtoHAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyDtoHAsync.stream = stream; \
};
// hipModuleLaunchKernel[('hipFunction_t', 'f'), ('unsigned int', 'gridDimX'), ('unsigned int', 'gridDimY'), ('unsigned int', 'gridDimZ'), ('unsigned int', 'blockDimX'), ('unsigned int', 'blockDimY'), ('unsigned int', 'blockDimZ'), ('unsigned int', 'sharedMemBytes'), ('hipStream_t', 'stream'), ('void**', 'kernelParams'), ('void**', 'extra')]
#define INIT_hipModuleLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLaunchKernel.f = f; \
  cb_data.args.hipModuleLaunchKernel.gridDimX = gridDimX; \
  cb_data.args.hipModuleLaunchKernel.gridDimY = gridDimY; \
  cb_data.args.hipModuleLaunchKernel.gridDimZ = gridDimZ; \
  cb_data.args.hipModuleLaunchKernel.blockDimX = blockDimX; \
  cb_data.args.hipModuleLaunchKernel.blockDimY = blockDimY; \
  cb_data.args.hipModuleLaunchKernel.blockDimZ = blockDimZ; \
  cb_data.args.hipModuleLaunchKernel.sharedMemBytes = sharedMemBytes; \
  cb_data.args.hipModuleLaunchKernel.stream = hStream; \
  cb_data.args.hipModuleLaunchKernel.kernelParams = kernelParams; \
  cb_data.args.hipModuleLaunchKernel.extra = extra; \
};
// hipMemAllocPitch[('hipDeviceptr_t*', 'dptr'), ('size_t*', 'pitch'), ('size_t', 'widthInBytes'), ('size_t', 'height'), ('unsigned int', 'elementSizeBytes')]
#define INIT_hipMemAllocPitch_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemAllocPitch.dptr = dptr; \
  cb_data.args.hipMemAllocPitch.pitch = pitch; \
  cb_data.args.hipMemAllocPitch.widthInBytes = widthInBytes; \
  cb_data.args.hipMemAllocPitch.height = height; \
  cb_data.args.hipMemAllocPitch.elementSizeBytes = elementSizeBytes; \
};
// hipMemcpy2DFromArrayAsync[('void*', 'dst'), ('size_t', 'dpitch'), ('hipArray_const_t', 'src'), ('size_t', 'wOffset'), ('size_t', 'hOffset'), ('size_t', 'width'), ('size_t', 'height'), ('hipMemcpyKind', 'kind'), ('hipStream_t', 'stream')]
#define INIT_hipMemcpy2DFromArrayAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DFromArrayAsync.dst = dst; \
  cb_data.args.hipMemcpy2DFromArrayAsync.dpitch = dpitch; \
  cb_data.args.hipMemcpy2DFromArrayAsync.src = src; \
  cb_data.args.hipMemcpy2DFromArrayAsync.wOffset = wOffset; \
  cb_data.args.hipMemcpy2DFromArrayAsync.hOffset = hOffset; \
  cb_data.args.hipMemcpy2DFromArrayAsync.width = width; \
  cb_data.args.hipMemcpy2DFromArrayAsync.height = height; \
  cb_data.args.hipMemcpy2DFromArrayAsync.kind = kind; \
  cb_data.args.hipMemcpy2DFromArrayAsync.stream = stream; \
};
// hipDeviceGetLimit[('size_t*', 'pValue'), ('hipLimit_t', 'limit')]
#define INIT_hipDeviceGetLimit_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetLimit.pValue = pValue; \
  cb_data.args.hipDeviceGetLimit.limit = limit; \
};
// hipModuleLoadDataEx[('hipModule_t*', 'module'), ('const void*', 'image'), ('unsigned int', 'numOptions'), ('hipJitOption*', 'options'), ('void**', 'optionValues')]
#define INIT_hipModuleLoadDataEx_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoadDataEx.module = module; \
  cb_data.args.hipModuleLoadDataEx.image = image; \
  cb_data.args.hipModuleLoadDataEx.numOptions = numOptions; \
  cb_data.args.hipModuleLoadDataEx.options = options; \
  cb_data.args.hipModuleLoadDataEx.optionValues = optionValues; \
};
// hipRuntimeGetVersion[('int*', 'runtimeVersion')]
#define INIT_hipRuntimeGetVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipRuntimeGetVersion.runtimeVersion = runtimeVersion; \
};
// __hipPopCallConfiguration[('dim3', '*gridDim'), ('dim3', '*blockDim'), ('size_t', '*sharedMem'), ('hipStream_t', '*stream')]
#define INIT___hipPopCallConfiguration_CB_ARGS_DATA(cb_data) { \
};
// hipGetDeviceProperties[('hipDeviceProp_t*', 'prop'), ('int', 'deviceId')]
#define INIT_hipGetDeviceProperties_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDeviceProperties.prop = props; \
  cb_data.args.hipGetDeviceProperties.deviceId = device; \
};
// hipFreeArray[('hipArray*', 'array')]
#define INIT_hipFreeArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFreeArray.array = array; \
};
// hipEventElapsedTime[('float*', 'ms'), ('hipEvent_t', 'start'), ('hipEvent_t', 'stop')]
#define INIT_hipEventElapsedTime_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventElapsedTime.ms = ms; \
  cb_data.args.hipEventElapsedTime.start = start; \
  cb_data.args.hipEventElapsedTime.stop = stop; \
};
// hipDevicePrimaryCtxRelease[('hipDevice_t', 'dev')]
#define INIT_hipDevicePrimaryCtxRelease_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxRelease.dev = dev; \
};
// hipHostGetDevicePointer[('void**', 'devPtr'), ('void*', 'hstPtr'), ('unsigned int', 'flags')]
#define INIT_hipHostGetDevicePointer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostGetDevicePointer.devPtr = devicePointer; \
  cb_data.args.hipHostGetDevicePointer.hstPtr = hostPointer; \
  cb_data.args.hipHostGetDevicePointer.flags = flags; \
};
// hipMemcpyParam2D[('const hip_Memcpy2D*', 'pCopy')]
#define INIT_hipMemcpyParam2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyParam2D.pCopy = pCopy; \
};
// hipModuleGetFunction[('hipFunction_t*', 'function'), ('hipModule_t', 'module'), ('const char*', 'kname')]
#define INIT_hipModuleGetFunction_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetFunction.function = hfunc; \
  cb_data.args.hipModuleGetFunction.module = hmod; \
  cb_data.args.hipModuleGetFunction.kname = name; \
};
// hipMemsetD32Async[('hipDeviceptr_t', 'dst'), ('int', 'value'), ('size_t', 'count'), ('hipStream_t', 'stream')]
#define INIT_hipMemsetD32Async_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD32Async.dst = dst; \
  cb_data.args.hipMemsetD32Async.value = value; \
  cb_data.args.hipMemsetD32Async.count = count; \
  cb_data.args.hipMemsetD32Async.stream = stream; \
};
// hipGetDevice[('int*', 'deviceId')]
#define INIT_hipGetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDevice.deviceId = deviceId; \
};
// hipGetDeviceCount[('int*', 'count')]
#define INIT_hipGetDeviceCount_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDeviceCount.count = count; \
};
#define INIT_CB_ARGS_DATA(cb_id, cb_data) INIT_##cb_id##_CB_ARGS_DATA(cb_data)
#endif  // _HIP_PROF_STR_H

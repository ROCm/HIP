// automatically generated sources
#ifndef _HIP_PROF_STR_H
#define _HIP_PROF_STR_H
#include <sstream>
#include <string>

// Dummy API callbacks definition
#define INIT_NONE_CB_ARGS_DATA(cb_data) {};
#define INIT_hipHccGetAccelerator_CB_ARGS_DATA(cb_data) {};
#define INIT_hipHccGetAcceleratorView_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDeviceCanAccessPeer2_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyPeer2_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyPeerAsync2_CB_ARGS_DATA(cb_data) {};
#define INIT_hipCreateTextureObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDestroyTextureObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectResourceDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectResourceViewDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectTextureDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTexture_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTexture2D_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTextureToArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTextureToMipmappedArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipUnbindTexture_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetChannelDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureAlignmentOffset_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureReference_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFormat_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFlags_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFilterMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddressMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddress_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddress2D_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyHtoH_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetErrorName_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetErrorString_CB_ARGS_DATA(cb_data) {};
#define INIT_hipCreateSurfaceObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDestroySurfaceObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipStreamCreateWithPriority_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDeviceGetStreamPriorityRange_CB_ARGS_DATA(cb_data) {};
#define INIT_hipStreamGetPriority_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetSymbolAddress_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetSymbolSize_CB_ARGS_DATA(cb_data) {};

// HIP API callbacks ID enumaration
enum hip_api_id_t {
  HIP_API_ID_hipHostFree = 0,
  HIP_API_ID_hipMemcpyToSymbolAsync = 1,
  HIP_API_ID_hipMallocPitch = 2,
  HIP_API_ID_hipMalloc = 3,
  HIP_API_ID_hipDeviceGetName = 4,
  HIP_API_ID_hipEventRecord = 5,
  HIP_API_ID_hipCtxSynchronize = 6,
  HIP_API_ID_hipSetDevice = 7,
  HIP_API_ID_hipSetupArgument = 8,
  HIP_API_ID_hipMemcpyFromSymbolAsync = 9,
  HIP_API_ID_hipMemcpyDtoD = 10,
  HIP_API_ID_hipMemcpy2DToArray = 11,
  HIP_API_ID_hipCtxGetCacheConfig = 12,
  HIP_API_ID_hipStreamWaitEvent = 13,
  HIP_API_ID_hipModuleLoad = 14,
  HIP_API_ID_hipDevicePrimaryCtxSetFlags = 15,
  HIP_API_ID_hipMemcpyAsync = 16,
  HIP_API_ID_hipMalloc3DArray = 17,
  HIP_API_ID_hipStreamCreate = 18,
  HIP_API_ID_hipCtxGetCurrent = 19,
  HIP_API_ID_hipDevicePrimaryCtxGetState = 20,
  HIP_API_ID_hipEventQuery = 21,
  HIP_API_ID_hipEventCreate = 22,
  HIP_API_ID_hipMemGetAddressRange = 23,
  HIP_API_ID_hipMemcpyFromSymbol = 24,
  HIP_API_ID_hipArrayCreate = 25,
  HIP_API_ID_hipStreamGetFlags = 26,
  HIP_API_ID_hipMallocArray = 27,
  HIP_API_ID_hipCtxGetSharedMemConfig = 28,
  HIP_API_ID_hipMemPtrGetInfo = 29,
  HIP_API_ID_hipCtxGetFlags = 30,
  HIP_API_ID_hipStreamDestroy = 31,
  HIP_API_ID_hipMemset3DAsync = 32,
  HIP_API_ID_hipMemcpy3D = 33,
  HIP_API_ID_hipInit = 34,
  HIP_API_ID_hipMemcpyAtoH = 35,
  HIP_API_ID_hipMemset2D = 36,
  HIP_API_ID_hipMemset2DAsync = 37,
  HIP_API_ID_hipDeviceCanAccessPeer = 38,
  HIP_API_ID_hipDeviceEnablePeerAccess = 39,
  HIP_API_ID_hipModuleUnload = 40,
  HIP_API_ID_hipHostUnregister = 41,
  HIP_API_ID_hipProfilerStop = 42,
  HIP_API_ID_hipLaunchByPtr = 43,
  HIP_API_ID_hipStreamSynchronize = 44,
  HIP_API_ID_hipFreeHost = 45,
  HIP_API_ID_hipRemoveApiCallback = 46,
  HIP_API_ID_hipDeviceSetCacheConfig = 47,
  HIP_API_ID_hipCtxGetApiVersion = 48,
  HIP_API_ID_hipMemcpyHtoD = 49,
  HIP_API_ID_hipModuleGetGlobal = 50,
  HIP_API_ID_hipMemcpyHtoA = 51,
  HIP_API_ID_hipCtxCreate = 52,
  HIP_API_ID_hipMemcpy2D = 53,
  HIP_API_ID_hipIpcCloseMemHandle = 54,
  HIP_API_ID_hipChooseDevice = 55,
  HIP_API_ID_hipDeviceSetSharedMemConfig = 56,
  HIP_API_ID_hipDeviceComputeCapability = 57,
  HIP_API_ID_hipRegisterApiCallback = 58,
  HIP_API_ID_hipDeviceGet = 59,
  HIP_API_ID_hipProfilerStart = 60,
  HIP_API_ID_hipCtxSetCacheConfig = 61,
  HIP_API_ID_hipFuncSetCacheConfig = 62,
  HIP_API_ID_hipMemcpyPeerAsync = 63,
  HIP_API_ID_hipEventElapsedTime = 64,
  HIP_API_ID_hipDevicePrimaryCtxReset = 65,
  HIP_API_ID_hipEventDestroy = 66,
  HIP_API_ID_hipCtxPopCurrent = 67,
  HIP_API_ID_hipHostGetFlags = 68,
  HIP_API_ID_hipHostMalloc = 69,
  HIP_API_ID_hipDriverGetVersion = 70,
  HIP_API_ID_hipMemGetInfo = 71,
  HIP_API_ID_hipDeviceReset = 72,
  HIP_API_ID_hipMemset = 73,
  HIP_API_ID_hipMemsetD8 = 74,
  HIP_API_ID_hipHostRegister = 75,
  HIP_API_ID_hipCtxSetSharedMemConfig = 76,
  HIP_API_ID_hipArray3DCreate = 77,
  HIP_API_ID_hipIpcOpenMemHandle = 78,
  HIP_API_ID_hipGetLastError = 79,
  HIP_API_ID_hipCtxDestroy = 80,
  HIP_API_ID_hipDeviceGetSharedMemConfig = 81,
  HIP_API_ID_hipRegisterActivityCallback = 82,
  HIP_API_ID_hipSetDeviceFlags = 83,
  HIP_API_ID_hipFree = 84,
  HIP_API_ID_hipDeviceGetAttribute = 85,
  HIP_API_ID_hipMemcpyDtoH = 86,
  HIP_API_ID_hipCtxDisablePeerAccess = 87,
  HIP_API_ID_hipDeviceGetByPCIBusId = 88,
  HIP_API_ID_hipIpcGetMemHandle = 89,
  HIP_API_ID_hipMemcpyHtoDAsync = 90,
  HIP_API_ID_hipCtxGetDevice = 91,
  HIP_API_ID_hipMemset3D = 92,
  HIP_API_ID_hipModuleLoadData = 93,
  HIP_API_ID_hipDeviceTotalMem = 94,
  HIP_API_ID_hipCtxSetCurrent = 95,
  HIP_API_ID_hipMallocHost = 96,
  HIP_API_ID_hipDevicePrimaryCtxRetain = 97,
  HIP_API_ID_hipDeviceDisablePeerAccess = 98,
  HIP_API_ID_hipStreamCreateWithFlags = 99,
  HIP_API_ID_hipMemcpyFromArray = 100,
  HIP_API_ID_hipMemcpy2DAsync = 101,
  HIP_API_ID_hipFuncGetAttributes = 102,
  HIP_API_ID_hipEventCreateWithFlags = 103,
  HIP_API_ID_hipStreamQuery = 104,
  HIP_API_ID_hipDeviceGetPCIBusId = 105,
  HIP_API_ID_hipMemcpy = 106,
  HIP_API_ID_hipPeekAtLastError = 107,
  HIP_API_ID_hipHostAlloc = 108,
  HIP_API_ID_hipStreamAddCallback = 109,
  HIP_API_ID_hipMemcpyToArray = 110,
  HIP_API_ID_hipDeviceSynchronize = 111,
  HIP_API_ID_hipDeviceGetCacheConfig = 112,
  HIP_API_ID_hipMalloc3D = 113,
  HIP_API_ID_hipPointerGetAttributes = 114,
  HIP_API_ID_hipMemsetAsync = 115,
  HIP_API_ID_hipMemcpyToSymbol = 116,
  HIP_API_ID_hipCtxPushCurrent = 117,
  HIP_API_ID_hipMemcpyPeer = 118,
  HIP_API_ID_hipEventSynchronize = 119,
  HIP_API_ID_hipMemcpyDtoDAsync = 120,
  HIP_API_ID_hipCtxEnablePeerAccess = 121,
  HIP_API_ID_hipMemcpyDtoHAsync = 122,
  HIP_API_ID_hipModuleLaunchKernel = 123,
  HIP_API_ID_hipModuleGetTexRef = 124,
  HIP_API_ID_hipRemoveActivityCallback = 125,
  HIP_API_ID_hipDeviceGetLimit = 126,
  HIP_API_ID_hipModuleLoadDataEx = 127,
  HIP_API_ID_hipRuntimeGetVersion = 128,
  HIP_API_ID_hipGetDeviceProperties = 129,
  HIP_API_ID_hipFreeArray = 130,
  HIP_API_ID_hipDevicePrimaryCtxRelease = 131,
  HIP_API_ID_hipHostGetDevicePointer = 132,
  HIP_API_ID_hipMemcpyParam2D = 133,
  HIP_API_ID_hipConfigureCall = 134,
  HIP_API_ID_hipModuleGetFunction = 135,
  HIP_API_ID_hipGetDevice = 136,
  HIP_API_ID_hipGetDeviceCount = 137,
  HIP_API_ID_hipHccModuleLaunchKernel = 138,
  HIP_API_ID_NUMBER = 139,
  HIP_API_ID_ANY = 140,

  HIP_API_ID_NONE = HIP_API_ID_NUMBER,
  HIP_API_ID_hipHccGetAccelerator = HIP_API_ID_NUMBER,
  HIP_API_ID_hipHccGetAcceleratorView = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDeviceCanAccessPeer2 = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyPeer2 = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyPeerAsync2 = HIP_API_ID_NUMBER,
  HIP_API_ID_hipCreateTextureObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDestroyTextureObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectResourceDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectResourceViewDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectTextureDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTexture = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTexture2D = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTextureToArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTextureToMipmappedArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipUnbindTexture = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetChannelDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureAlignmentOffset = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureReference = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFormat = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFlags = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFilterMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddressMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddress = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddress2D = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyHtoH = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetErrorName = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetErrorString = HIP_API_ID_NUMBER,
  HIP_API_ID_hipCreateSurfaceObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDestroySurfaceObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipStreamCreateWithPriority = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDeviceGetStreamPriorityRange = HIP_API_ID_NUMBER,
  HIP_API_ID_hipStreamGetPriority = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetSymbolAddress = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetSymbolSize = HIP_API_ID_NUMBER,
};

// Return HIP API string
static const char* hip_api_name(const uint32_t& id) {
  switch(id) {
    case HIP_API_ID_hipHostFree: return "hipHostFree";
    case HIP_API_ID_hipMemcpyToSymbolAsync: return "hipMemcpyToSymbolAsync";
    case HIP_API_ID_hipMallocPitch: return "hipMallocPitch";
    case HIP_API_ID_hipMalloc: return "hipMalloc";
    case HIP_API_ID_hipDeviceGetName: return "hipDeviceGetName";
    case HIP_API_ID_hipEventRecord: return "hipEventRecord";
    case HIP_API_ID_hipCtxSynchronize: return "hipCtxSynchronize";
    case HIP_API_ID_hipSetDevice: return "hipSetDevice";
    case HIP_API_ID_hipSetupArgument: return "hipSetupArgument";
    case HIP_API_ID_hipMemcpyFromSymbolAsync: return "hipMemcpyFromSymbolAsync";
    case HIP_API_ID_hipMemcpyDtoD: return "hipMemcpyDtoD";
    case HIP_API_ID_hipMemcpy2DToArray: return "hipMemcpy2DToArray";
    case HIP_API_ID_hipCtxGetCacheConfig: return "hipCtxGetCacheConfig";
    case HIP_API_ID_hipStreamWaitEvent: return "hipStreamWaitEvent";
    case HIP_API_ID_hipModuleLoad: return "hipModuleLoad";
    case HIP_API_ID_hipDevicePrimaryCtxSetFlags: return "hipDevicePrimaryCtxSetFlags";
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
    case HIP_API_ID_hipCtxGetFlags: return "hipCtxGetFlags";
    case HIP_API_ID_hipStreamDestroy: return "hipStreamDestroy";
    case HIP_API_ID_hipMemset3DAsync: return "hipMemset3DAsync";
    case HIP_API_ID_hipMemcpy3D: return "hipMemcpy3D";
    case HIP_API_ID_hipInit: return "hipInit";
    case HIP_API_ID_hipMemcpyAtoH: return "hipMemcpyAtoH";
    case HIP_API_ID_hipMemset2D: return "hipMemset2D";
    case HIP_API_ID_hipMemset2DAsync: return "hipMemset2DAsync";
    case HIP_API_ID_hipDeviceCanAccessPeer: return "hipDeviceCanAccessPeer";
    case HIP_API_ID_hipDeviceEnablePeerAccess: return "hipDeviceEnablePeerAccess";
    case HIP_API_ID_hipModuleUnload: return "hipModuleUnload";
    case HIP_API_ID_hipHostUnregister: return "hipHostUnregister";
    case HIP_API_ID_hipProfilerStop: return "hipProfilerStop";
    case HIP_API_ID_hipLaunchByPtr: return "hipLaunchByPtr";
    case HIP_API_ID_hipStreamSynchronize: return "hipStreamSynchronize";
    case HIP_API_ID_hipFreeHost: return "hipFreeHost";
    case HIP_API_ID_hipRemoveApiCallback: return "hipRemoveApiCallback";
    case HIP_API_ID_hipDeviceSetCacheConfig: return "hipDeviceSetCacheConfig";
    case HIP_API_ID_hipCtxGetApiVersion: return "hipCtxGetApiVersion";
    case HIP_API_ID_hipMemcpyHtoD: return "hipMemcpyHtoD";
    case HIP_API_ID_hipModuleGetGlobal: return "hipModuleGetGlobal";
    case HIP_API_ID_hipMemcpyHtoA: return "hipMemcpyHtoA";
    case HIP_API_ID_hipCtxCreate: return "hipCtxCreate";
    case HIP_API_ID_hipMemcpy2D: return "hipMemcpy2D";
    case HIP_API_ID_hipIpcCloseMemHandle: return "hipIpcCloseMemHandle";
    case HIP_API_ID_hipChooseDevice: return "hipChooseDevice";
    case HIP_API_ID_hipDeviceSetSharedMemConfig: return "hipDeviceSetSharedMemConfig";
    case HIP_API_ID_hipDeviceComputeCapability: return "hipDeviceComputeCapability";
    case HIP_API_ID_hipRegisterApiCallback: return "hipRegisterApiCallback";
    case HIP_API_ID_hipDeviceGet: return "hipDeviceGet";
    case HIP_API_ID_hipProfilerStart: return "hipProfilerStart";
    case HIP_API_ID_hipCtxSetCacheConfig: return "hipCtxSetCacheConfig";
    case HIP_API_ID_hipFuncSetCacheConfig: return "hipFuncSetCacheConfig";
    case HIP_API_ID_hipMemcpyPeerAsync: return "hipMemcpyPeerAsync";
    case HIP_API_ID_hipEventElapsedTime: return "hipEventElapsedTime";
    case HIP_API_ID_hipDevicePrimaryCtxReset: return "hipDevicePrimaryCtxReset";
    case HIP_API_ID_hipEventDestroy: return "hipEventDestroy";
    case HIP_API_ID_hipCtxPopCurrent: return "hipCtxPopCurrent";
    case HIP_API_ID_hipHostGetFlags: return "hipHostGetFlags";
    case HIP_API_ID_hipHostMalloc: return "hipHostMalloc";
    case HIP_API_ID_hipDriverGetVersion: return "hipDriverGetVersion";
    case HIP_API_ID_hipMemGetInfo: return "hipMemGetInfo";
    case HIP_API_ID_hipDeviceReset: return "hipDeviceReset";
    case HIP_API_ID_hipMemset: return "hipMemset";
    case HIP_API_ID_hipMemsetD8: return "hipMemsetD8";
    case HIP_API_ID_hipHostRegister: return "hipHostRegister";
    case HIP_API_ID_hipCtxSetSharedMemConfig: return "hipCtxSetSharedMemConfig";
    case HIP_API_ID_hipArray3DCreate: return "hipArray3DCreate";
    case HIP_API_ID_hipIpcOpenMemHandle: return "hipIpcOpenMemHandle";
    case HIP_API_ID_hipGetLastError: return "hipGetLastError";
    case HIP_API_ID_hipCtxDestroy: return "hipCtxDestroy";
    case HIP_API_ID_hipDeviceGetSharedMemConfig: return "hipDeviceGetSharedMemConfig";
    case HIP_API_ID_hipRegisterActivityCallback: return "hipRegisterActivityCallback";
    case HIP_API_ID_hipSetDeviceFlags: return "hipSetDeviceFlags";
    case HIP_API_ID_hipFree: return "hipFree";
    case HIP_API_ID_hipDeviceGetAttribute: return "hipDeviceGetAttribute";
    case HIP_API_ID_hipMemcpyDtoH: return "hipMemcpyDtoH";
    case HIP_API_ID_hipCtxDisablePeerAccess: return "hipCtxDisablePeerAccess";
    case HIP_API_ID_hipDeviceGetByPCIBusId: return "hipDeviceGetByPCIBusId";
    case HIP_API_ID_hipIpcGetMemHandle: return "hipIpcGetMemHandle";
    case HIP_API_ID_hipMemcpyHtoDAsync: return "hipMemcpyHtoDAsync";
    case HIP_API_ID_hipCtxGetDevice: return "hipCtxGetDevice";
    case HIP_API_ID_hipMemset3D: return "hipMemset3D";
    case HIP_API_ID_hipModuleLoadData: return "hipModuleLoadData";
    case HIP_API_ID_hipDeviceTotalMem: return "hipDeviceTotalMem";
    case HIP_API_ID_hipCtxSetCurrent: return "hipCtxSetCurrent";
    case HIP_API_ID_hipMallocHost: return "hipMallocHost";
    case HIP_API_ID_hipDevicePrimaryCtxRetain: return "hipDevicePrimaryCtxRetain";
    case HIP_API_ID_hipDeviceDisablePeerAccess: return "hipDeviceDisablePeerAccess";
    case HIP_API_ID_hipStreamCreateWithFlags: return "hipStreamCreateWithFlags";
    case HIP_API_ID_hipMemcpyFromArray: return "hipMemcpyFromArray";
    case HIP_API_ID_hipMemcpy2DAsync: return "hipMemcpy2DAsync";
    case HIP_API_ID_hipFuncGetAttributes: return "hipFuncGetAttributes";
    case HIP_API_ID_hipEventCreateWithFlags: return "hipEventCreateWithFlags";
    case HIP_API_ID_hipStreamQuery: return "hipStreamQuery";
    case HIP_API_ID_hipDeviceGetPCIBusId: return "hipDeviceGetPCIBusId";
    case HIP_API_ID_hipMemcpy: return "hipMemcpy";
    case HIP_API_ID_hipPeekAtLastError: return "hipPeekAtLastError";
    case HIP_API_ID_hipHostAlloc: return "hipHostAlloc";
    case HIP_API_ID_hipStreamAddCallback: return "hipStreamAddCallback";
    case HIP_API_ID_hipMemcpyToArray: return "hipMemcpyToArray";
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
    case HIP_API_ID_hipCtxEnablePeerAccess: return "hipCtxEnablePeerAccess";
    case HIP_API_ID_hipMemcpyDtoHAsync: return "hipMemcpyDtoHAsync";
    case HIP_API_ID_hipModuleLaunchKernel: return "hipModuleLaunchKernel";
    case HIP_API_ID_hipModuleGetTexRef: return "hipModuleGetTexRef";
    case HIP_API_ID_hipRemoveActivityCallback: return "hipRemoveActivityCallback";
    case HIP_API_ID_hipDeviceGetLimit: return "hipDeviceGetLimit";
    case HIP_API_ID_hipModuleLoadDataEx: return "hipModuleLoadDataEx";
    case HIP_API_ID_hipRuntimeGetVersion: return "hipRuntimeGetVersion";
    case HIP_API_ID_hipGetDeviceProperties: return "hipGetDeviceProperties";
    case HIP_API_ID_hipFreeArray: return "hipFreeArray";
    case HIP_API_ID_hipDevicePrimaryCtxRelease: return "hipDevicePrimaryCtxRelease";
    case HIP_API_ID_hipHostGetDevicePointer: return "hipHostGetDevicePointer";
    case HIP_API_ID_hipMemcpyParam2D: return "hipMemcpyParam2D";
    case HIP_API_ID_hipConfigureCall: return "hipConfigureCall";
    case HIP_API_ID_hipModuleGetFunction: return "hipModuleGetFunction";
    case HIP_API_ID_hipGetDevice: return "hipGetDevice";
    case HIP_API_ID_hipGetDeviceCount: return "hipGetDeviceCount";
  };
  return "unknown";
};

// HIP API callbacks data structure
struct hip_api_data_t {
  uint64_t correlation_id;
  uint32_t phase;
  union {
    struct {
      void* ptr;
    } hipHostFree;
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
      hipDeviceptr_t dst;
      hipDeviceptr_t src;
      size_t sizeBytes;
    } hipMemcpyDtoD;
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
      hipFuncCache_t* cacheConfig;
    } hipCtxGetCacheConfig;
    struct {
      hipStream_t stream;
      hipEvent_t event;
      unsigned int flags;
    } hipStreamWaitEvent;
    struct {
      hipModule_t* module;
      const char* fname;
    } hipModuleLoad;
    struct {
      hipDevice_t dev;
      unsigned int flags;
    } hipDevicePrimaryCtxSetFlags;
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
      unsigned int* flags;
    } hipCtxGetFlags;
    struct {
      hipStream_t stream;
    } hipStreamDestroy;
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
      void* ptr;
    } hipFreeHost;
    struct {
      uint32_t id;
    } hipRemoveApiCallback;
    struct {
      hipFuncCache_t cacheConfig;
    } hipDeviceSetCacheConfig;
    struct {
      hipCtx_t ctx;
      int* apiVersion;
    } hipCtxGetApiVersion;
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
      uint32_t id;
      void* fun;
      void* arg;
    } hipRegisterApiCallback;
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
      void* dst;
      int dstDeviceId;
      const void* src;
      int srcDevice;
      size_t sizeBytes;
      hipStream_t stream;
    } hipMemcpyPeerAsync;
    struct {
      float* ms;
      hipEvent_t start;
      hipEvent_t stop;
    } hipEventElapsedTime;
    struct {
      hipDevice_t dev;
    } hipDevicePrimaryCtxReset;
    struct {
      hipEvent_t event;
    } hipEventDestroy;
    struct {
      hipCtx_t* ctx;
    } hipCtxPopCurrent;
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
      size_t sizeBytes;
    } hipMemsetD8;
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
      const HIP_ARRAY_DESCRIPTOR* pAllocateArray;
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
      uint32_t id;
      void* fun;
      void* arg;
    } hipRegisterActivityCallback;
    struct {
      unsigned flags;
    } hipSetDeviceFlags;
    struct {
      void* ptr;
    } hipFree;
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
      hipCtx_t ctx;
    } hipCtxSetCurrent;
    struct {
      void** ptr;
      size_t size;
    } hipMallocHost;
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
      hipEvent_t* event;
      unsigned flags;
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
      void** ptr;
      size_t size;
      unsigned int flags;
    } hipHostAlloc;
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
      hipFunction_t f;
    } hipHccModuleLaunchKernel;
    struct {
      textureReference** texRef;
      hipModule_t hmod;
      const char* name;
    } hipModuleGetTexRef;
    struct {
      uint32_t id;
    } hipRemoveActivityCallback;
    struct {
      size_t* pValue;
      hipLimit_t limit;
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
      hipDeviceProp_t* prop;
      int deviceId;
    } hipGetDeviceProperties;
    struct {
      hipArray* array;
    } hipFreeArray;
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
      dim3 gridDim;
      dim3 blockDim;
      size_t sharedMem;
      hipStream_t stream;
    } hipConfigureCall;
    struct {
      hipFunction_t* function;
      hipModule_t module;
      const char* kname;
    } hipModuleGetFunction;
    struct {
      int* deviceId;
    } hipGetDevice;
    struct {
      int* count;
    } hipGetDeviceCount;
  } args;
};

// HIP API callbacks args data filling macros
#define INIT_hipHostFree_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostFree.ptr = (void*)ptr; \
};
#define INIT_hipMemcpyToSymbolAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyToSymbolAsync.symbolName = (const void*)symbolName; \
  cb_data.args.hipMemcpyToSymbolAsync.src = (const void*)src; \
  cb_data.args.hipMemcpyToSymbolAsync.sizeBytes = (size_t)count; \
  cb_data.args.hipMemcpyToSymbolAsync.offset = (size_t)offset; \
  cb_data.args.hipMemcpyToSymbolAsync.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpyToSymbolAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipMallocPitch_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocPitch.ptr = (void**)ptr; \
  cb_data.args.hipMallocPitch.pitch = (size_t*)pitch; \
  cb_data.args.hipMallocPitch.width = (size_t)width; \
  cb_data.args.hipMallocPitch.height = (size_t)height; \
};
#define INIT_hipMalloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc.ptr = (void**)ptr; \
  cb_data.args.hipMalloc.size = (size_t)sizeBytes; \
};
#define INIT_hipDeviceGetName_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetName.name = (char*)name; \
  cb_data.args.hipDeviceGetName.len = (int)len; \
  cb_data.args.hipDeviceGetName.device = (hipDevice_t)device; \
};
#define INIT_hipEventRecord_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventRecord.event = (hipEvent_t)event; \
  cb_data.args.hipEventRecord.stream = (hipStream_t)stream; \
};
#define INIT_hipCtxSynchronize_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipSetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetDevice.deviceId = (int)deviceId; \
};
#define INIT_hipSetupArgument_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetupArgument.arg = (const void*)arg; \
  cb_data.args.hipSetupArgument.size = (size_t)size; \
  cb_data.args.hipSetupArgument.offset = (size_t)offset; \
};
#define INIT_hipMemcpyFromSymbolAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyFromSymbolAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpyFromSymbolAsync.symbolName = (const void*)symbolName; \
  cb_data.args.hipMemcpyFromSymbolAsync.sizeBytes = (size_t)count; \
  cb_data.args.hipMemcpyFromSymbolAsync.offset = (size_t)offset; \
  cb_data.args.hipMemcpyFromSymbolAsync.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpyFromSymbolAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipMemcpyDtoD_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoD.dst = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemcpyDtoD.src = (hipDeviceptr_t)src; \
  cb_data.args.hipMemcpyDtoD.sizeBytes = (size_t)sizeBytes; \
};
#define INIT_hipMemcpy2DToArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DToArray.dst = (hipArray*)dst; \
  cb_data.args.hipMemcpy2DToArray.wOffset = (size_t)wOffset; \
  cb_data.args.hipMemcpy2DToArray.hOffset = (size_t)hOffset; \
  cb_data.args.hipMemcpy2DToArray.src = (const void*)src; \
  cb_data.args.hipMemcpy2DToArray.spitch = (size_t)spitch; \
  cb_data.args.hipMemcpy2DToArray.width = (size_t)width; \
  cb_data.args.hipMemcpy2DToArray.height = (size_t)height; \
  cb_data.args.hipMemcpy2DToArray.kind = (hipMemcpyKind)kind; \
};
#define INIT_hipCtxGetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetCacheConfig.cacheConfig = (hipFuncCache_t*)cacheConfig; \
};
#define INIT_hipStreamWaitEvent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamWaitEvent.stream = (hipStream_t)stream; \
  cb_data.args.hipStreamWaitEvent.event = (hipEvent_t)event; \
  cb_data.args.hipStreamWaitEvent.flags = (unsigned int)flags; \
};
#define INIT_hipModuleLoad_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoad.module = (hipModule_t*)module; \
  cb_data.args.hipModuleLoad.fname = (const char*)fname; \
};
#define INIT_hipDevicePrimaryCtxSetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxSetFlags.dev = (hipDevice_t)dev; \
  cb_data.args.hipDevicePrimaryCtxSetFlags.flags = (unsigned int)flags; \
};
#define INIT_hipMemcpyAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpyAsync.src = (const void*)src; \
  cb_data.args.hipMemcpyAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyAsync.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpyAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipMalloc3DArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc3DArray.array = (hipArray**)array; \
  cb_data.args.hipMalloc3DArray.desc = (const hipChannelFormatDesc*)desc; \
  cb_data.args.hipMalloc3DArray.extent = (hipExtent)extent; \
  cb_data.args.hipMalloc3DArray.flags = (unsigned int)flags; \
};
#define INIT_hipStreamCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreate.stream = (hipStream_t*)stream; \
};
#define INIT_hipCtxGetCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetCurrent.ctx = (hipCtx_t*)ctx; \
};
#define INIT_hipDevicePrimaryCtxGetState_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxGetState.dev = (hipDevice_t)dev; \
  cb_data.args.hipDevicePrimaryCtxGetState.flags = (unsigned int*)flags; \
  cb_data.args.hipDevicePrimaryCtxGetState.active = (int*)active; \
};
#define INIT_hipEventQuery_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventQuery.event = (hipEvent_t)event; \
};
#define INIT_hipEventCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventCreate.event = (hipEvent_t*)event; \
};
#define INIT_hipMemGetAddressRange_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemGetAddressRange.pbase = (hipDeviceptr_t*)pbase; \
  cb_data.args.hipMemGetAddressRange.psize = (size_t*)psize; \
  cb_data.args.hipMemGetAddressRange.dptr = (hipDeviceptr_t)dptr; \
};
#define INIT_hipMemcpyFromSymbol_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyFromSymbol.dst = (void*)dst; \
  cb_data.args.hipMemcpyFromSymbol.symbolName = (const void*)symbolName; \
  cb_data.args.hipMemcpyFromSymbol.sizeBytes = (size_t)count; \
  cb_data.args.hipMemcpyFromSymbol.offset = (size_t)offset; \
  cb_data.args.hipMemcpyFromSymbol.kind = (hipMemcpyKind)kind; \
};
#define INIT_hipArrayCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipArrayCreate.pHandle = (hipArray**)array; \
  cb_data.args.hipArrayCreate.pAllocateArray = (const HIP_ARRAY_DESCRIPTOR*)pAllocateArray; \
};
#define INIT_hipStreamGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamGetFlags.stream = (hipStream_t)stream; \
  cb_data.args.hipStreamGetFlags.flags = (unsigned int*)flags; \
};
#define INIT_hipMallocArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocArray.array = (hipArray**)array; \
  cb_data.args.hipMallocArray.desc = (const hipChannelFormatDesc*)desc; \
  cb_data.args.hipMallocArray.width = (size_t)width; \
  cb_data.args.hipMallocArray.height = (size_t)height; \
  cb_data.args.hipMallocArray.flags = (unsigned int)flags; \
};
#define INIT_hipCtxGetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetSharedMemConfig.pConfig = (hipSharedMemConfig*)pConfig; \
};
#define INIT_hipMemPtrGetInfo_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemPtrGetInfo.ptr = (void*)ptr; \
  cb_data.args.hipMemPtrGetInfo.size = (size_t*)size; \
};
#define INIT_hipCtxGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetFlags.flags = (unsigned int*)flags; \
};
#define INIT_hipStreamDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamDestroy.stream = (hipStream_t)stream; \
};
#define INIT_hipMemset3DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset3DAsync.pitchedDevPtr = (hipPitchedPtr)pitchedDevPtr; \
  cb_data.args.hipMemset3DAsync.value = (int)value; \
  cb_data.args.hipMemset3DAsync.extent = (hipExtent)extent; \
  cb_data.args.hipMemset3DAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipMemcpy3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy3D.p = (const hipMemcpy3DParms*)p; \
};
#define INIT_hipInit_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipInit.flags = (unsigned int)flags; \
};
#define INIT_hipMemcpyAtoH_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyAtoH.dst = (void*)dst; \
  cb_data.args.hipMemcpyAtoH.srcArray = (hipArray*)srcArray; \
  cb_data.args.hipMemcpyAtoH.srcOffset = (size_t)srcOffset; \
  cb_data.args.hipMemcpyAtoH.count = (size_t)count; \
};
#define INIT_hipMemset2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset2D.dst = (void*)dst; \
  cb_data.args.hipMemset2D.pitch = (size_t)pitch; \
  cb_data.args.hipMemset2D.value = (int)value; \
  cb_data.args.hipMemset2D.width = (size_t)width; \
  cb_data.args.hipMemset2D.height = (size_t)height; \
};
#define INIT_hipMemset2DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset2DAsync.dst = (void*)dst; \
  cb_data.args.hipMemset2DAsync.pitch = (size_t)pitch; \
  cb_data.args.hipMemset2DAsync.value = (int)value; \
  cb_data.args.hipMemset2DAsync.width = (size_t)width; \
  cb_data.args.hipMemset2DAsync.height = (size_t)height; \
  cb_data.args.hipMemset2DAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipDeviceCanAccessPeer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceCanAccessPeer.canAccessPeer = (int*)canAccessPeer; \
  cb_data.args.hipDeviceCanAccessPeer.deviceId = (int)deviceId; \
  cb_data.args.hipDeviceCanAccessPeer.peerDeviceId = (int)peerDeviceId; \
};
#define INIT_hipDeviceEnablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceEnablePeerAccess.peerDeviceId = (int)peerDeviceId; \
  cb_data.args.hipDeviceEnablePeerAccess.flags = (unsigned int)flags; \
};
#define INIT_hipModuleUnload_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleUnload.module = (hipModule_t)hmod; \
};
#define INIT_hipHostUnregister_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostUnregister.hostPtr = (void*)hostPtr; \
};
#define INIT_hipProfilerStop_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipLaunchByPtr_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipLaunchByPtr.func = (const void*)hostFunction; \
};
#define INIT_hipStreamSynchronize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamSynchronize.stream = (hipStream_t)stream; \
};
#define INIT_hipFreeHost_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFreeHost.ptr = (void*)ptr; \
};
#define INIT_hipRemoveApiCallback_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipRemoveApiCallback.id = (uint32_t)id; \
};
#define INIT_hipDeviceSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceSetCacheConfig.cacheConfig = (hipFuncCache_t)cacheConfig; \
};
#define INIT_hipCtxGetApiVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetApiVersion.ctx = (hipCtx_t)ctx; \
  cb_data.args.hipCtxGetApiVersion.apiVersion = (int*)apiVersion; \
};
#define INIT_hipMemcpyHtoD_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoD.dst = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemcpyHtoD.src = (void*)src; \
  cb_data.args.hipMemcpyHtoD.sizeBytes = (size_t)sizeBytes; \
};
#define INIT_hipModuleGetGlobal_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetGlobal.dptr = (hipDeviceptr_t*)dptr; \
  cb_data.args.hipModuleGetGlobal.bytes = (size_t*)bytes; \
  cb_data.args.hipModuleGetGlobal.hmod = (hipModule_t)hmod; \
  cb_data.args.hipModuleGetGlobal.name = (const char*)name; \
};
#define INIT_hipMemcpyHtoA_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoA.dstArray = (hipArray*)dstArray; \
  cb_data.args.hipMemcpyHtoA.dstOffset = (size_t)dstOffset; \
  cb_data.args.hipMemcpyHtoA.srcHost = (const void*)srcHost; \
  cb_data.args.hipMemcpyHtoA.count = (size_t)count; \
};
#define INIT_hipCtxCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxCreate.ctx = (hipCtx_t*)ctx; \
  cb_data.args.hipCtxCreate.flags = (unsigned int)flags; \
  cb_data.args.hipCtxCreate.device = (hipDevice_t)device; \
};
#define INIT_hipMemcpy2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2D.dst = (void*)dst; \
  cb_data.args.hipMemcpy2D.dpitch = (size_t)dpitch; \
  cb_data.args.hipMemcpy2D.src = (const void*)src; \
  cb_data.args.hipMemcpy2D.spitch = (size_t)spitch; \
  cb_data.args.hipMemcpy2D.width = (size_t)width; \
  cb_data.args.hipMemcpy2D.height = (size_t)height; \
  cb_data.args.hipMemcpy2D.kind = (hipMemcpyKind)kind; \
};
#define INIT_hipIpcCloseMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcCloseMemHandle.devPtr = (void*)devPtr; \
};
#define INIT_hipChooseDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipChooseDevice.device = (int*)device; \
  cb_data.args.hipChooseDevice.prop = (const hipDeviceProp_t*)prop; \
};
#define INIT_hipDeviceSetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceSetSharedMemConfig.config = (hipSharedMemConfig)config; \
};
#define INIT_hipDeviceComputeCapability_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceComputeCapability.major = (int*)major; \
  cb_data.args.hipDeviceComputeCapability.minor = (int*)minor; \
  cb_data.args.hipDeviceComputeCapability.device = (hipDevice_t)device; \
};
#define INIT_hipRegisterApiCallback_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipRegisterApiCallback.id = (uint32_t)id; \
  cb_data.args.hipRegisterApiCallback.fun = (void*)fun; \
  cb_data.args.hipRegisterApiCallback.arg = (void*)arg; \
};
#define INIT_hipDeviceGet_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGet.device = (hipDevice_t*)device; \
  cb_data.args.hipDeviceGet.ordinal = (int)deviceId; \
};
#define INIT_hipProfilerStart_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipCtxSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetCacheConfig.cacheConfig = (hipFuncCache_t)cacheConfig; \
};
#define INIT_hipFuncSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncSetCacheConfig.func = (const void*)func; \
  cb_data.args.hipFuncSetCacheConfig.config = (hipFuncCache_t)cacheConfig; \
};
#define INIT_hipMemcpyPeerAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyPeerAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpyPeerAsync.dstDeviceId = (int)dstDevice; \
  cb_data.args.hipMemcpyPeerAsync.src = (const void*)src; \
  cb_data.args.hipMemcpyPeerAsync.srcDevice = (int)srcDevice; \
  cb_data.args.hipMemcpyPeerAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyPeerAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipEventElapsedTime_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventElapsedTime.ms = (float*)ms; \
  cb_data.args.hipEventElapsedTime.start = (hipEvent_t)start; \
  cb_data.args.hipEventElapsedTime.stop = (hipEvent_t)stop; \
};
#define INIT_hipDevicePrimaryCtxReset_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxReset.dev = (hipDevice_t)dev; \
};
#define INIT_hipEventDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventDestroy.event = (hipEvent_t)event; \
};
#define INIT_hipCtxPopCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxPopCurrent.ctx = (hipCtx_t*)ctx; \
};
#define INIT_hipHostGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostGetFlags.flagsPtr = (unsigned int*)flagsPtr; \
  cb_data.args.hipHostGetFlags.hostPtr = (void*)hostPtr; \
};
#define INIT_hipHostMalloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostMalloc.ptr = (void**)ptr; \
  cb_data.args.hipHostMalloc.size = (size_t)sizeBytes; \
  cb_data.args.hipHostMalloc.flags = (unsigned int)flags; \
};
#define INIT_hipDriverGetVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDriverGetVersion.driverVersion = (int*)driverVersion; \
};
#define INIT_hipMemGetInfo_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemGetInfo.free = (size_t*)free; \
  cb_data.args.hipMemGetInfo.total = (size_t*)total; \
};
#define INIT_hipDeviceReset_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipMemset_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset.dst = (void*)dst; \
  cb_data.args.hipMemset.value = (int)value; \
  cb_data.args.hipMemset.sizeBytes = (size_t)sizeBytes; \
};
#define INIT_hipMemsetD8_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD8.dest = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemsetD8.value = (unsigned char)value; \
  cb_data.args.hipMemsetD8.sizeBytes = (size_t)sizeBytes; \
};
#define INIT_hipHostRegister_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostRegister.hostPtr = (void*)hostPtr; \
  cb_data.args.hipHostRegister.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipHostRegister.flags = (unsigned int)flags; \
};
#define INIT_hipCtxSetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetSharedMemConfig.config = (hipSharedMemConfig)config; \
};
#define INIT_hipArray3DCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipArray3DCreate.array = (hipArray**)array; \
  cb_data.args.hipArray3DCreate.pAllocateArray = (const HIP_ARRAY_DESCRIPTOR*)pAllocateArray; \
};
#define INIT_hipIpcOpenMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcOpenMemHandle.devPtr = (void**)devPtr; \
  cb_data.args.hipIpcOpenMemHandle.handle = (hipIpcMemHandle_t)handle; \
  cb_data.args.hipIpcOpenMemHandle.flags = (unsigned int)flags; \
};
#define INIT_hipGetLastError_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipCtxDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxDestroy.ctx = (hipCtx_t)ctx; \
};
#define INIT_hipDeviceGetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetSharedMemConfig.pConfig = (hipSharedMemConfig*)pConfig; \
};
#define INIT_hipRegisterActivityCallback_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipRegisterActivityCallback.id = (uint32_t)id; \
  cb_data.args.hipRegisterActivityCallback.fun = (void*)fun; \
  cb_data.args.hipRegisterActivityCallback.arg = (void*)arg; \
};
#define INIT_hipSetDeviceFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetDeviceFlags.flags = (unsigned)flags; \
};
#define INIT_hipFree_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFree.ptr = (void*)ptr; \
};
#define INIT_hipDeviceGetAttribute_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetAttribute.pi = (int*)pi; \
  cb_data.args.hipDeviceGetAttribute.attr = (hipDeviceAttribute_t)attr; \
  cb_data.args.hipDeviceGetAttribute.deviceId = (int)device; \
};
#define INIT_hipMemcpyDtoH_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoH.dst = (void*)dst; \
  cb_data.args.hipMemcpyDtoH.src = (hipDeviceptr_t)src; \
  cb_data.args.hipMemcpyDtoH.sizeBytes = (size_t)sizeBytes; \
};
#define INIT_hipCtxDisablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxDisablePeerAccess.peerCtx = (hipCtx_t)peerCtx; \
};
#define INIT_hipDeviceGetByPCIBusId_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetByPCIBusId.device = (int*)device; \
  cb_data.args.hipDeviceGetByPCIBusId.pciBusId = (const char*)pciBusId; \
};
#define INIT_hipIpcGetMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcGetMemHandle.handle = (hipIpcMemHandle_t*)handle; \
  cb_data.args.hipIpcGetMemHandle.devPtr = (void*)devPtr; \
};
#define INIT_hipMemcpyHtoDAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoDAsync.dst = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemcpyHtoDAsync.src = (void*)src; \
  cb_data.args.hipMemcpyHtoDAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyHtoDAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipCtxGetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetDevice.device = (hipDevice_t*)device; \
};
#define INIT_hipMemset3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset3D.pitchedDevPtr = (hipPitchedPtr)pitchedDevPtr; \
  cb_data.args.hipMemset3D.value = (int)value; \
  cb_data.args.hipMemset3D.extent = (hipExtent)extent; \
};
#define INIT_hipModuleLoadData_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoadData.module = (hipModule_t*)module; \
  cb_data.args.hipModuleLoadData.image = (const void*)image; \
};
#define INIT_hipDeviceTotalMem_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceTotalMem.bytes = (size_t*)bytes; \
  cb_data.args.hipDeviceTotalMem.device = (hipDevice_t)device; \
};
#define INIT_hipCtxSetCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetCurrent.ctx = (hipCtx_t)ctx; \
};
#define INIT_hipMallocHost_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocHost.ptr = (void**)ptr; \
  cb_data.args.hipMallocHost.size = (size_t)sizeBytes; \
};
#define INIT_hipDevicePrimaryCtxRetain_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxRetain.pctx = (hipCtx_t*)pctx; \
  cb_data.args.hipDevicePrimaryCtxRetain.dev = (hipDevice_t)dev; \
};
#define INIT_hipDeviceDisablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceDisablePeerAccess.peerDeviceId = (int)peerDeviceId; \
};
#define INIT_hipStreamCreateWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreateWithFlags.stream = (hipStream_t*)stream; \
  cb_data.args.hipStreamCreateWithFlags.flags = (unsigned int)flags; \
};
#define INIT_hipMemcpyFromArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyFromArray.dst = (void*)dst; \
  cb_data.args.hipMemcpyFromArray.srcArray = (hipArray_const_t)srcArray; \
  cb_data.args.hipMemcpyFromArray.wOffset = (size_t)wOffset; \
  cb_data.args.hipMemcpyFromArray.hOffset = (size_t)hOffset; \
  cb_data.args.hipMemcpyFromArray.count = (size_t)count; \
  cb_data.args.hipMemcpyFromArray.kind = (hipMemcpyKind)kind; \
};
#define INIT_hipMemcpy2DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2DAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpy2DAsync.dpitch = (size_t)dpitch; \
  cb_data.args.hipMemcpy2DAsync.src = (const void*)src; \
  cb_data.args.hipMemcpy2DAsync.spitch = (size_t)spitch; \
  cb_data.args.hipMemcpy2DAsync.width = (size_t)width; \
  cb_data.args.hipMemcpy2DAsync.height = (size_t)height; \
  cb_data.args.hipMemcpy2DAsync.kind = (hipMemcpyKind)kind; \
  cb_data.args.hipMemcpy2DAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipFuncGetAttributes_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncGetAttributes.attr = (hipFuncAttributes*)attr; \
  cb_data.args.hipFuncGetAttributes.func = (const void*)func; \
};
#define INIT_hipEventCreateWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventCreateWithFlags.event = (hipEvent_t*)event; \
  cb_data.args.hipEventCreateWithFlags.flags = (unsigned)flags; \
};
#define INIT_hipStreamQuery_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamQuery.stream = (hipStream_t)stream; \
};
#define INIT_hipDeviceGetPCIBusId_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetPCIBusId.pciBusId = (char*)pciBusId; \
  cb_data.args.hipDeviceGetPCIBusId.len = (int)len; \
  cb_data.args.hipDeviceGetPCIBusId.device = (int)device; \
};
#define INIT_hipMemcpy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy.dst = (void*)dst; \
  cb_data.args.hipMemcpy.src = (const void*)src; \
  cb_data.args.hipMemcpy.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpy.kind = (hipMemcpyKind)kind; \
};
#define INIT_hipPeekAtLastError_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipHostAlloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostAlloc.ptr = (void**)ptr; \
  cb_data.args.hipHostAlloc.size = (size_t)sizeBytes; \
  cb_data.args.hipHostAlloc.flags = (unsigned int)flags; \
};
#define INIT_hipStreamAddCallback_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamAddCallback.stream = (hipStream_t)stream; \
  cb_data.args.hipStreamAddCallback.callback = (hipStreamCallback_t)callback; \
  cb_data.args.hipStreamAddCallback.userData = (void*)userData; \
  cb_data.args.hipStreamAddCallback.flags = (unsigned int)flags; \
};
#define INIT_hipMemcpyToArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyToArray.dst = (hipArray*)dst; \
  cb_data.args.hipMemcpyToArray.wOffset = (size_t)wOffset; \
  cb_data.args.hipMemcpyToArray.hOffset = (size_t)hOffset; \
  cb_data.args.hipMemcpyToArray.src = (const void*)src; \
  cb_data.args.hipMemcpyToArray.count = (size_t)count; \
  cb_data.args.hipMemcpyToArray.kind = (hipMemcpyKind)kind; \
};
#define INIT_hipDeviceSynchronize_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipDeviceGetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetCacheConfig.cacheConfig = (hipFuncCache_t*)cacheConfig; \
};
#define INIT_hipMalloc3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc3D.pitchedDevPtr = (hipPitchedPtr*)pitchedDevPtr; \
  cb_data.args.hipMalloc3D.extent = (hipExtent)extent; \
};
#define INIT_hipPointerGetAttributes_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipPointerGetAttributes.attributes = (hipPointerAttribute_t*)attributes; \
  cb_data.args.hipPointerGetAttributes.ptr = (const void*)ptr; \
};
#define INIT_hipMemsetAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetAsync.dst = (void*)dst; \
  cb_data.args.hipMemsetAsync.value = (int)value; \
  cb_data.args.hipMemsetAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemsetAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipMemcpyToSymbol_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyToSymbol.symbolName = (const void*)symbolName; \
  cb_data.args.hipMemcpyToSymbol.src = (const void*)src; \
  cb_data.args.hipMemcpyToSymbol.sizeBytes = (size_t)count; \
  cb_data.args.hipMemcpyToSymbol.offset = (size_t)offset; \
  cb_data.args.hipMemcpyToSymbol.kind = (hipMemcpyKind)kind; \
};
#define INIT_hipCtxPushCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxPushCurrent.ctx = (hipCtx_t)ctx; \
};
#define INIT_hipMemcpyPeer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyPeer.dst = (void*)dst; \
  cb_data.args.hipMemcpyPeer.dstDeviceId = (int)dstDevice; \
  cb_data.args.hipMemcpyPeer.src = (const void*)src; \
  cb_data.args.hipMemcpyPeer.srcDeviceId = (int)srcDevice; \
  cb_data.args.hipMemcpyPeer.sizeBytes = (size_t)sizeBytes; \
};
#define INIT_hipEventSynchronize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventSynchronize.event = (hipEvent_t)event; \
};
#define INIT_hipMemcpyDtoDAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoDAsync.dst = (hipDeviceptr_t)dst; \
  cb_data.args.hipMemcpyDtoDAsync.src = (hipDeviceptr_t)src; \
  cb_data.args.hipMemcpyDtoDAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyDtoDAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipCtxEnablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxEnablePeerAccess.peerCtx = (hipCtx_t)peerCtx; \
  cb_data.args.hipCtxEnablePeerAccess.flags = (unsigned int)flags; \
};
#define INIT_hipMemcpyDtoHAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoHAsync.dst = (void*)dst; \
  cb_data.args.hipMemcpyDtoHAsync.src = (hipDeviceptr_t)src; \
  cb_data.args.hipMemcpyDtoHAsync.sizeBytes = (size_t)sizeBytes; \
  cb_data.args.hipMemcpyDtoHAsync.stream = (hipStream_t)stream; \
};
#define INIT_hipModuleLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLaunchKernel.f = (hipFunction_t)f; \
  cb_data.args.hipModuleLaunchKernel.gridDimX = (unsigned int)gridDimX; \
  cb_data.args.hipModuleLaunchKernel.gridDimY = (unsigned int)gridDimY; \
  cb_data.args.hipModuleLaunchKernel.gridDimZ = (unsigned int)gridDimZ; \
  cb_data.args.hipModuleLaunchKernel.blockDimX = (unsigned int)blockDimX; \
  cb_data.args.hipModuleLaunchKernel.blockDimY = (unsigned int)blockDimY; \
  cb_data.args.hipModuleLaunchKernel.blockDimZ = (unsigned int)blockDimZ; \
  cb_data.args.hipModuleLaunchKernel.sharedMemBytes = (unsigned int)sharedMemBytes; \
  cb_data.args.hipModuleLaunchKernel.stream = (hipStream_t)hStream; \
  cb_data.args.hipModuleLaunchKernel.kernelParams = (void**)kernelParams; \
  cb_data.args.hipModuleLaunchKernel.extra = (void**)extra; \
};
#define INIT_hipHccModuleLaunchKernel_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLaunchKernel.f = (hipFunction_t)f; \
};
#define INIT_hipModuleGetTexRef_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetTexRef.texRef = (textureReference**)texRef; \
  cb_data.args.hipModuleGetTexRef.hmod = (hipModule_t)hmod; \
  cb_data.args.hipModuleGetTexRef.name = (const char*)name; \
};
#define INIT_hipRemoveActivityCallback_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipRemoveActivityCallback.id = (uint32_t)id; \
};
#define INIT_hipDeviceGetLimit_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetLimit.pValue = (size_t*)pValue; \
  cb_data.args.hipDeviceGetLimit.limit = (hipLimit_t)limit; \
};
#define INIT_hipModuleLoadDataEx_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoadDataEx.module = (hipModule_t*)module; \
  cb_data.args.hipModuleLoadDataEx.image = (const void*)image; \
  cb_data.args.hipModuleLoadDataEx.numOptions = (unsigned int)numOptions; \
  cb_data.args.hipModuleLoadDataEx.options = (hipJitOption*)options; \
  cb_data.args.hipModuleLoadDataEx.optionValues = (void**)optionValues; \
};
#define INIT_hipRuntimeGetVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipRuntimeGetVersion.runtimeVersion = (int*)runtimeVersion; \
};
#define INIT_hipGetDeviceProperties_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDeviceProperties.prop = (hipDeviceProp_t*)props; \
  cb_data.args.hipGetDeviceProperties.deviceId = (int)device; \
};
#define INIT_hipFreeArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFreeArray.array = (hipArray*)array; \
};
#define INIT_hipDevicePrimaryCtxRelease_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxRelease.dev = (hipDevice_t)dev; \
};
#define INIT_hipHostGetDevicePointer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostGetDevicePointer.devPtr = (void**)devicePointer; \
  cb_data.args.hipHostGetDevicePointer.hstPtr = (void*)hostPointer; \
  cb_data.args.hipHostGetDevicePointer.flags = (unsigned int)flags; \
};
#define INIT_hipMemcpyParam2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyParam2D.pCopy = (const hip_Memcpy2D*)pCopy; \
};
#define INIT_hipConfigureCall_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipConfigureCall.gridDim = (dim3)gridDim; \
  cb_data.args.hipConfigureCall.blockDim = (dim3)blockDim; \
  cb_data.args.hipConfigureCall.sharedMem = (size_t)sharedMem; \
  cb_data.args.hipConfigureCall.stream = (hipStream_t)stream; \
};
#define INIT_hipModuleGetFunction_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetFunction.function = (hipFunction_t*)hfunc; \
  cb_data.args.hipModuleGetFunction.module = (hipModule_t)hmod; \
  cb_data.args.hipModuleGetFunction.kname = (const char*)name; \
};
#define INIT_hipGetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDevice.deviceId = (int*)deviceId; \
};
#define INIT_hipGetDeviceCount_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDeviceCount.count = (int*)count; \
};
#define INIT_CB_ARGS_DATA(cb_id, cb_data) INIT_##cb_id##_CB_ARGS_DATA(cb_data)

#if 0
// HIP API string method, method name and parameters
const char* hipApiString(hip_api_id_t id, const hip_api_data_t* data) {
  std::ostringstream oss;
  switch (id) {
    case HIP_API_ID_hipHostFree:
      oss << "hipHostFree("
          << " ptr=" << data->args.hipHostFree.ptr
          << ")";
    break;
    case HIP_API_ID_hipMemcpyToSymbolAsync:
      oss << "hipMemcpyToSymbolAsync("
          << " symbolName=" << data->args.hipMemcpyToSymbolAsync.symbolName << ","
          << " src=" << data->args.hipMemcpyToSymbolAsync.src << ","
          << " sizeBytes=" << data->args.hipMemcpyToSymbolAsync.sizeBytes << ","
          << " offset=" << data->args.hipMemcpyToSymbolAsync.offset << ","
          << " kind=" << data->args.hipMemcpyToSymbolAsync.kind << ","
          << " stream=" << data->args.hipMemcpyToSymbolAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipMallocPitch:
      oss << "hipMallocPitch("
          << " ptr=" << data->args.hipMallocPitch.ptr << ","
          << " pitch=" << data->args.hipMallocPitch.pitch << ","
          << " width=" << data->args.hipMallocPitch.width << ","
          << " height=" << data->args.hipMallocPitch.height
          << ")";
    break;
    case HIP_API_ID_hipMalloc:
      oss << "hipMalloc("
          << " ptr=" << data->args.hipMalloc.ptr << ","
          << " size=" << data->args.hipMalloc.size
          << ")";
    break;
    case HIP_API_ID_hipDeviceGetName:
      oss << "hipDeviceGetName("
          << " name=" << data->args.hipDeviceGetName.name << ","
          << " len=" << data->args.hipDeviceGetName.len << ","
          << " device=" << data->args.hipDeviceGetName.device
          << ")";
    break;
    case HIP_API_ID_hipEventRecord:
      oss << "hipEventRecord("
          << " event=" << data->args.hipEventRecord.event << ","
          << " stream=" << data->args.hipEventRecord.stream
          << ")";
    break;
    case HIP_API_ID_hipCtxSynchronize:
      oss << "hipCtxSynchronize("
          << ")";
    break;
    case HIP_API_ID_hipSetDevice:
      oss << "hipSetDevice("
          << " deviceId=" << data->args.hipSetDevice.deviceId
          << ")";
    break;
    case HIP_API_ID_hipSetupArgument:
      oss << "hipSetupArgument("
          << " arg=" << data->args.hipSetupArgument.arg << ","
          << " size=" << data->args.hipSetupArgument.size << ","
          << " offset=" << data->args.hipSetupArgument.offset
          << ")";
    break;
    case HIP_API_ID_hipMemcpyFromSymbolAsync:
      oss << "hipMemcpyFromSymbolAsync("
          << " dst=" << data->args.hipMemcpyFromSymbolAsync.dst << ","
          << " symbolName=" << data->args.hipMemcpyFromSymbolAsync.symbolName << ","
          << " sizeBytes=" << data->args.hipMemcpyFromSymbolAsync.sizeBytes << ","
          << " offset=" << data->args.hipMemcpyFromSymbolAsync.offset << ","
          << " kind=" << data->args.hipMemcpyFromSymbolAsync.kind << ","
          << " stream=" << data->args.hipMemcpyFromSymbolAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipMemcpyDtoD:
      oss << "hipMemcpyDtoD("
          << " dst=" << data->args.hipMemcpyDtoD.dst << ","
          << " src=" << data->args.hipMemcpyDtoD.src << ","
          << " sizeBytes=" << data->args.hipMemcpyDtoD.sizeBytes
          << ")";
    break;
    case HIP_API_ID_hipMemcpy2DToArray:
      oss << "hipMemcpy2DToArray("
          << " dst=" << data->args.hipMemcpy2DToArray.dst << ","
          << " wOffset=" << data->args.hipMemcpy2DToArray.wOffset << ","
          << " hOffset=" << data->args.hipMemcpy2DToArray.hOffset << ","
          << " src=" << data->args.hipMemcpy2DToArray.src << ","
          << " spitch=" << data->args.hipMemcpy2DToArray.spitch << ","
          << " width=" << data->args.hipMemcpy2DToArray.width << ","
          << " height=" << data->args.hipMemcpy2DToArray.height << ","
          << " kind=" << data->args.hipMemcpy2DToArray.kind
          << ")";
    break;
    case HIP_API_ID_hipCtxGetCacheConfig:
      oss << "hipCtxGetCacheConfig("
          << " cacheConfig=" << data->args.hipCtxGetCacheConfig.cacheConfig
          << ")";
    break;
    case HIP_API_ID_hipStreamWaitEvent:
      oss << "hipStreamWaitEvent("
          << " stream=" << data->args.hipStreamWaitEvent.stream << ","
          << " event=" << data->args.hipStreamWaitEvent.event << ","
          << " flags=" << data->args.hipStreamWaitEvent.flags
          << ")";
    break;
    case HIP_API_ID_hipModuleLoad:
      oss << "hipModuleLoad("
          << " module=" << data->args.hipModuleLoad.module << ","
          << " fname=" << data->args.hipModuleLoad.fname
          << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxSetFlags:
      oss << "hipDevicePrimaryCtxSetFlags("
          << " dev=" << data->args.hipDevicePrimaryCtxSetFlags.dev << ","
          << " flags=" << data->args.hipDevicePrimaryCtxSetFlags.flags
          << ")";
    break;
    case HIP_API_ID_hipMemcpyAsync:
      oss << "hipMemcpyAsync("
          << " dst=" << data->args.hipMemcpyAsync.dst << ","
          << " src=" << data->args.hipMemcpyAsync.src << ","
          << " sizeBytes=" << data->args.hipMemcpyAsync.sizeBytes << ","
          << " kind=" << data->args.hipMemcpyAsync.kind << ","
          << " stream=" << data->args.hipMemcpyAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipMalloc3DArray:
      oss << "hipMalloc3DArray("
          << " array=" << data->args.hipMalloc3DArray.array << ","
          << " desc=" << data->args.hipMalloc3DArray.desc << ","
          << " extent=" << data->args.hipMalloc3DArray.extent << ","
          << " flags=" << data->args.hipMalloc3DArray.flags
          << ")";
    break;
    case HIP_API_ID_hipStreamCreate:
      oss << "hipStreamCreate("
          << " stream=" << data->args.hipStreamCreate.stream
          << ")";
    break;
    case HIP_API_ID_hipCtxGetCurrent:
      oss << "hipCtxGetCurrent("
          << " ctx=" << data->args.hipCtxGetCurrent.ctx
          << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxGetState:
      oss << "hipDevicePrimaryCtxGetState("
          << " dev=" << data->args.hipDevicePrimaryCtxGetState.dev << ","
          << " flags=" << data->args.hipDevicePrimaryCtxGetState.flags << ","
          << " active=" << data->args.hipDevicePrimaryCtxGetState.active
          << ")";
    break;
    case HIP_API_ID_hipEventQuery:
      oss << "hipEventQuery("
          << " event=" << data->args.hipEventQuery.event
          << ")";
    break;
    case HIP_API_ID_hipEventCreate:
      oss << "hipEventCreate("
          << " event=" << data->args.hipEventCreate.event
          << ")";
    break;
    case HIP_API_ID_hipMemGetAddressRange:
      oss << "hipMemGetAddressRange("
          << " pbase=" << data->args.hipMemGetAddressRange.pbase << ","
          << " psize=" << data->args.hipMemGetAddressRange.psize << ","
          << " dptr=" << data->args.hipMemGetAddressRange.dptr
          << ")";
    break;
    case HIP_API_ID_hipMemcpyFromSymbol:
      oss << "hipMemcpyFromSymbol("
          << " dst=" << data->args.hipMemcpyFromSymbol.dst << ","
          << " symbolName=" << data->args.hipMemcpyFromSymbol.symbolName << ","
          << " sizeBytes=" << data->args.hipMemcpyFromSymbol.sizeBytes << ","
          << " offset=" << data->args.hipMemcpyFromSymbol.offset << ","
          << " kind=" << data->args.hipMemcpyFromSymbol.kind
          << ")";
    break;
    case HIP_API_ID_hipArrayCreate:
      oss << "hipArrayCreate("
          << " pHandle=" << data->args.hipArrayCreate.pHandle << ","
          << " pAllocateArray=" << data->args.hipArrayCreate.pAllocateArray
          << ")";
    break;
    case HIP_API_ID_hipStreamGetFlags:
      oss << "hipStreamGetFlags("
          << " stream=" << data->args.hipStreamGetFlags.stream << ","
          << " flags=" << data->args.hipStreamGetFlags.flags
          << ")";
    break;
    case HIP_API_ID_hipMallocArray:
      oss << "hipMallocArray("
          << " array=" << data->args.hipMallocArray.array << ","
          << " desc=" << data->args.hipMallocArray.desc << ","
          << " width=" << data->args.hipMallocArray.width << ","
          << " height=" << data->args.hipMallocArray.height << ","
          << " flags=" << data->args.hipMallocArray.flags
          << ")";
    break;
    case HIP_API_ID_hipCtxGetSharedMemConfig:
      oss << "hipCtxGetSharedMemConfig("
          << " pConfig=" << data->args.hipCtxGetSharedMemConfig.pConfig
          << ")";
    break;
    case HIP_API_ID_hipMemPtrGetInfo:
      oss << "hipMemPtrGetInfo("
          << " ptr=" << data->args.hipMemPtrGetInfo.ptr << ","
          << " size=" << data->args.hipMemPtrGetInfo.size
          << ")";
    break;
    case HIP_API_ID_hipCtxGetFlags:
      oss << "hipCtxGetFlags("
          << " flags=" << data->args.hipCtxGetFlags.flags
          << ")";
    break;
    case HIP_API_ID_hipStreamDestroy:
      oss << "hipStreamDestroy("
          << " stream=" << data->args.hipStreamDestroy.stream
          << ")";
    break;
    case HIP_API_ID_hipMemset3DAsync:
      oss << "hipMemset3DAsync("
          << " pitchedDevPtr=" << data->args.hipMemset3DAsync.pitchedDevPtr << ","
          << " value=" << data->args.hipMemset3DAsync.value << ","
          << " extent=" << data->args.hipMemset3DAsync.extent << ","
          << " stream=" << data->args.hipMemset3DAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipMemcpy3D:
      oss << "hipMemcpy3D("
          << " p=" << data->args.hipMemcpy3D.p
          << ")";
    break;
    case HIP_API_ID_hipInit:
      oss << "hipInit("
          << " flags=" << data->args.hipInit.flags
          << ")";
    break;
    case HIP_API_ID_hipMemcpyAtoH:
      oss << "hipMemcpyAtoH("
          << " dst=" << data->args.hipMemcpyAtoH.dst << ","
          << " srcArray=" << data->args.hipMemcpyAtoH.srcArray << ","
          << " srcOffset=" << data->args.hipMemcpyAtoH.srcOffset << ","
          << " count=" << data->args.hipMemcpyAtoH.count
          << ")";
    break;
    case HIP_API_ID_hipMemset2D:
      oss << "hipMemset2D("
          << " dst=" << data->args.hipMemset2D.dst << ","
          << " pitch=" << data->args.hipMemset2D.pitch << ","
          << " value=" << data->args.hipMemset2D.value << ","
          << " width=" << data->args.hipMemset2D.width << ","
          << " height=" << data->args.hipMemset2D.height
          << ")";
    break;
    case HIP_API_ID_hipMemset2DAsync:
      oss << "hipMemset2DAsync("
          << " dst=" << data->args.hipMemset2DAsync.dst << ","
          << " pitch=" << data->args.hipMemset2DAsync.pitch << ","
          << " value=" << data->args.hipMemset2DAsync.value << ","
          << " width=" << data->args.hipMemset2DAsync.width << ","
          << " height=" << data->args.hipMemset2DAsync.height << ","
          << " stream=" << data->args.hipMemset2DAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipDeviceCanAccessPeer:
      oss << "hipDeviceCanAccessPeer("
          << " canAccessPeer=" << data->args.hipDeviceCanAccessPeer.canAccessPeer << ","
          << " deviceId=" << data->args.hipDeviceCanAccessPeer.deviceId << ","
          << " peerDeviceId=" << data->args.hipDeviceCanAccessPeer.peerDeviceId
          << ")";
    break;
    case HIP_API_ID_hipDeviceEnablePeerAccess:
      oss << "hipDeviceEnablePeerAccess("
          << " peerDeviceId=" << data->args.hipDeviceEnablePeerAccess.peerDeviceId << ","
          << " flags=" << data->args.hipDeviceEnablePeerAccess.flags
          << ")";
    break;
    case HIP_API_ID_hipModuleUnload:
      oss << "hipModuleUnload("
          << " module=" << data->args.hipModuleUnload.module
          << ")";
    break;
    case HIP_API_ID_hipHostUnregister:
      oss << "hipHostUnregister("
          << " hostPtr=" << data->args.hipHostUnregister.hostPtr
          << ")";
    break;
    case HIP_API_ID_hipProfilerStop:
      oss << "hipProfilerStop("
          << ")";
    break;
    case HIP_API_ID_hipLaunchByPtr:
      oss << "hipLaunchByPtr("
          << " func=" << data->args.hipLaunchByPtr.func
          << ")";
    break;
    case HIP_API_ID_hipStreamSynchronize:
      oss << "hipStreamSynchronize("
          << " stream=" << data->args.hipStreamSynchronize.stream
          << ")";
    break;
    case HIP_API_ID_hipFreeHost:
      oss << "hipFreeHost("
          << " ptr=" << data->args.hipFreeHost.ptr
          << ")";
    break;
    case HIP_API_ID_hipRemoveApiCallback:
      oss << "hipRemoveApiCallback("
          << " id=" << data->args.hipRemoveApiCallback.id
          << ")";
    break;
    case HIP_API_ID_hipDeviceSetCacheConfig:
      oss << "hipDeviceSetCacheConfig("
          << " cacheConfig=" << data->args.hipDeviceSetCacheConfig.cacheConfig
          << ")";
    break;
    case HIP_API_ID_hipCtxGetApiVersion:
      oss << "hipCtxGetApiVersion("
          << " ctx=" << data->args.hipCtxGetApiVersion.ctx << ","
          << " apiVersion=" << data->args.hipCtxGetApiVersion.apiVersion
          << ")";
    break;
    case HIP_API_ID_hipMemcpyHtoD:
      oss << "hipMemcpyHtoD("
          << " dst=" << data->args.hipMemcpyHtoD.dst << ","
          << " src=" << data->args.hipMemcpyHtoD.src << ","
          << " sizeBytes=" << data->args.hipMemcpyHtoD.sizeBytes
          << ")";
    break;
    case HIP_API_ID_hipModuleGetGlobal:
      oss << "hipModuleGetGlobal("
          << " dptr=" << data->args.hipModuleGetGlobal.dptr << ","
          << " bytes=" << data->args.hipModuleGetGlobal.bytes << ","
          << " hmod=" << data->args.hipModuleGetGlobal.hmod << ","
          << " name=" << data->args.hipModuleGetGlobal.name
          << ")";
    break;
    case HIP_API_ID_hipMemcpyHtoA:
      oss << "hipMemcpyHtoA("
          << " dstArray=" << data->args.hipMemcpyHtoA.dstArray << ","
          << " dstOffset=" << data->args.hipMemcpyHtoA.dstOffset << ","
          << " srcHost=" << data->args.hipMemcpyHtoA.srcHost << ","
          << " count=" << data->args.hipMemcpyHtoA.count
          << ")";
    break;
    case HIP_API_ID_hipCtxCreate:
      oss << "hipCtxCreate("
          << " ctx=" << data->args.hipCtxCreate.ctx << ","
          << " flags=" << data->args.hipCtxCreate.flags << ","
          << " device=" << data->args.hipCtxCreate.device
          << ")";
    break;
    case HIP_API_ID_hipMemcpy2D:
      oss << "hipMemcpy2D("
          << " dst=" << data->args.hipMemcpy2D.dst << ","
          << " dpitch=" << data->args.hipMemcpy2D.dpitch << ","
          << " src=" << data->args.hipMemcpy2D.src << ","
          << " spitch=" << data->args.hipMemcpy2D.spitch << ","
          << " width=" << data->args.hipMemcpy2D.width << ","
          << " height=" << data->args.hipMemcpy2D.height << ","
          << " kind=" << data->args.hipMemcpy2D.kind
          << ")";
    break;
    case HIP_API_ID_hipIpcCloseMemHandle:
      oss << "hipIpcCloseMemHandle("
          << " devPtr=" << data->args.hipIpcCloseMemHandle.devPtr
          << ")";
    break;
    case HIP_API_ID_hipChooseDevice:
      oss << "hipChooseDevice("
          << " device=" << data->args.hipChooseDevice.device << ","
          << " prop=" << data->args.hipChooseDevice.prop
          << ")";
    break;
    case HIP_API_ID_hipDeviceSetSharedMemConfig:
      oss << "hipDeviceSetSharedMemConfig("
          << " config=" << data->args.hipDeviceSetSharedMemConfig.config
          << ")";
    break;
    case HIP_API_ID_hipDeviceComputeCapability:
      oss << "hipDeviceComputeCapability("
          << " major=" << data->args.hipDeviceComputeCapability.major << ","
          << " minor=" << data->args.hipDeviceComputeCapability.minor << ","
          << " device=" << data->args.hipDeviceComputeCapability.device
          << ")";
    break;
    case HIP_API_ID_hipRegisterApiCallback:
      oss << "hipRegisterApiCallback("
          << " id=" << data->args.hipRegisterApiCallback.id << ","
          << " fun=" << data->args.hipRegisterApiCallback.fun << ","
          << " arg=" << data->args.hipRegisterApiCallback.arg
          << ")";
    break;
    case HIP_API_ID_hipDeviceGet:
      oss << "hipDeviceGet("
          << " device=" << data->args.hipDeviceGet.device << ","
          << " ordinal=" << data->args.hipDeviceGet.ordinal
          << ")";
    break;
    case HIP_API_ID_hipProfilerStart:
      oss << "hipProfilerStart("
          << ")";
    break;
    case HIP_API_ID_hipCtxSetCacheConfig:
      oss << "hipCtxSetCacheConfig("
          << " cacheConfig=" << data->args.hipCtxSetCacheConfig.cacheConfig
          << ")";
    break;
    case HIP_API_ID_hipFuncSetCacheConfig:
      oss << "hipFuncSetCacheConfig("
          << " func=" << data->args.hipFuncSetCacheConfig.func << ","
          << " config=" << data->args.hipFuncSetCacheConfig.config
          << ")";
    break;
    case HIP_API_ID_hipMemcpyPeerAsync:
      oss << "hipMemcpyPeerAsync("
          << " dst=" << data->args.hipMemcpyPeerAsync.dst << ","
          << " dstDeviceId=" << data->args.hipMemcpyPeerAsync.dstDeviceId << ","
          << " src=" << data->args.hipMemcpyPeerAsync.src << ","
          << " srcDevice=" << data->args.hipMemcpyPeerAsync.srcDevice << ","
          << " sizeBytes=" << data->args.hipMemcpyPeerAsync.sizeBytes << ","
          << " stream=" << data->args.hipMemcpyPeerAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipEventElapsedTime:
      oss << "hipEventElapsedTime("
          << " ms=" << data->args.hipEventElapsedTime.ms << ","
          << " start=" << data->args.hipEventElapsedTime.start << ","
          << " stop=" << data->args.hipEventElapsedTime.stop
          << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxReset:
      oss << "hipDevicePrimaryCtxReset("
          << " dev=" << data->args.hipDevicePrimaryCtxReset.dev
          << ")";
    break;
    case HIP_API_ID_hipEventDestroy:
      oss << "hipEventDestroy("
          << " event=" << data->args.hipEventDestroy.event
          << ")";
    break;
    case HIP_API_ID_hipCtxPopCurrent:
      oss << "hipCtxPopCurrent("
          << " ctx=" << data->args.hipCtxPopCurrent.ctx
          << ")";
    break;
    case HIP_API_ID_hipHostGetFlags:
      oss << "hipHostGetFlags("
          << " flagsPtr=" << data->args.hipHostGetFlags.flagsPtr << ","
          << " hostPtr=" << data->args.hipHostGetFlags.hostPtr
          << ")";
    break;
    case HIP_API_ID_hipHostMalloc:
      oss << "hipHostMalloc("
          << " ptr=" << data->args.hipHostMalloc.ptr << ","
          << " size=" << data->args.hipHostMalloc.size << ","
          << " flags=" << data->args.hipHostMalloc.flags
          << ")";
    break;
    case HIP_API_ID_hipDriverGetVersion:
      oss << "hipDriverGetVersion("
          << " driverVersion=" << data->args.hipDriverGetVersion.driverVersion
          << ")";
    break;
    case HIP_API_ID_hipMemGetInfo:
      oss << "hipMemGetInfo("
          << " free=" << data->args.hipMemGetInfo.free << ","
          << " total=" << data->args.hipMemGetInfo.total
          << ")";
    break;
    case HIP_API_ID_hipDeviceReset:
      oss << "hipDeviceReset("
          << ")";
    break;
    case HIP_API_ID_hipMemset:
      oss << "hipMemset("
          << " dst=" << data->args.hipMemset.dst << ","
          << " value=" << data->args.hipMemset.value << ","
          << " sizeBytes=" << data->args.hipMemset.sizeBytes
          << ")";
    break;
    case HIP_API_ID_hipMemsetD8:
      oss << "hipMemsetD8("
          << " dest=" << data->args.hipMemsetD8.dest << ","
          << " value=" << data->args.hipMemsetD8.value << ","
          << " sizeBytes=" << data->args.hipMemsetD8.sizeBytes
          << ")";
    break;
    case HIP_API_ID_hipHostRegister:
      oss << "hipHostRegister("
          << " hostPtr=" << data->args.hipHostRegister.hostPtr << ","
          << " sizeBytes=" << data->args.hipHostRegister.sizeBytes << ","
          << " flags=" << data->args.hipHostRegister.flags
          << ")";
    break;
    case HIP_API_ID_hipCtxSetSharedMemConfig:
      oss << "hipCtxSetSharedMemConfig("
          << " config=" << data->args.hipCtxSetSharedMemConfig.config
          << ")";
    break;
    case HIP_API_ID_hipArray3DCreate:
      oss << "hipArray3DCreate("
          << " array=" << data->args.hipArray3DCreate.array << ","
          << " pAllocateArray=" << data->args.hipArray3DCreate.pAllocateArray
          << ")";
    break;
    case HIP_API_ID_hipIpcOpenMemHandle:
      oss << "hipIpcOpenMemHandle("
          << " devPtr=" << data->args.hipIpcOpenMemHandle.devPtr << ","
          << " handle=" << data->args.hipIpcOpenMemHandle.handle << ","
          << " flags=" << data->args.hipIpcOpenMemHandle.flags
          << ")";
    break;
    case HIP_API_ID_hipGetLastError:
      oss << "hipGetLastError("
          << ")";
    break;
    case HIP_API_ID_hipCtxDestroy:
      oss << "hipCtxDestroy("
          << " ctx=" << data->args.hipCtxDestroy.ctx
          << ")";
    break;
    case HIP_API_ID_hipDeviceGetSharedMemConfig:
      oss << "hipDeviceGetSharedMemConfig("
          << " pConfig=" << data->args.hipDeviceGetSharedMemConfig.pConfig
          << ")";
    break;
    case HIP_API_ID_hipRegisterActivityCallback:
      oss << "hipRegisterActivityCallback("
          << " id=" << data->args.hipRegisterActivityCallback.id << ","
          << " fun=" << data->args.hipRegisterActivityCallback.fun << ","
          << " arg=" << data->args.hipRegisterActivityCallback.arg
          << ")";
    break;
    case HIP_API_ID_hipSetDeviceFlags:
      oss << "hipSetDeviceFlags("
          << " flags=" << data->args.hipSetDeviceFlags.flags
          << ")";
    break;
    case HIP_API_ID_hipFree:
      oss << "hipFree("
          << " ptr=" << data->args.hipFree.ptr
          << ")";
    break;
    case HIP_API_ID_hipDeviceGetAttribute:
      oss << "hipDeviceGetAttribute("
          << " pi=" << data->args.hipDeviceGetAttribute.pi << ","
          << " attr=" << data->args.hipDeviceGetAttribute.attr << ","
          << " deviceId=" << data->args.hipDeviceGetAttribute.deviceId
          << ")";
    break;
    case HIP_API_ID_hipMemcpyDtoH:
      oss << "hipMemcpyDtoH("
          << " dst=" << data->args.hipMemcpyDtoH.dst << ","
          << " src=" << data->args.hipMemcpyDtoH.src << ","
          << " sizeBytes=" << data->args.hipMemcpyDtoH.sizeBytes
          << ")";
    break;
    case HIP_API_ID_hipCtxDisablePeerAccess:
      oss << "hipCtxDisablePeerAccess("
          << " peerCtx=" << data->args.hipCtxDisablePeerAccess.peerCtx
          << ")";
    break;
    case HIP_API_ID_hipDeviceGetByPCIBusId:
      oss << "hipDeviceGetByPCIBusId("
          << " device=" << data->args.hipDeviceGetByPCIBusId.device << ","
          << " pciBusId=" << data->args.hipDeviceGetByPCIBusId.pciBusId
          << ")";
    break;
    case HIP_API_ID_hipIpcGetMemHandle:
      oss << "hipIpcGetMemHandle("
          << " handle=" << data->args.hipIpcGetMemHandle.handle << ","
          << " devPtr=" << data->args.hipIpcGetMemHandle.devPtr
          << ")";
    break;
    case HIP_API_ID_hipMemcpyHtoDAsync:
      oss << "hipMemcpyHtoDAsync("
          << " dst=" << data->args.hipMemcpyHtoDAsync.dst << ","
          << " src=" << data->args.hipMemcpyHtoDAsync.src << ","
          << " sizeBytes=" << data->args.hipMemcpyHtoDAsync.sizeBytes << ","
          << " stream=" << data->args.hipMemcpyHtoDAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipCtxGetDevice:
      oss << "hipCtxGetDevice("
          << " device=" << data->args.hipCtxGetDevice.device
          << ")";
    break;
    case HIP_API_ID_hipMemset3D:
      oss << "hipMemset3D("
          << " pitchedDevPtr=" << data->args.hipMemset3D.pitchedDevPtr << ","
          << " value=" << data->args.hipMemset3D.value << ","
          << " extent=" << data->args.hipMemset3D.extent
          << ")";
    break;
    case HIP_API_ID_hipModuleLoadData:
      oss << "hipModuleLoadData("
          << " module=" << data->args.hipModuleLoadData.module << ","
          << " image=" << data->args.hipModuleLoadData.image
          << ")";
    break;
    case HIP_API_ID_hipDeviceTotalMem:
      oss << "hipDeviceTotalMem("
          << " bytes=" << data->args.hipDeviceTotalMem.bytes << ","
          << " device=" << data->args.hipDeviceTotalMem.device
          << ")";
    break;
    case HIP_API_ID_hipCtxSetCurrent:
      oss << "hipCtxSetCurrent("
          << " ctx=" << data->args.hipCtxSetCurrent.ctx
          << ")";
    break;
    case HIP_API_ID_hipMallocHost:
      oss << "hipMallocHost("
          << " ptr=" << data->args.hipMallocHost.ptr << ","
          << " size=" << data->args.hipMallocHost.size
          << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxRetain:
      oss << "hipDevicePrimaryCtxRetain("
          << " pctx=" << data->args.hipDevicePrimaryCtxRetain.pctx << ","
          << " dev=" << data->args.hipDevicePrimaryCtxRetain.dev
          << ")";
    break;
    case HIP_API_ID_hipDeviceDisablePeerAccess:
      oss << "hipDeviceDisablePeerAccess("
          << " peerDeviceId=" << data->args.hipDeviceDisablePeerAccess.peerDeviceId
          << ")";
    break;
    case HIP_API_ID_hipStreamCreateWithFlags:
      oss << "hipStreamCreateWithFlags("
          << " stream=" << data->args.hipStreamCreateWithFlags.stream << ","
          << " flags=" << data->args.hipStreamCreateWithFlags.flags
          << ")";
    break;
    case HIP_API_ID_hipMemcpyFromArray:
      oss << "hipMemcpyFromArray("
          << " dst=" << data->args.hipMemcpyFromArray.dst << ","
          << " srcArray=" << data->args.hipMemcpyFromArray.srcArray << ","
          << " wOffset=" << data->args.hipMemcpyFromArray.wOffset << ","
          << " hOffset=" << data->args.hipMemcpyFromArray.hOffset << ","
          << " count=" << data->args.hipMemcpyFromArray.count << ","
          << " kind=" << data->args.hipMemcpyFromArray.kind
          << ")";
    break;
    case HIP_API_ID_hipMemcpy2DAsync:
      oss << "hipMemcpy2DAsync("
          << " dst=" << data->args.hipMemcpy2DAsync.dst << ","
          << " dpitch=" << data->args.hipMemcpy2DAsync.dpitch << ","
          << " src=" << data->args.hipMemcpy2DAsync.src << ","
          << " spitch=" << data->args.hipMemcpy2DAsync.spitch << ","
          << " width=" << data->args.hipMemcpy2DAsync.width << ","
          << " height=" << data->args.hipMemcpy2DAsync.height << ","
          << " kind=" << data->args.hipMemcpy2DAsync.kind << ","
          << " stream=" << data->args.hipMemcpy2DAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipFuncGetAttributes:
      oss << "hipFuncGetAttributes("
          << " attr=" << data->args.hipFuncGetAttributes.attr << ","
          << " func=" << data->args.hipFuncGetAttributes.func
          << ")";
    break;
    case HIP_API_ID_hipEventCreateWithFlags:
      oss << "hipEventCreateWithFlags("
          << " event=" << data->args.hipEventCreateWithFlags.event << ","
          << " flags=" << data->args.hipEventCreateWithFlags.flags
          << ")";
    break;
    case HIP_API_ID_hipStreamQuery:
      oss << "hipStreamQuery("
          << " stream=" << data->args.hipStreamQuery.stream
          << ")";
    break;
    case HIP_API_ID_hipDeviceGetPCIBusId:
      oss << "hipDeviceGetPCIBusId("
          << " pciBusId=" << data->args.hipDeviceGetPCIBusId.pciBusId << ","
          << " len=" << data->args.hipDeviceGetPCIBusId.len << ","
          << " device=" << data->args.hipDeviceGetPCIBusId.device
          << ")";
    break;
    case HIP_API_ID_hipMemcpy:
      oss << "hipMemcpy("
          << " dst=" << data->args.hipMemcpy.dst << ","
          << " src=" << data->args.hipMemcpy.src << ","
          << " sizeBytes=" << data->args.hipMemcpy.sizeBytes << ","
          << " kind=" << data->args.hipMemcpy.kind
          << ")";
    break;
    case HIP_API_ID_hipPeekAtLastError:
      oss << "hipPeekAtLastError("
          << ")";
    break;
    case HIP_API_ID_hipHostAlloc:
      oss << "hipHostAlloc("
          << " ptr=" << data->args.hipHostAlloc.ptr << ","
          << " size=" << data->args.hipHostAlloc.size << ","
          << " flags=" << data->args.hipHostAlloc.flags
          << ")";
    break;
    case HIP_API_ID_hipStreamAddCallback:
      oss << "hipStreamAddCallback("
          << " stream=" << data->args.hipStreamAddCallback.stream << ","
          << " callback=" << data->args.hipStreamAddCallback.callback << ","
          << " userData=" << data->args.hipStreamAddCallback.userData << ","
          << " flags=" << data->args.hipStreamAddCallback.flags
          << ")";
    break;
    case HIP_API_ID_hipMemcpyToArray:
      oss << "hipMemcpyToArray("
          << " dst=" << data->args.hipMemcpyToArray.dst << ","
          << " wOffset=" << data->args.hipMemcpyToArray.wOffset << ","
          << " hOffset=" << data->args.hipMemcpyToArray.hOffset << ","
          << " src=" << data->args.hipMemcpyToArray.src << ","
          << " count=" << data->args.hipMemcpyToArray.count << ","
          << " kind=" << data->args.hipMemcpyToArray.kind
          << ")";
    break;
    case HIP_API_ID_hipDeviceSynchronize:
      oss << "hipDeviceSynchronize("
          << ")";
    break;
    case HIP_API_ID_hipDeviceGetCacheConfig:
      oss << "hipDeviceGetCacheConfig("
          << " cacheConfig=" << data->args.hipDeviceGetCacheConfig.cacheConfig
          << ")";
    break;
    case HIP_API_ID_hipMalloc3D:
      oss << "hipMalloc3D("
          << " pitchedDevPtr=" << data->args.hipMalloc3D.pitchedDevPtr << ","
          << " extent=" << data->args.hipMalloc3D.extent
          << ")";
    break;
    case HIP_API_ID_hipPointerGetAttributes:
      oss << "hipPointerGetAttributes("
          << " attributes=" << data->args.hipPointerGetAttributes.attributes << ","
          << " ptr=" << data->args.hipPointerGetAttributes.ptr
          << ")";
    break;
    case HIP_API_ID_hipMemsetAsync:
      oss << "hipMemsetAsync("
          << " dst=" << data->args.hipMemsetAsync.dst << ","
          << " value=" << data->args.hipMemsetAsync.value << ","
          << " sizeBytes=" << data->args.hipMemsetAsync.sizeBytes << ","
          << " stream=" << data->args.hipMemsetAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipMemcpyToSymbol:
      oss << "hipMemcpyToSymbol("
          << " symbolName=" << data->args.hipMemcpyToSymbol.symbolName << ","
          << " src=" << data->args.hipMemcpyToSymbol.src << ","
          << " sizeBytes=" << data->args.hipMemcpyToSymbol.sizeBytes << ","
          << " offset=" << data->args.hipMemcpyToSymbol.offset << ","
          << " kind=" << data->args.hipMemcpyToSymbol.kind
          << ")";
    break;
    case HIP_API_ID_hipCtxPushCurrent:
      oss << "hipCtxPushCurrent("
          << " ctx=" << data->args.hipCtxPushCurrent.ctx
          << ")";
    break;
    case HIP_API_ID_hipMemcpyPeer:
      oss << "hipMemcpyPeer("
          << " dst=" << data->args.hipMemcpyPeer.dst << ","
          << " dstDeviceId=" << data->args.hipMemcpyPeer.dstDeviceId << ","
          << " src=" << data->args.hipMemcpyPeer.src << ","
          << " srcDeviceId=" << data->args.hipMemcpyPeer.srcDeviceId << ","
          << " sizeBytes=" << data->args.hipMemcpyPeer.sizeBytes
          << ")";
    break;
    case HIP_API_ID_hipEventSynchronize:
      oss << "hipEventSynchronize("
          << " event=" << data->args.hipEventSynchronize.event
          << ")";
    break;
    case HIP_API_ID_hipMemcpyDtoDAsync:
      oss << "hipMemcpyDtoDAsync("
          << " dst=" << data->args.hipMemcpyDtoDAsync.dst << ","
          << " src=" << data->args.hipMemcpyDtoDAsync.src << ","
          << " sizeBytes=" << data->args.hipMemcpyDtoDAsync.sizeBytes << ","
          << " stream=" << data->args.hipMemcpyDtoDAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipCtxEnablePeerAccess:
      oss << "hipCtxEnablePeerAccess("
          << " peerCtx=" << data->args.hipCtxEnablePeerAccess.peerCtx << ","
          << " flags=" << data->args.hipCtxEnablePeerAccess.flags
          << ")";
    break;
    case HIP_API_ID_hipMemcpyDtoHAsync:
      oss << "hipMemcpyDtoHAsync("
          << " dst=" << data->args.hipMemcpyDtoHAsync.dst << ","
          << " src=" << data->args.hipMemcpyDtoHAsync.src << ","
          << " sizeBytes=" << data->args.hipMemcpyDtoHAsync.sizeBytes << ","
          << " stream=" << data->args.hipMemcpyDtoHAsync.stream
          << ")";
    break;
    case HIP_API_ID_hipModuleLaunchKernel:
      oss << "hipModuleLaunchKernel("
          << " f=" << data->args.hipModuleLaunchKernel.f << ","
          << " gridDimX=" << data->args.hipModuleLaunchKernel.gridDimX << ","
          << " gridDimY=" << data->args.hipModuleLaunchKernel.gridDimY << ","
          << " gridDimZ=" << data->args.hipModuleLaunchKernel.gridDimZ << ","
          << " blockDimX=" << data->args.hipModuleLaunchKernel.blockDimX << ","
          << " blockDimY=" << data->args.hipModuleLaunchKernel.blockDimY << ","
          << " blockDimZ=" << data->args.hipModuleLaunchKernel.blockDimZ << ","
          << " sharedMemBytes=" << data->args.hipModuleLaunchKernel.sharedMemBytes << ","
          << " stream=" << data->args.hipModuleLaunchKernel.stream << ","
          << " kernelParams=" << data->args.hipModuleLaunchKernel.kernelParams << ","
          << " extra=" << data->args.hipModuleLaunchKernel.extra
          << ")";
    break;
    case HIP_API_ID_hipHccModuleLaunchKernel:
      oss << "hipHccModuleLaunchKernel("
          << " f=" << data->args.hipHccModuleLaunchKernel.f << ","
          << ")";
    break;
    case HIP_API_ID_hipModuleGetTexRef:
      oss << "hipModuleGetTexRef("
          << " texRef=" << data->args.hipModuleGetTexRef.texRef << ","
          << " hmod=" << data->args.hipModuleGetTexRef.hmod << ","
          << " name=" << data->args.hipModuleGetTexRef.name
          << ")";
    break;
    case HIP_API_ID_hipRemoveActivityCallback:
      oss << "hipRemoveActivityCallback("
          << " id=" << data->args.hipRemoveActivityCallback.id
          << ")";
    break;
    case HIP_API_ID_hipDeviceGetLimit:
      oss << "hipDeviceGetLimit("
          << " pValue=" << data->args.hipDeviceGetLimit.pValue << ","
          << " limit=" << data->args.hipDeviceGetLimit.limit
          << ")";
    break;
    case HIP_API_ID_hipModuleLoadDataEx:
      oss << "hipModuleLoadDataEx("
          << " module=" << data->args.hipModuleLoadDataEx.module << ","
          << " image=" << data->args.hipModuleLoadDataEx.image << ","
          << " numOptions=" << data->args.hipModuleLoadDataEx.numOptions << ","
          << " options=" << data->args.hipModuleLoadDataEx.options << ","
          << " optionValues=" << data->args.hipModuleLoadDataEx.optionValues
          << ")";
    break;
    case HIP_API_ID_hipRuntimeGetVersion:
      oss << "hipRuntimeGetVersion("
          << " runtimeVersion=" << data->args.hipRuntimeGetVersion.runtimeVersion
          << ")";
    break;
    case HIP_API_ID_hipGetDeviceProperties:
      oss << "hipGetDeviceProperties("
          << " prop=" << data->args.hipGetDeviceProperties.prop << ","
          << " deviceId=" << data->args.hipGetDeviceProperties.deviceId
          << ")";
    break;
    case HIP_API_ID_hipFreeArray:
      oss << "hipFreeArray("
          << " array=" << data->args.hipFreeArray.array
          << ")";
    break;
    case HIP_API_ID_hipDevicePrimaryCtxRelease:
      oss << "hipDevicePrimaryCtxRelease("
          << " dev=" << data->args.hipDevicePrimaryCtxRelease.dev
          << ")";
    break;
    case HIP_API_ID_hipHostGetDevicePointer:
      oss << "hipHostGetDevicePointer("
          << " devPtr=" << data->args.hipHostGetDevicePointer.devPtr << ","
          << " hstPtr=" << data->args.hipHostGetDevicePointer.hstPtr << ","
          << " flags=" << data->args.hipHostGetDevicePointer.flags
          << ")";
    break;
    case HIP_API_ID_hipMemcpyParam2D:
      oss << "hipMemcpyParam2D("
          << " pCopy=" << data->args.hipMemcpyParam2D.pCopy
          << ")";
    break;
    case HIP_API_ID_hipConfigureCall:
      oss << "hipConfigureCall("
          << " gridDim=" << data->args.hipConfigureCall.gridDim << ","
          << " blockDim=" << data->args.hipConfigureCall.blockDim << ","
          << " sharedMem=" << data->args.hipConfigureCall.sharedMem << ","
          << " stream=" << data->args.hipConfigureCall.stream
          << ")";
    break;
    case HIP_API_ID_hipModuleGetFunction:
      oss << "hipModuleGetFunction("
          << " function=" << data->args.hipModuleGetFunction.function << ","
          << " module=" << data->args.hipModuleGetFunction.module << ","
          << " kname=" << data->args.hipModuleGetFunction.kname
          << ")";
    break;
    case HIP_API_ID_hipGetDevice:
      oss << "hipGetDevice("
          << " deviceId=" << data->args.hipGetDevice.deviceId
          << ")";
    break;
    case HIP_API_ID_hipGetDeviceCount:
      oss << "hipGetDeviceCount("
          << " count=" << data->args.hipGetDeviceCount.count
          << ")";
    break;
    default: oss << "unknown";
  };
  return strdup(oss.str().c_str());
};
#endif

#endif  // _HIP_CBSTR

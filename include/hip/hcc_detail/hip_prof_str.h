// automatically generated sources
#ifndef _HIP_PROF_STR_H
#define _HIP_PROF_STR_H
#include <sstream>
#include <string>

// Dummy API primitives
#define INIT_NONE_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyToSymbolAsync_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyFromSymbolAsync_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDestroySurfaceObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddress_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectTextureDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTexture2D_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddressMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipCreateTextureObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTextureToMipmappedArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTextureToArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFormat_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyFromSymbol_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyHtoH_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureReference_CB_ARGS_DATA(cb_data) {};
#define INIT_hipDestroyTextureObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetArray_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetAddress2D_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectResourceViewDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipUnbindTexture_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFilterMode_CB_ARGS_DATA(cb_data) {};
#define INIT_hipCreateSurfaceObject_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetChannelDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipHccGetAcceleratorView_CB_ARGS_DATA(cb_data) {};
#define INIT_hipExtModuleLaunchKernel_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureObjectResourceDesc_CB_ARGS_DATA(cb_data) {};
#define INIT_hipMemcpyToSymbol_CB_ARGS_DATA(cb_data) {};
#define INIT_hipGetTextureAlignmentOffset_CB_ARGS_DATA(cb_data) {};
#define INIT_hipTexRefSetFlags_CB_ARGS_DATA(cb_data) {};
#define INIT_hipBindTexture_CB_ARGS_DATA(cb_data) {};
#define INIT_hipHccGetAccelerator_CB_ARGS_DATA(cb_data) {};

// HIP API callbacks ID enumaration
enum hip_api_id_t {
  HIP_API_ID_hipStreamCreateWithPriority = 0,
  HIP_API_ID_hipMallocPitch = 1,
  HIP_API_ID_hipMalloc = 2,
  HIP_API_ID_hipEventRecord = 3,
  HIP_API_ID_hipCtxSynchronize = 4,
  HIP_API_ID_hipSetDevice = 5,
  HIP_API_ID_hipCtxGetApiVersion = 6,
  HIP_API_ID_hipSetupArgument = 7,
  HIP_API_ID_hipMemcpyDtoD = 8,
  HIP_API_ID_hipHostFree = 9,
  HIP_API_ID_hipMemcpy2DToArray = 10,
  HIP_API_ID_hipCtxGetCacheConfig = 11,
  HIP_API_ID_hipStreamWaitEvent = 12,
  HIP_API_ID_hipDeviceGetStreamPriorityRange = 13,
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
  HIP_API_ID_hipArrayCreate = 24,
  HIP_API_ID_hipStreamGetFlags = 25,
  HIP_API_ID_hipMallocArray = 26,
  HIP_API_ID_hipCtxGetSharedMemConfig = 27,
  HIP_API_ID_hipMemPtrGetInfo = 28,
  HIP_API_ID_hipCtxGetFlags = 29,
  HIP_API_ID_hipStreamDestroy = 30,
  HIP_API_ID_hipMemset3DAsync = 31,
  HIP_API_ID_hipMemcpy3D = 32,
  HIP_API_ID_hipInit = 33,
  HIP_API_ID_hipMemcpyAtoH = 34,
  HIP_API_ID_hipStreamGetPriority = 35,
  HIP_API_ID_hipMemset2D = 36,
  HIP_API_ID_hipMemset2DAsync = 37,
  HIP_API_ID_hipDeviceCanAccessPeer = 38,
  HIP_API_ID_hipDeviceEnablePeerAccess = 39,
  HIP_API_ID_hipModuleUnload = 40,
  HIP_API_ID_hipHostUnregister = 41,
  HIP_API_ID_hipProfilerStop = 42,
  HIP_API_ID_hipLaunchByPtr = 43,
  HIP_API_ID_hipStreamSynchronize = 44,
  HIP_API_ID_hipDeviceSetCacheConfig = 45,
  HIP_API_ID_hipGetErrorName = 46,
  HIP_API_ID_hipMemcpyHtoD = 47,
  HIP_API_ID_hipMemcpyHtoA = 48,
  HIP_API_ID_hipCtxCreate = 49,
  HIP_API_ID_hipMemcpy2D = 50,
  HIP_API_ID_hipIpcCloseMemHandle = 51,
  HIP_API_ID_hipChooseDevice = 52,
  HIP_API_ID_hipDeviceSetSharedMemConfig = 53,
  HIP_API_ID_hipDeviceComputeCapability = 54,
  HIP_API_ID_hipDeviceGet = 55,
  HIP_API_ID_hipProfilerStart = 56,
  HIP_API_ID_hipCtxSetCacheConfig = 57,
  HIP_API_ID_hipFuncSetCacheConfig = 58,
  HIP_API_ID_hipMemcpyPeerAsync = 59,
  HIP_API_ID_hipEventElapsedTime = 60,
  HIP_API_ID_hipDevicePrimaryCtxReset = 61,
  HIP_API_ID_hipEventDestroy = 62,
  HIP_API_ID_hipCtxPopCurrent = 63,
  HIP_API_ID_hipHostGetFlags = 64,
  HIP_API_ID_hipHostMalloc = 65,
  HIP_API_ID_hipDriverGetVersion = 66,
  HIP_API_ID_hipMemGetInfo = 67,
  HIP_API_ID_hipDeviceReset = 68,
  HIP_API_ID_hipMemset = 69,
  HIP_API_ID_hipMemsetD8 = 70,
  HIP_API_ID_hipHostRegister = 71,
  HIP_API_ID_hipCtxSetSharedMemConfig = 72,
  HIP_API_ID_hipArray3DCreate = 73,
  HIP_API_ID_hipIpcOpenMemHandle = 74,
  HIP_API_ID_hipGetLastError = 75,
  HIP_API_ID_hipCtxDestroy = 76,
  HIP_API_ID_hipDeviceGetSharedMemConfig = 77,
  HIP_API_ID_hipSetDeviceFlags = 78,
  HIP_API_ID_hipHccModuleLaunchKernel = 79,
  HIP_API_ID_hipFree = 80,
  HIP_API_ID_hipDeviceGetAttribute = 81,
  HIP_API_ID_hipMemcpyDtoH = 82,
  HIP_API_ID_hipCtxDisablePeerAccess = 83,
  HIP_API_ID_hipDeviceGetByPCIBusId = 84,
  HIP_API_ID_hipIpcGetMemHandle = 85,
  HIP_API_ID_hipMemcpyHtoDAsync = 86,
  HIP_API_ID_hipCtxGetDevice = 87,
  HIP_API_ID_hipMemset3D = 88,
  HIP_API_ID_hipModuleLoadData = 89,
  HIP_API_ID_hipDeviceTotalMem = 90,
  HIP_API_ID_hipCtxSetCurrent = 91,
  HIP_API_ID_hipGetErrorString = 92,
  HIP_API_ID_hipDevicePrimaryCtxRetain = 93,
  HIP_API_ID_hipDeviceDisablePeerAccess = 94,
  HIP_API_ID_hipStreamCreateWithFlags = 95,
  HIP_API_ID_hipMemcpyFromArray = 96,
  HIP_API_ID_hipMemcpy2DAsync = 97,
  HIP_API_ID_hipEventCreateWithFlags = 98,
  HIP_API_ID_hipStreamQuery = 99,
  HIP_API_ID_hipDeviceGetPCIBusId = 100,
  HIP_API_ID_hipMemcpy = 101,
  HIP_API_ID_hipPeekAtLastError = 102,
  HIP_API_ID_hipStreamAddCallback = 103,
  HIP_API_ID_hipMemcpyToArray = 104,
  HIP_API_ID_hipMemsetD32 = 105,
  HIP_API_ID_hipDeviceSynchronize = 106,
  HIP_API_ID_hipDeviceGetCacheConfig = 107,
  HIP_API_ID_hipMalloc3D = 108,
  HIP_API_ID_hipPointerGetAttributes = 109,
  HIP_API_ID_hipMemsetAsync = 110,
  HIP_API_ID_hipDeviceGetName = 111,
  HIP_API_ID_hipCtxPushCurrent = 112,
  HIP_API_ID_hipMemcpyPeer = 113,
  HIP_API_ID_hipEventSynchronize = 114,
  HIP_API_ID_hipMemcpyDtoDAsync = 115,
  HIP_API_ID_hipCtxEnablePeerAccess = 116,
  HIP_API_ID_hipMemcpyDtoHAsync = 117,
  HIP_API_ID_hipModuleLaunchKernel = 118,
  HIP_API_ID_hipModuleGetTexRef = 119,
  HIP_API_ID_hipDeviceGetLimit = 120,
  HIP_API_ID_hipModuleLoadDataEx = 121,
  HIP_API_ID_hipRuntimeGetVersion = 122,
  HIP_API_ID_hipGetDeviceProperties = 123,
  HIP_API_ID_hipFreeArray = 124,
  HIP_API_ID_hipDevicePrimaryCtxRelease = 125,
  HIP_API_ID_hipHostGetDevicePointer = 126,
  HIP_API_ID_hipMemcpyParam2D = 127,
  HIP_API_ID_hipModuleGetFunction = 128,
  HIP_API_ID_hipMemsetD32Async = 129,
  HIP_API_ID_hipGetDevice = 130,
  HIP_API_ID_hipGetDeviceCount = 131,
  HIP_API_ID_NUMBER = 132,
  HIP_API_ID_ANY = 133,

  HIP_API_ID_NONE = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyToSymbolAsync = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyFromSymbolAsync = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDestroySurfaceObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddress = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectTextureDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTexture2D = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddressMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipCreateTextureObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTextureToMipmappedArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTextureToArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFormat = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyFromSymbol = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyHtoH = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureReference = HIP_API_ID_NUMBER,
  HIP_API_ID_hipDestroyTextureObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetArray = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetAddress2D = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectResourceViewDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipUnbindTexture = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFilterMode = HIP_API_ID_NUMBER,
  HIP_API_ID_hipCreateSurfaceObject = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetChannelDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipHccGetAcceleratorView = HIP_API_ID_NUMBER,
  HIP_API_ID_hipExtModuleLaunchKernel = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureObjectResourceDesc = HIP_API_ID_NUMBER,
  HIP_API_ID_hipMemcpyToSymbol = HIP_API_ID_NUMBER,
  HIP_API_ID_hipGetTextureAlignmentOffset = HIP_API_ID_NUMBER,
  HIP_API_ID_hipTexRefSetFlags = HIP_API_ID_NUMBER,
  HIP_API_ID_hipBindTexture = HIP_API_ID_NUMBER,
  HIP_API_ID_hipHccGetAccelerator = HIP_API_ID_NUMBER,
};

// Return HIP API string
static const char* hip_api_name(const uint32_t& id) {
  switch(id) {
    case HIP_API_ID_hipStreamCreateWithPriority: return "hipStreamCreateWithPriority";
    case HIP_API_ID_hipMallocPitch: return "hipMallocPitch";
    case HIP_API_ID_hipMalloc: return "hipMalloc";
    case HIP_API_ID_hipEventRecord: return "hipEventRecord";
    case HIP_API_ID_hipCtxSynchronize: return "hipCtxSynchronize";
    case HIP_API_ID_hipSetDevice: return "hipSetDevice";
    case HIP_API_ID_hipCtxGetApiVersion: return "hipCtxGetApiVersion";
    case HIP_API_ID_hipSetupArgument: return "hipSetupArgument";
    case HIP_API_ID_hipMemcpyDtoD: return "hipMemcpyDtoD";
    case HIP_API_ID_hipHostFree: return "hipHostFree";
    case HIP_API_ID_hipMemcpy2DToArray: return "hipMemcpy2DToArray";
    case HIP_API_ID_hipCtxGetCacheConfig: return "hipCtxGetCacheConfig";
    case HIP_API_ID_hipStreamWaitEvent: return "hipStreamWaitEvent";
    case HIP_API_ID_hipDeviceGetStreamPriorityRange: return "hipDeviceGetStreamPriorityRange";
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
    case HIP_API_ID_hipStreamGetPriority: return "hipStreamGetPriority";
    case HIP_API_ID_hipMemset2D: return "hipMemset2D";
    case HIP_API_ID_hipMemset2DAsync: return "hipMemset2DAsync";
    case HIP_API_ID_hipDeviceCanAccessPeer: return "hipDeviceCanAccessPeer";
    case HIP_API_ID_hipDeviceEnablePeerAccess: return "hipDeviceEnablePeerAccess";
    case HIP_API_ID_hipModuleUnload: return "hipModuleUnload";
    case HIP_API_ID_hipHostUnregister: return "hipHostUnregister";
    case HIP_API_ID_hipProfilerStop: return "hipProfilerStop";
    case HIP_API_ID_hipLaunchByPtr: return "hipLaunchByPtr";
    case HIP_API_ID_hipStreamSynchronize: return "hipStreamSynchronize";
    case HIP_API_ID_hipDeviceSetCacheConfig: return "hipDeviceSetCacheConfig";
    case HIP_API_ID_hipGetErrorName: return "hipGetErrorName";
    case HIP_API_ID_hipMemcpyHtoD: return "hipMemcpyHtoD";
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
    case HIP_API_ID_hipSetDeviceFlags: return "hipSetDeviceFlags";
    case HIP_API_ID_hipHccModuleLaunchKernel: return "hipHccModuleLaunchKernel";
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
    case HIP_API_ID_hipGetErrorString: return "hipGetErrorString";
    case HIP_API_ID_hipDevicePrimaryCtxRetain: return "hipDevicePrimaryCtxRetain";
    case HIP_API_ID_hipDeviceDisablePeerAccess: return "hipDeviceDisablePeerAccess";
    case HIP_API_ID_hipStreamCreateWithFlags: return "hipStreamCreateWithFlags";
    case HIP_API_ID_hipMemcpyFromArray: return "hipMemcpyFromArray";
    case HIP_API_ID_hipMemcpy2DAsync: return "hipMemcpy2DAsync";
    case HIP_API_ID_hipEventCreateWithFlags: return "hipEventCreateWithFlags";
    case HIP_API_ID_hipStreamQuery: return "hipStreamQuery";
    case HIP_API_ID_hipDeviceGetPCIBusId: return "hipDeviceGetPCIBusId";
    case HIP_API_ID_hipMemcpy: return "hipMemcpy";
    case HIP_API_ID_hipPeekAtLastError: return "hipPeekAtLastError";
    case HIP_API_ID_hipStreamAddCallback: return "hipStreamAddCallback";
    case HIP_API_ID_hipMemcpyToArray: return "hipMemcpyToArray";
    case HIP_API_ID_hipMemsetD32: return "hipMemsetD32";
    case HIP_API_ID_hipDeviceSynchronize: return "hipDeviceSynchronize";
    case HIP_API_ID_hipDeviceGetCacheConfig: return "hipDeviceGetCacheConfig";
    case HIP_API_ID_hipMalloc3D: return "hipMalloc3D";
    case HIP_API_ID_hipPointerGetAttributes: return "hipPointerGetAttributes";
    case HIP_API_ID_hipMemsetAsync: return "hipMemsetAsync";
    case HIP_API_ID_hipDeviceGetName: return "hipDeviceGetName";
    case HIP_API_ID_hipCtxPushCurrent: return "hipCtxPushCurrent";
    case HIP_API_ID_hipMemcpyPeer: return "hipMemcpyPeer";
    case HIP_API_ID_hipEventSynchronize: return "hipEventSynchronize";
    case HIP_API_ID_hipMemcpyDtoDAsync: return "hipMemcpyDtoDAsync";
    case HIP_API_ID_hipCtxEnablePeerAccess: return "hipCtxEnablePeerAccess";
    case HIP_API_ID_hipMemcpyDtoHAsync: return "hipMemcpyDtoHAsync";
    case HIP_API_ID_hipModuleLaunchKernel: return "hipModuleLaunchKernel";
    case HIP_API_ID_hipModuleGetTexRef: return "hipModuleGetTexRef";
    case HIP_API_ID_hipDeviceGetLimit: return "hipDeviceGetLimit";
    case HIP_API_ID_hipModuleLoadDataEx: return "hipModuleLoadDataEx";
    case HIP_API_ID_hipRuntimeGetVersion: return "hipRuntimeGetVersion";
    case HIP_API_ID_hipGetDeviceProperties: return "hipGetDeviceProperties";
    case HIP_API_ID_hipFreeArray: return "hipFreeArray";
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
struct hip_api_data_t {
  uint64_t correlation_id;
  uint32_t phase;
  union {
    struct {
      hipStream_t* stream;
      unsigned int flags;
      int priority;
    } hipStreamCreateWithPriority;
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
      char* name;
      int len;
      hipDevice_t device;
    } hipDeviceGetName;
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
      textureReference** texRef;
      hipModule_t hmod;
      const char* name;
    } hipModuleGetTexRef;
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
};

// HIP API callbacks args data filling macros
#define INIT_hipStreamCreateWithPriority_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreateWithPriority.stream = stream; \
  cb_data.args.hipStreamCreateWithPriority.flags = flags; \
  cb_data.args.hipStreamCreateWithPriority.priority = priority; \
};
#define INIT_hipMallocPitch_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocPitch.ptr = ptr; \
  cb_data.args.hipMallocPitch.pitch = pitch; \
  cb_data.args.hipMallocPitch.width = width; \
  cb_data.args.hipMallocPitch.height = height; \
};
#define INIT_hipMalloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc.ptr = ptr; \
  cb_data.args.hipMalloc.size = sizeBytes; \
};
#define INIT_hipEventRecord_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventRecord.event = event; \
  cb_data.args.hipEventRecord.stream = stream; \
};
#define INIT_hipCtxSynchronize_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipSetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetDevice.deviceId = deviceId; \
};
#define INIT_hipCtxGetApiVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetApiVersion.ctx = ctx; \
  cb_data.args.hipCtxGetApiVersion.apiVersion = apiVersion; \
};
#define INIT_hipSetupArgument_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetupArgument.arg = arg; \
  cb_data.args.hipSetupArgument.size = size; \
  cb_data.args.hipSetupArgument.offset = offset; \
};
#define INIT_hipMemcpyDtoD_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoD.dst = dst; \
  cb_data.args.hipMemcpyDtoD.src = src; \
  cb_data.args.hipMemcpyDtoD.sizeBytes = sizeBytes; \
};
#define INIT_hipHostFree_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostFree.ptr = ptr; \
};
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
#define INIT_hipCtxGetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetCacheConfig.cacheConfig = cacheConfig; \
};
#define INIT_hipStreamWaitEvent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamWaitEvent.stream = stream; \
  cb_data.args.hipStreamWaitEvent.event = event; \
  cb_data.args.hipStreamWaitEvent.flags = flags; \
};
#define INIT_hipDeviceGetStreamPriorityRange_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetStreamPriorityRange.leastPriority = leastPriority; \
  cb_data.args.hipDeviceGetStreamPriorityRange.greatestPriority = greatestPriority; \
};
#define INIT_hipModuleLoad_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoad.module = module; \
  cb_data.args.hipModuleLoad.fname = fname; \
};
#define INIT_hipDevicePrimaryCtxSetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxSetFlags.dev = dev; \
  cb_data.args.hipDevicePrimaryCtxSetFlags.flags = flags; \
};
#define INIT_hipMemcpyAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyAsync.dst = dst; \
  cb_data.args.hipMemcpyAsync.src = src; \
  cb_data.args.hipMemcpyAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyAsync.kind = kind; \
  cb_data.args.hipMemcpyAsync.stream = stream; \
};
#define INIT_hipMalloc3DArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc3DArray.array = array; \
  cb_data.args.hipMalloc3DArray.desc = desc; \
  cb_data.args.hipMalloc3DArray.extent = extent; \
  cb_data.args.hipMalloc3DArray.flags = flags; \
};
#define INIT_hipStreamCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreate.stream = stream; \
};
#define INIT_hipCtxGetCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetCurrent.ctx = ctx; \
};
#define INIT_hipDevicePrimaryCtxGetState_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxGetState.dev = dev; \
  cb_data.args.hipDevicePrimaryCtxGetState.flags = flags; \
  cb_data.args.hipDevicePrimaryCtxGetState.active = active; \
};
#define INIT_hipEventQuery_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventQuery.event = event; \
};
#define INIT_hipEventCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventCreate.event = event; \
};
#define INIT_hipMemGetAddressRange_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemGetAddressRange.pbase = pbase; \
  cb_data.args.hipMemGetAddressRange.psize = psize; \
  cb_data.args.hipMemGetAddressRange.dptr = dptr; \
};
#define INIT_hipArrayCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipArrayCreate.pHandle = array; \
  cb_data.args.hipArrayCreate.pAllocateArray = pAllocateArray; \
};
#define INIT_hipStreamGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamGetFlags.stream = stream; \
  cb_data.args.hipStreamGetFlags.flags = flags; \
};
#define INIT_hipMallocArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMallocArray.array = array; \
  cb_data.args.hipMallocArray.desc = desc; \
  cb_data.args.hipMallocArray.width = width; \
  cb_data.args.hipMallocArray.height = height; \
  cb_data.args.hipMallocArray.flags = flags; \
};
#define INIT_hipCtxGetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetSharedMemConfig.pConfig = pConfig; \
};
#define INIT_hipMemPtrGetInfo_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemPtrGetInfo.ptr = ptr; \
  cb_data.args.hipMemPtrGetInfo.size = size; \
};
#define INIT_hipCtxGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetFlags.flags = flags; \
};
#define INIT_hipStreamDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamDestroy.stream = stream; \
};
#define INIT_hipMemset3DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset3DAsync.pitchedDevPtr = pitchedDevPtr; \
  cb_data.args.hipMemset3DAsync.value = value; \
  cb_data.args.hipMemset3DAsync.extent = extent; \
  cb_data.args.hipMemset3DAsync.stream = stream; \
};
#define INIT_hipMemcpy3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy3D.p = p; \
};
#define INIT_hipInit_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipInit.flags = flags; \
};
#define INIT_hipMemcpyAtoH_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyAtoH.dst = dst; \
  cb_data.args.hipMemcpyAtoH.srcArray = srcArray; \
  cb_data.args.hipMemcpyAtoH.srcOffset = srcOffset; \
  cb_data.args.hipMemcpyAtoH.count = count; \
};
#define INIT_hipStreamGetPriority_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamGetPriority.stream = stream; \
  cb_data.args.hipStreamGetPriority.priority = priority; \
};
#define INIT_hipMemset2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset2D.dst = dst; \
  cb_data.args.hipMemset2D.pitch = pitch; \
  cb_data.args.hipMemset2D.value = value; \
  cb_data.args.hipMemset2D.width = width; \
  cb_data.args.hipMemset2D.height = height; \
};
#define INIT_hipMemset2DAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset2DAsync.dst = dst; \
  cb_data.args.hipMemset2DAsync.pitch = pitch; \
  cb_data.args.hipMemset2DAsync.value = value; \
  cb_data.args.hipMemset2DAsync.width = width; \
  cb_data.args.hipMemset2DAsync.height = height; \
  cb_data.args.hipMemset2DAsync.stream = stream; \
};
#define INIT_hipDeviceCanAccessPeer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceCanAccessPeer.canAccessPeer = canAccessPeer; \
  cb_data.args.hipDeviceCanAccessPeer.deviceId = deviceId; \
  cb_data.args.hipDeviceCanAccessPeer.peerDeviceId = peerDeviceId; \
};
#define INIT_hipDeviceEnablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceEnablePeerAccess.peerDeviceId = peerDeviceId; \
  cb_data.args.hipDeviceEnablePeerAccess.flags = flags; \
};
#define INIT_hipModuleUnload_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleUnload.module = hmod; \
};
#define INIT_hipHostUnregister_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostUnregister.hostPtr = hostPtr; \
};
#define INIT_hipProfilerStop_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipLaunchByPtr_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipLaunchByPtr.func = hostFunction; \
};
#define INIT_hipStreamSynchronize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamSynchronize.stream = stream; \
};
#define INIT_hipDeviceSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceSetCacheConfig.cacheConfig = cacheConfig; \
};
#define INIT_hipGetErrorName_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetErrorName.hip_error = hip_error; \
};
#define INIT_hipMemcpyHtoD_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoD.dst = dst; \
  cb_data.args.hipMemcpyHtoD.src = src; \
  cb_data.args.hipMemcpyHtoD.sizeBytes = sizeBytes; \
};
#define INIT_hipMemcpyHtoA_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoA.dstArray = dstArray; \
  cb_data.args.hipMemcpyHtoA.dstOffset = dstOffset; \
  cb_data.args.hipMemcpyHtoA.srcHost = srcHost; \
  cb_data.args.hipMemcpyHtoA.count = count; \
};
#define INIT_hipCtxCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxCreate.ctx = ctx; \
  cb_data.args.hipCtxCreate.flags = flags; \
  cb_data.args.hipCtxCreate.device = device; \
};
#define INIT_hipMemcpy2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy2D.dst = dst; \
  cb_data.args.hipMemcpy2D.dpitch = dpitch; \
  cb_data.args.hipMemcpy2D.src = src; \
  cb_data.args.hipMemcpy2D.spitch = spitch; \
  cb_data.args.hipMemcpy2D.width = width; \
  cb_data.args.hipMemcpy2D.height = height; \
  cb_data.args.hipMemcpy2D.kind = kind; \
};
#define INIT_hipIpcCloseMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcCloseMemHandle.devPtr = devPtr; \
};
#define INIT_hipChooseDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipChooseDevice.device = device; \
  cb_data.args.hipChooseDevice.prop = prop; \
};
#define INIT_hipDeviceSetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceSetSharedMemConfig.config = config; \
};
#define INIT_hipDeviceComputeCapability_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceComputeCapability.major = major; \
  cb_data.args.hipDeviceComputeCapability.minor = minor; \
  cb_data.args.hipDeviceComputeCapability.device = device; \
};
#define INIT_hipDeviceGet_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGet.device = device; \
  cb_data.args.hipDeviceGet.ordinal = deviceId; \
};
#define INIT_hipProfilerStart_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipCtxSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetCacheConfig.cacheConfig = cacheConfig; \
};
#define INIT_hipFuncSetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFuncSetCacheConfig.func = func; \
  cb_data.args.hipFuncSetCacheConfig.config = cacheConfig; \
};
#define INIT_hipMemcpyPeerAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyPeerAsync.dst = dst; \
  cb_data.args.hipMemcpyPeerAsync.dstDeviceId = dstDevice; \
  cb_data.args.hipMemcpyPeerAsync.src = src; \
  cb_data.args.hipMemcpyPeerAsync.srcDevice = srcDevice; \
  cb_data.args.hipMemcpyPeerAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyPeerAsync.stream = stream; \
};
#define INIT_hipEventElapsedTime_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventElapsedTime.ms = ms; \
  cb_data.args.hipEventElapsedTime.start = start; \
  cb_data.args.hipEventElapsedTime.stop = stop; \
};
#define INIT_hipDevicePrimaryCtxReset_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxReset.dev = dev; \
};
#define INIT_hipEventDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventDestroy.event = event; \
};
#define INIT_hipCtxPopCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxPopCurrent.ctx = ctx; \
};
#define INIT_hipHostGetFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostGetFlags.flagsPtr = flagsPtr; \
  cb_data.args.hipHostGetFlags.hostPtr = hostPtr; \
};
#define INIT_hipHostMalloc_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostMalloc.ptr = ptr; \
  cb_data.args.hipHostMalloc.size = sizeBytes; \
  cb_data.args.hipHostMalloc.flags = flags; \
};
#define INIT_hipDriverGetVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDriverGetVersion.driverVersion = driverVersion; \
};
#define INIT_hipMemGetInfo_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemGetInfo.free = free; \
  cb_data.args.hipMemGetInfo.total = total; \
};
#define INIT_hipDeviceReset_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipMemset_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset.dst = dst; \
  cb_data.args.hipMemset.value = value; \
  cb_data.args.hipMemset.sizeBytes = sizeBytes; \
};
#define INIT_hipMemsetD8_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD8.dest = dst; \
  cb_data.args.hipMemsetD8.value = value; \
  cb_data.args.hipMemsetD8.sizeBytes = sizeBytes; \
};
#define INIT_hipHostRegister_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostRegister.hostPtr = hostPtr; \
  cb_data.args.hipHostRegister.sizeBytes = sizeBytes; \
  cb_data.args.hipHostRegister.flags = flags; \
};
#define INIT_hipCtxSetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetSharedMemConfig.config = config; \
};
#define INIT_hipArray3DCreate_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipArray3DCreate.array = array; \
  cb_data.args.hipArray3DCreate.pAllocateArray = pAllocateArray; \
};
#define INIT_hipIpcOpenMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcOpenMemHandle.devPtr = devPtr; \
  cb_data.args.hipIpcOpenMemHandle.handle = handle; \
  cb_data.args.hipIpcOpenMemHandle.flags = flags; \
};
#define INIT_hipGetLastError_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipCtxDestroy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxDestroy.ctx = ctx; \
};
#define INIT_hipDeviceGetSharedMemConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetSharedMemConfig.pConfig = pConfig; \
};
#define INIT_hipSetDeviceFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipSetDeviceFlags.flags = flags; \
};
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
#define INIT_hipFree_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFree.ptr = ptr; \
};
#define INIT_hipDeviceGetAttribute_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetAttribute.pi = pi; \
  cb_data.args.hipDeviceGetAttribute.attr = attr; \
  cb_data.args.hipDeviceGetAttribute.deviceId = device; \
};
#define INIT_hipMemcpyDtoH_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoH.dst = dst; \
  cb_data.args.hipMemcpyDtoH.src = src; \
  cb_data.args.hipMemcpyDtoH.sizeBytes = sizeBytes; \
};
#define INIT_hipCtxDisablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxDisablePeerAccess.peerCtx = peerCtx; \
};
#define INIT_hipDeviceGetByPCIBusId_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetByPCIBusId.device = device; \
  cb_data.args.hipDeviceGetByPCIBusId.pciBusId = pciBusId; \
};
#define INIT_hipIpcGetMemHandle_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipIpcGetMemHandle.handle = handle; \
  cb_data.args.hipIpcGetMemHandle.devPtr = devPtr; \
};
#define INIT_hipMemcpyHtoDAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyHtoDAsync.dst = dst; \
  cb_data.args.hipMemcpyHtoDAsync.src = src; \
  cb_data.args.hipMemcpyHtoDAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyHtoDAsync.stream = stream; \
};
#define INIT_hipCtxGetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxGetDevice.device = device; \
};
#define INIT_hipMemset3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemset3D.pitchedDevPtr = pitchedDevPtr; \
  cb_data.args.hipMemset3D.value = value; \
  cb_data.args.hipMemset3D.extent = extent; \
};
#define INIT_hipModuleLoadData_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoadData.module = module; \
  cb_data.args.hipModuleLoadData.image = image; \
};
#define INIT_hipDeviceTotalMem_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceTotalMem.bytes = bytes; \
  cb_data.args.hipDeviceTotalMem.device = device; \
};
#define INIT_hipCtxSetCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxSetCurrent.ctx = ctx; \
};
#define INIT_hipGetErrorString_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetErrorString.hipError = hip_error; \
};
#define INIT_hipDevicePrimaryCtxRetain_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxRetain.pctx = pctx; \
  cb_data.args.hipDevicePrimaryCtxRetain.dev = dev; \
};
#define INIT_hipDeviceDisablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceDisablePeerAccess.peerDeviceId = peerDeviceId; \
};
#define INIT_hipStreamCreateWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamCreateWithFlags.stream = stream; \
  cb_data.args.hipStreamCreateWithFlags.flags = flags; \
};
#define INIT_hipMemcpyFromArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyFromArray.dst = dst; \
  cb_data.args.hipMemcpyFromArray.srcArray = srcArray; \
  cb_data.args.hipMemcpyFromArray.wOffset = wOffset; \
  cb_data.args.hipMemcpyFromArray.hOffset = hOffset; \
  cb_data.args.hipMemcpyFromArray.count = count; \
  cb_data.args.hipMemcpyFromArray.kind = kind; \
};
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
#define INIT_hipEventCreateWithFlags_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventCreateWithFlags.event = event; \
  cb_data.args.hipEventCreateWithFlags.flags = flags; \
};
#define INIT_hipStreamQuery_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamQuery.stream = stream; \
};
#define INIT_hipDeviceGetPCIBusId_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetPCIBusId.pciBusId = pciBusId; \
  cb_data.args.hipDeviceGetPCIBusId.len = len; \
  cb_data.args.hipDeviceGetPCIBusId.device = device; \
};
#define INIT_hipMemcpy_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpy.dst = dst; \
  cb_data.args.hipMemcpy.src = src; \
  cb_data.args.hipMemcpy.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpy.kind = kind; \
};
#define INIT_hipPeekAtLastError_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipStreamAddCallback_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipStreamAddCallback.stream = stream; \
  cb_data.args.hipStreamAddCallback.callback = callback; \
  cb_data.args.hipStreamAddCallback.userData = userData; \
  cb_data.args.hipStreamAddCallback.flags = flags; \
};
#define INIT_hipMemcpyToArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyToArray.dst = dst; \
  cb_data.args.hipMemcpyToArray.wOffset = wOffset; \
  cb_data.args.hipMemcpyToArray.hOffset = hOffset; \
  cb_data.args.hipMemcpyToArray.src = src; \
  cb_data.args.hipMemcpyToArray.count = count; \
  cb_data.args.hipMemcpyToArray.kind = kind; \
};
#define INIT_hipMemsetD32_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD32.dest = dst; \
  cb_data.args.hipMemsetD32.value = value; \
  cb_data.args.hipMemsetD32.count = count; \
};
#define INIT_hipDeviceSynchronize_CB_ARGS_DATA(cb_data) { \
};
#define INIT_hipDeviceGetCacheConfig_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetCacheConfig.cacheConfig = cacheConfig; \
};
#define INIT_hipMalloc3D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMalloc3D.pitchedDevPtr = pitchedDevPtr; \
  cb_data.args.hipMalloc3D.extent = extent; \
};
#define INIT_hipPointerGetAttributes_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipPointerGetAttributes.attributes = attributes; \
  cb_data.args.hipPointerGetAttributes.ptr = ptr; \
};
#define INIT_hipMemsetAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetAsync.dst = dst; \
  cb_data.args.hipMemsetAsync.value = value; \
  cb_data.args.hipMemsetAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemsetAsync.stream = stream; \
};
#define INIT_hipDeviceGetName_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetName.name = name; \
  cb_data.args.hipDeviceGetName.len = len; \
  cb_data.args.hipDeviceGetName.device = device; \
};
#define INIT_hipCtxPushCurrent_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxPushCurrent.ctx = ctx; \
};
#define INIT_hipMemcpyPeer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyPeer.dst = dst; \
  cb_data.args.hipMemcpyPeer.dstDeviceId = dstDevice; \
  cb_data.args.hipMemcpyPeer.src = src; \
  cb_data.args.hipMemcpyPeer.srcDeviceId = srcDevice; \
  cb_data.args.hipMemcpyPeer.sizeBytes = sizeBytes; \
};
#define INIT_hipEventSynchronize_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipEventSynchronize.event = event; \
};
#define INIT_hipMemcpyDtoDAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoDAsync.dst = dst; \
  cb_data.args.hipMemcpyDtoDAsync.src = src; \
  cb_data.args.hipMemcpyDtoDAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyDtoDAsync.stream = stream; \
};
#define INIT_hipCtxEnablePeerAccess_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipCtxEnablePeerAccess.peerCtx = peerCtx; \
  cb_data.args.hipCtxEnablePeerAccess.flags = flags; \
};
#define INIT_hipMemcpyDtoHAsync_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyDtoHAsync.dst = dst; \
  cb_data.args.hipMemcpyDtoHAsync.src = src; \
  cb_data.args.hipMemcpyDtoHAsync.sizeBytes = sizeBytes; \
  cb_data.args.hipMemcpyDtoHAsync.stream = stream; \
};
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
#define INIT_hipModuleGetTexRef_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetTexRef.texRef = texRef; \
  cb_data.args.hipModuleGetTexRef.hmod = hmod; \
  cb_data.args.hipModuleGetTexRef.name = name; \
};
#define INIT_hipDeviceGetLimit_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDeviceGetLimit.pValue = pValue; \
  cb_data.args.hipDeviceGetLimit.limit = limit; \
};
#define INIT_hipModuleLoadDataEx_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleLoadDataEx.module = module; \
  cb_data.args.hipModuleLoadDataEx.image = image; \
  cb_data.args.hipModuleLoadDataEx.numOptions = numOptions; \
  cb_data.args.hipModuleLoadDataEx.options = options; \
  cb_data.args.hipModuleLoadDataEx.optionValues = optionValues; \
};
#define INIT_hipRuntimeGetVersion_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipRuntimeGetVersion.runtimeVersion = runtimeVersion; \
};
#define INIT_hipGetDeviceProperties_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDeviceProperties.prop = props; \
  cb_data.args.hipGetDeviceProperties.deviceId = device; \
};
#define INIT_hipFreeArray_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipFreeArray.array = array; \
};
#define INIT_hipDevicePrimaryCtxRelease_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipDevicePrimaryCtxRelease.dev = dev; \
};
#define INIT_hipHostGetDevicePointer_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipHostGetDevicePointer.devPtr = devicePointer; \
  cb_data.args.hipHostGetDevicePointer.hstPtr = hostPointer; \
  cb_data.args.hipHostGetDevicePointer.flags = flags; \
};
#define INIT_hipMemcpyParam2D_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemcpyParam2D.pCopy = pCopy; \
};
#define INIT_hipModuleGetFunction_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipModuleGetFunction.function = hfunc; \
  cb_data.args.hipModuleGetFunction.module = hmod; \
  cb_data.args.hipModuleGetFunction.kname = name; \
};
#define INIT_hipMemsetD32Async_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipMemsetD32Async.dst = dst; \
  cb_data.args.hipMemsetD32Async.value = value; \
  cb_data.args.hipMemsetD32Async.count = count; \
  cb_data.args.hipMemsetD32Async.stream = stream; \
};
#define INIT_hipGetDevice_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDevice.deviceId = deviceId; \
};
#define INIT_hipGetDeviceCount_CB_ARGS_DATA(cb_data) { \
  cb_data.args.hipGetDeviceCount.count = count; \
};
#define INIT_CB_ARGS_DATA(cb_id, cb_data) INIT_##cb_id##_CB_ARGS_DATA(cb_data)

#if 0
// HIP API string method, method name and parameters
const char* hipApiString(hip_api_id_t id, const hip_api_data_t* data) {
  std::ostringstream oss;
  switch (id) {
    case HIP_API_ID_hipStreamCreateWithPriority:
      oss << "hipStreamCreateWithPriority("
          << " stream=" << data->args.hipStreamCreateWithPriority.stream << ","
          << " flags=" << data->args.hipStreamCreateWithPriority.flags << ","
          << " priority=" << data->args.hipStreamCreateWithPriority.priority
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
    case HIP_API_ID_hipCtxGetApiVersion:
      oss << "hipCtxGetApiVersion("
          << " ctx=" << data->args.hipCtxGetApiVersion.ctx << ","
          << " apiVersion=" << data->args.hipCtxGetApiVersion.apiVersion
          << ")";
    break;
    case HIP_API_ID_hipSetupArgument:
      oss << "hipSetupArgument("
          << " arg=" << data->args.hipSetupArgument.arg << ","
          << " size=" << data->args.hipSetupArgument.size << ","
          << " offset=" << data->args.hipSetupArgument.offset
          << ")";
    break;
    case HIP_API_ID_hipMemcpyDtoD:
      oss << "hipMemcpyDtoD("
          << " dst=" << data->args.hipMemcpyDtoD.dst << ","
          << " src=" << data->args.hipMemcpyDtoD.src << ","
          << " sizeBytes=" << data->args.hipMemcpyDtoD.sizeBytes
          << ")";
    break;
    case HIP_API_ID_hipHostFree:
      oss << "hipHostFree("
          << " ptr=" << data->args.hipHostFree.ptr
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
    case HIP_API_ID_hipDeviceGetStreamPriorityRange:
      oss << "hipDeviceGetStreamPriorityRange("
          << " leastPriority=" << data->args.hipDeviceGetStreamPriorityRange.leastPriority << ","
          << " greatestPriority=" << data->args.hipDeviceGetStreamPriorityRange.greatestPriority
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
    case HIP_API_ID_hipStreamGetPriority:
      oss << "hipStreamGetPriority("
          << " stream=" << data->args.hipStreamGetPriority.stream << ","
          << " priority=" << data->args.hipStreamGetPriority.priority
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
    case HIP_API_ID_hipDeviceSetCacheConfig:
      oss << "hipDeviceSetCacheConfig("
          << " cacheConfig=" << data->args.hipDeviceSetCacheConfig.cacheConfig
          << ")";
    break;
    case HIP_API_ID_hipGetErrorName:
      oss << "hipGetErrorName("
          << " hip_error=" << data->args.hipGetErrorName.hip_error
          << ")";
    break;
    case HIP_API_ID_hipMemcpyHtoD:
      oss << "hipMemcpyHtoD("
          << " dst=" << data->args.hipMemcpyHtoD.dst << ","
          << " src=" << data->args.hipMemcpyHtoD.src << ","
          << " sizeBytes=" << data->args.hipMemcpyHtoD.sizeBytes
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
    case HIP_API_ID_hipSetDeviceFlags:
      oss << "hipSetDeviceFlags("
          << " flags=" << data->args.hipSetDeviceFlags.flags
          << ")";
    break;
    case HIP_API_ID_hipHccModuleLaunchKernel:
      oss << "hipHccModuleLaunchKernel("
          << " f=" << data->args.hipHccModuleLaunchKernel.f << ","
          << " globalWorkSizeX=" << data->args.hipHccModuleLaunchKernel.globalWorkSizeX << ","
          << " globalWorkSizeY=" << data->args.hipHccModuleLaunchKernel.globalWorkSizeY << ","
          << " globalWorkSizeZ=" << data->args.hipHccModuleLaunchKernel.globalWorkSizeZ << ","
          << " localWorkSizeX=" << data->args.hipHccModuleLaunchKernel.localWorkSizeX << ","
          << " localWorkSizeY=" << data->args.hipHccModuleLaunchKernel.localWorkSizeY << ","
          << " localWorkSizeZ=" << data->args.hipHccModuleLaunchKernel.localWorkSizeZ << ","
          << " sharedMemBytes=" << data->args.hipHccModuleLaunchKernel.sharedMemBytes << ","
          << " hStream=" << data->args.hipHccModuleLaunchKernel.hStream << ","
          << " kernelParams=" << data->args.hipHccModuleLaunchKernel.kernelParams << ","
          << " extra=" << data->args.hipHccModuleLaunchKernel.extra << ","
          << " startEvent=" << data->args.hipHccModuleLaunchKernel.startEvent << ","
          << " stopEvent=" << data->args.hipHccModuleLaunchKernel.stopEvent
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
    case HIP_API_ID_hipGetErrorString:
      oss << "hipGetErrorString("
          << " hipError=" << data->args.hipGetErrorString.hipError
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
    case HIP_API_ID_hipMemsetD32:
      oss << "hipMemsetD32("
          << " dest=" << data->args.hipMemsetD32.dest << ","
          << " value=" << data->args.hipMemsetD32.value << ","
          << " count=" << data->args.hipMemsetD32.count
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
    case HIP_API_ID_hipDeviceGetName:
      oss << "hipDeviceGetName("
          << " name=" << data->args.hipDeviceGetName.name << ","
          << " len=" << data->args.hipDeviceGetName.len << ","
          << " device=" << data->args.hipDeviceGetName.device
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
    case HIP_API_ID_hipModuleGetTexRef:
      oss << "hipModuleGetTexRef("
          << " texRef=" << data->args.hipModuleGetTexRef.texRef << ","
          << " hmod=" << data->args.hipModuleGetTexRef.hmod << ","
          << " name=" << data->args.hipModuleGetTexRef.name
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
    case HIP_API_ID_hipModuleGetFunction:
      oss << "hipModuleGetFunction("
          << " function=" << data->args.hipModuleGetFunction.function << ","
          << " module=" << data->args.hipModuleGetFunction.module << ","
          << " kname=" << data->args.hipModuleGetFunction.kname
          << ")";
    break;
    case HIP_API_ID_hipMemsetD32Async:
      oss << "hipMemsetD32Async("
          << " dst=" << data->args.hipMemsetD32Async.dst << ","
          << " value=" << data->args.hipMemsetD32Async.value << ","
          << " count=" << data->args.hipMemsetD32Async.count << ","
          << " stream=" << data->args.hipMemsetD32Async.stream
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
#endif  // _HIP_PROF_STR_H

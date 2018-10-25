#include "CUDA2HIP.h"

// Maps the names of CUDA RUNTIME API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_RUNTIME_TYPE_NAME_MAP {
  // 1. Structs
  // no analogue
  {"cudaChannelFormatDesc",                                            {"hipChannelFormatDesc",                                     CONV_TYPE, API_RUNTIME}},
  // no analogue
  {"cudaDeviceProp",                                                   {"hipDeviceProp_t",                                          CONV_TYPE, API_RUNTIME}},

  // no analogue
  {"cudaEglFrame",                                                     {"hipEglFrame",                                              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaEglFrame_st",                                                  {"hipEglFrame",                                              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // no analogue
  {"cudaEglPlaneDesc",                                                 {"hipEglPlaneDesc",                                          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaEglPlaneDesc_st",                                              {"hipEglPlaneDesc",                                          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // no analogue
  {"cudaExtent",                                                       {"hipExtent",                                                CONV_TYPE, API_RUNTIME}},

  // CUDA_EXTERNAL_MEMORY_BUFFER_DESC
  {"cudaExternalMemoryBufferDesc",                                     {"HIP_EXTERNAL_MEMORY_BUFFER_DESC",                          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_MEMORY_HANDLE_DESC
  {"cudaExternalMemoryHandleDesc",                                     {"HIP_EXTERNAL_MEMORY_HANDLE_DESC",                          CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC
  {"cudaExternalMemoryMipmappedArrayDesc",                             {"HIP_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC",                 CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC
  {"cudaExternalSemaphoreHandleDesc",                                  {"HIP_EXTERNAL_SEMAPHORE_HANDLE_DESC",                       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS
  {"cudaExternalSemaphoreSignalParams",                                {"HIP_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS",                     CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS
  {"cudaExternalSemaphoreWaitParams",                                  {"HIP_EXTERNAL_SEMAPHORE_WAIT_PARAMS",                       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // no analogue
  {"cudaFuncAttributes",                                               {"hipFuncAttributes",                                        CONV_TYPE, API_RUNTIME}},

  // CUDA_HOST_NODE_PARAMS
  {"cudaHostNodeParams",                                               {"HIP_HOST_NODE_PARAMS",                                     CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // CUipcEventHandle
  {"cudaIpcEventHandle_t",                                             {"ihipIpcEventHandle_t",                                     CONV_TYPE, API_RUNTIME}},
  // CUipcEventHandle_st
  {"cudaIpcEventHandle_st",                                            {"ihipIpcEventHandle_t",                                     CONV_TYPE, API_RUNTIME}},

  // CUipcMemHandle
  {"cudaIpcMemHandle_t",                                               {"hipIpcMemHandle_t",                                        CONV_TYPE, API_RUNTIME}},
  // CUipcMemHandle_st
  {"cudaIpcMemHandle_st",                                              {"hipIpcMemHandle_st",                                       CONV_TYPE, API_RUNTIME}},

  // CUDA_KERNEL_NODE_PARAMS
  {"cudaKernelNodeParams",                                             {"hipKernelNodeParams",                                     CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},


  // 3. Enums
  // CUresult
  {"cudaError",                                                        {"hipError_t",                                               CONV_TYPE, API_RUNTIME}},
  {"cudaError_t",                                                      {"hipError_t",                                               CONV_TYPE, API_RUNTIME}},
  // cudaError enum values
  // CUDA_SUCCESS = 0
  {"cudaSuccess",                                                      {"hipSuccess",                                               CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 0
  // no analogue
  {"cudaErrorMissingConfiguration",                                    {"hipErrorMissingConfiguration",                             CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 1
  // CUDA_ERROR_OUT_OF_MEMORY = 2
  {"cudaErrorMemoryAllocation",                                        {"hipErrorMemoryAllocation",                                 CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 2
  // CUDA_ERROR_NOT_INITIALIZED = 3
  {"cudaErrorInitializationError",                                     {"hipErrorInitializationError",                              CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 3
  // CUDA_ERROR_LAUNCH_FAILED = 719
  {"cudaErrorLaunchFailure",                                           {"hipErrorLaunchFailure",                                    CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 4
  // no analogue
  {"cudaErrorPriorLaunchFailure",                                      {"hipErrorPriorLaunchFailure",                               CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 5
  // CUDA_ERROR_LAUNCH_TIMEOUT = 702
  {"cudaErrorLaunchTimeout",                                           {"hipErrorLaunchTimeOut",                                    CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 6
  // CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
  {"cudaErrorLaunchOutOfResources",                                    {"hipErrorLaunchOutOfResources",                             CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 7
  // no analogue
  {"cudaErrorInvalidDeviceFunction",                                   {"hipErrorInvalidDeviceFunction",                            CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 8
  // no analogue
  {"cudaErrorInvalidConfiguration",                                    {"hipErrorInvalidConfiguration",                             CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 9
  // CUDA_ERROR_INVALID_DEVICE = 101
  {"cudaErrorInvalidDevice",                                           {"hipErrorInvalidDevice",                                    CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 10
  // CUDA_ERROR_INVALID_VALUE = 1
  {"cudaErrorInvalidValue",                                            {"hipErrorInvalidValue",                                     CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 11
  // no analogue
  {"cudaErrorInvalidPitchValue",                                       {"hipErrorInvalidPitchValue",                                CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 12
  // no analogue
  {"cudaErrorInvalidSymbol",                                           {"hipErrorInvalidSymbol",                                    CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 13
  // CUDA_ERROR_MAP_FAILED = 205
  // TODO: double check the matching
  {"cudaErrorMapBufferObjectFailed",                                   {"hipErrorMapFailed",                                        CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 14
  // CUDA_ERROR_UNMAP_FAILED = 206
  // TODO: double check the matching
  {"cudaErrorUnmapBufferObjectFailed",                                 {"hipErrorUnmapFailed",                                      CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 15
  // no analogue
  {"cudaErrorInvalidHostPointer",                                      {"hipErrorInvalidHostPointer",                               CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 16
  // no analogue
  {"cudaErrorInvalidDevicePointer",                                    {"hipErrorInvalidDevicePointer",                             CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 17
  // no analogue
  {"cudaErrorInvalidTexture",                                          {"hipErrorInvalidTexture",                                   CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 18
  // no analogue
  {"cudaErrorInvalidTextureBinding",                                   {"hipErrorInvalidTextureBinding",                            CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 19
  // no analogue
  {"cudaErrorInvalidChannelDescriptor",                                {"hipErrorInvalidChannelDescriptor",                         CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 20
  // no analogue
  {"cudaErrorInvalidMemcpyDirection",                                  {"hipErrorInvalidMemcpyDirection",                           CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 21
  // no analogue
  {"cudaErrorAddressOfConstant",                                       {"hipErrorAddressOfConstant",                                CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 22
  // no analogue
  {"cudaErrorTextureFetchFailed",                                      {"hipErrorTextureFetchFailed",                               CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 23
  // no analogue
  {"cudaErrorTextureNotBound",                                         {"hipErrorTextureNotBound",                                  CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 24
  // no analogue
  {"cudaErrorSynchronizationError",                                    {"hipErrorSynchronizationError",                             CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 25
  // no analogue
  {"cudaErrorInvalidFilterSetting",                                    {"hipErrorInvalidFilterSetting",                             CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 26
  // no analogue
  {"cudaErrorInvalidNormSetting",                                      {"hipErrorInvalidNormSetting",                               CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 27
  // no analogue
  {"cudaErrorMixedDeviceExecution",                                    {"hipErrorMixedDeviceExecution",                             CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 28
  // CUDA_ERROR_DEINITIALIZED = 4
  {"cudaErrorCudartUnloading",                                         {"hipErrorDeinitialized",                                    CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 29
  // CUDA_ERROR_UNKNOWN = 999
  {"cudaErrorUnknown",                                                 {"hipErrorUnknown",                                          CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 30
  // Deprecated since CUDA 4.1
  // no analogue
  {"cudaErrorNotYetImplemented",                                       {"hipErrorNotYetImplemented",                                CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 31
  // Deprecated since CUDA 3.1
  // no analogue
  {"cudaErrorMemoryValueTooLarge",                                     {"hipErrorMemoryValueTooLarge",                              CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 32
  // CUDA_ERROR_INVALID_HANDLE = 400
  {"cudaErrorInvalidResourceHandle",                                   {"hipErrorInvalidResourceHandle",                            CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 33
  // CUDA_ERROR_NOT_READY = 600
  {"cudaErrorNotReady",                                                {"hipErrorNotReady",                                         CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 34
  // no analogue
  {"cudaErrorInsufficientDriver",                                      {"hipErrorInsufficientDriver",                               CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 35
  // no analogue
  {"cudaErrorSetOnActiveProcess",                                      {"hipErrorSetOnActiveProcess",                               CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 36
  // no analogue
  {"cudaErrorInvalidSurface",                                          {"hipErrorInvalidSurface",                                   CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 37
  // CUDA_ERROR_NO_DEVICE = 100
  {"cudaErrorNoDevice",                                                {"hipErrorNoDevice",                                         CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 38
  // CUDA_ERROR_ECC_UNCORRECTABLE = 214
  {"cudaErrorECCUncorrectable",                                        {"hipErrorECCNotCorrectable",                                CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 39
  // CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
  {"cudaErrorSharedObjectSymbolNotFound",                              {"hipErrorSharedObjectSymbolNotFound",                       CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 40
  // CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
  {"cudaErrorSharedObjectInitFailed",                                  {"hipErrorSharedObjectInitFailed",                           CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 41
  // CUDA_ERROR_UNSUPPORTED_LIMIT = 215
  {"cudaErrorUnsupportedLimit",                                        {"hipErrorUnsupportedLimit",                                 CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 42
  // no analogue
  {"cudaErrorDuplicateVariableName",                                   {"hipErrorDuplicateVariableName",                            CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 43
  // no analogue
  {"cudaErrorDuplicateTextureName",                                    {"hipErrorDuplicateTextureName",                             CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 44
  // no analogue
  {"cudaErrorDuplicateSurfaceName",                                    {"hipErrorDuplicateSurfaceName",                             CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 45
  // no analogue
  {"cudaErrorDevicesUnavailable",                                      {"hipErrorDevicesUnavailable",                               CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 46
  // CUDA_ERROR_INVALID_IMAGE = 200
  {"cudaErrorInvalidKernelImage",                                      {"hipErrorInvalidImage",                                     CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 47
  // CUDA_ERROR_NO_BINARY_FOR_GPU = 209
  {"cudaErrorNoKernelImageForDevice",                                  {"hipErrorNoBinaryForGpu",                                   CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 48
  // no analogue
  {"cudaErrorIncompatibleDriverContext",                               {"hipErrorIncompatibleDriverContext",                        CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 49
  // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
  {"cudaErrorPeerAccessAlreadyEnabled",                                {"hipErrorPeerAccessAlreadyEnabled",                         CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 50
  // CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
  {"cudaErrorPeerAccessNotEnabled",                                    {"hipErrorPeerAccessNotEnabled",                             CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 51
  // no analogue
  {"cudaErrorDeviceAlreadyInUse",                                      {"hipErrorDeviceAlreadyInUse",                               CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 54
  // CUDA_ERROR_PROFILER_DISABLED = 5
  {"cudaErrorProfilerDisabled",                                        {"hipErrorProfilerDisabled",                                 CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 55
  // Deprecated since CUDA 5.0
  // CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
  {"cudaErrorProfilerNotInitialized",                                  {"hipErrorProfilerNotInitialized",                           CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 56
  // Deprecated since CUDA 5.0
  // CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
  {"cudaErrorProfilerAlreadyStarted",                                  {"hipErrorProfilerAlreadyStarted",                           CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 57
  // Deprecated since CUDA 5.0
  // CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
  {"cudaErrorProfilerAlreadyStopped",                                  {"hipErrorProfilerAlreadyStopped",                           CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 58
  // CUDA_ERROR_ASSERT = 710
  {"cudaErrorAssert",                                                  {"hipErrorAssert",                                           CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 59
  // CUDA_ERROR_TOO_MANY_PEERS = 711
  {"cudaErrorTooManyPeers",                                            {"hipErrorTooManyPeers",                                     CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 60
  // CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
  {"cudaErrorHostMemoryAlreadyRegistered",                             {"hipErrorHostMemoryAlreadyRegistered",                      CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 61
  // CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
  {"cudaErrorHostMemoryNotRegistered",                                 {"hipErrorHostMemoryNotRegistered",                          CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 62
  // CUDA_ERROR_OPERATING_SYSTEM = 304
  {"cudaErrorOperatingSystem",                                         {"hipErrorOperatingSystem",                                  CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 63
  // CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
  {"cudaErrorPeerAccessUnsupported",                                   {"hipErrorPeerAccessUnsupported",                            CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 64
  // no analogue
  {"cudaErrorLaunchMaxDepthExceeded",                                  {"hipErrorLaunchMaxDepthExceeded",                           CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 65
  // no analogue
  {"cudaErrorLaunchFileScopedTex",                                     {"hipErrorLaunchFileScopedTex",                              CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 66
  // no analogue
  {"cudaErrorLaunchFileScopedSurf",                                    {"hipErrorLaunchFileScopedSurf",                             CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 67
  // no analogue
  {"cudaErrorSyncDepthExceeded",                                       {"hipErrorSyncDepthExceeded",                                CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 68
  // no analogue
  {"cudaErrorLaunchPendingCountExceeded",                              {"hipErrorLaunchPendingCountExceeded",                       CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 69
  // CUDA_ERROR_NOT_PERMITTED = 800
  {"cudaErrorNotPermitted",                                            {"hipErrorNotPermitted",                                     CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 70
  // CUDA_ERROR_NOT_SUPPORTED = 801
  {"cudaErrorNotSupported",                                            {"hipErrorNotSupported",                                     CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 71
  // CUDA_ERROR_HARDWARE_STACK_ERROR = 714
  {"cudaErrorHardwareStackError",                                      {"hipErrorHardwareStackError",                               CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 72
  // CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
  {"cudaErrorIllegalInstruction",                                      {"hipErrorIllegalInstruction",                               CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 73
  // CUDA_ERROR_MISALIGNED_ADDRESS = 716
  {"cudaErrorMisalignedAddress",                                       {"hipErrorMisalignedAddress",                                CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 74
  // CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
  {"cudaErrorInvalidAddressSpace",                                     {"hipErrorInvalidAddressSpace",                              CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 75
  // CUDA_ERROR_INVALID_PC = 718
  {"cudaErrorInvalidPc",                                               {"hipErrorInvalidPc",                                        CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 76
  // CUDA_ERROR_ILLEGAL_ADDRESS = 700
  {"cudaErrorIllegalAddress",                                          {"hipErrorIllegalAddress",                                   CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 77
  // CUDA_ERROR_INVALID_PTX = 218
  {"cudaErrorInvalidPtx",                                              {"hipErrorInvalidKernelFile",                                CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 78
  // CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
  {"cudaErrorInvalidGraphicsContext",                                  {"hipErrorInvalidGraphicsContext",                           CONV_NUMERIC_LITERAL, API_RUNTIME}}, // 79
  // CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
  {"cudaErrorNvlinkUncorrectable",                                     {"hipErrorNvlinkUncorrectable",                              CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 80
  // no analogue
  {"cudaErrorJitCompilerNotFound",                                     {"hipErrorJitCompilerNotFound",                              CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 81
  // no analogue
  {"cudaErrorCooperativeLaunchTooLarge",                               {"hipErrorCooperativeLaunchTooLarge",                        CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 82
  // CUDA_ERROR_SYSTEM_NOT_READY = 802
  {"cudaErrorSystemNotReady",                                          {"hipErrorSystemNotReady",                                   CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 83
  // CUDA_ERROR_ILLEGAL_STATE = 401
  {"cudaErrorIllegalState",                                            {"hipErrorIllegalState",                                     CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 84
  // no analogue
  {"cudaErrorStartupFailure",                                          {"hipErrorStartupFailure",                                   CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 127
  // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
  {"cudaErrorStreamCaptureUnsupported",                                {"hipErrorStreamCaptureUnsupported",                         CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 900
  // CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901
  {"cudaErrorStreamCaptureInvalidated",                                {"hipErrorStreamCaptureInvalidated",                         CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 901
  // CUDA_ERROR_STREAM_CAPTURE_MERGE = 902
  {"cudaErrorStreamCaptureMerge",                                      {"hipErrorStreamCaptureMerge",                               CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 902
  // CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903
  {"cudaErrorStreamCaptureUnmatched",                                  {"hipErrorStreamCaptureUnmatched",                           CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 903
  // CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904
  {"cudaErrorStreamCaptureUnjoined",                                   {"hipErrorStreamCaptureUnjoined",                            CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 904
  // CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905
  {"cudaErrorStreamCaptureIsolation",                                  {"hipErrorStreamCaptureIsolation",                           CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 905
  // CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906
  {"cudaErrorStreamCaptureImplicit",                                   {"hipErrorStreamCaptureImplicit",                            CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 906
  // CUDA_ERROR_CAPTURED_EVENT = 907
  {"cudaErrorCapturedEvent",                                           {"hipErrorCapturedEvent",                                    CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 907
  // Deprecated since CUDA 4.1
  {"cudaErrorApiFailureBase",                                          {"hipErrorApiFailureBase",                                   CONV_NUMERIC_LITERAL, API_RUNTIME, HIP_UNSUPPORTED}}, // 10000


  {"libraryPropertyType_t", {"hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"libraryPropertyType",   {"hipLibraryPropertyType_t", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  {"cudaStreamCallback_t", {"hipStreamCallback_t", CONV_TYPE, API_RUNTIME}},

  // Arrays
  {"cudaArray",                  {"hipArray",                  CONV_TYPE, API_RUNTIME}},
  // typedef struct cudaArray *cudaArray_t;

  {"cudaArray_t",                {"hipArray_t",                CONV_TYPE, API_RUNTIME}},
  // typedef const struct cudaArray *cudaArray_const_t;

  {"cudaArray_const_t",          {"hipArray_const_t",          CONV_TYPE, API_RUNTIME}},
  {"cudaMipmappedArray_t",       {"hipMipmappedArray_t",       CONV_TYPE, API_RUNTIME}},
  {"cudaMipmappedArray_const_t", {"hipMipmappedArray_const_t", CONV_TYPE, API_RUNTIME}},

  // defines
  {"cudaArrayDefault",           {"hipArrayDefault",          CONV_TYPE, API_RUNTIME}},
  {"cudaArrayLayered",           {"hipArrayLayered",          CONV_TYPE, API_RUNTIME}},
  {"cudaArraySurfaceLoadStore",  {"hipArraySurfaceLoadStore", CONV_TYPE, API_RUNTIME}},
  {"cudaArrayCubemap",           {"hipArrayCubemap",          CONV_TYPE, API_RUNTIME}},
  {"cudaArrayTextureGather",     {"hipArrayTextureGather",    CONV_TYPE, API_RUNTIME}},

  {"cudaMemoryAdvise", {"hipMemAdvise", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUmem_advise)
  {"cudaMemRangeAttribute", {"hipMemRangeAttribute", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUmem_range_attribute)
  {"cudaMemcpyKind", {"hipMemcpyKind", CONV_TYPE, API_RUNTIME}},
  {"cudaMemoryType", {"hipMemoryType", CONV_TYPE, API_RUNTIME}}, // API_Driver ANALOGUE (no -  CUmemorytype is not an analogue)

  {"cudaPitchedPtr", {"hipPitchedPtr", CONV_TYPE, API_RUNTIME}},
  {"cudaPos",        {"hipPos",        CONV_TYPE, API_RUNTIME}},

  {"cudaEvent_t",           {"hipEvent_t",            CONV_TYPE, API_RUNTIME}},
  {"cudaStream_t",          {"hipStream_t",           CONV_TYPE, API_RUNTIME}},
  {"cudaPointerAttributes", {"hipPointerAttribute_t", CONV_TYPE, API_RUNTIME}},

  {"cudaDeviceAttr",      {"hipDeviceAttribute_t",  CONV_TYPE, API_RUNTIME}},                  // API_DRIVER ANALOGUE (CUdevice_attribute)
  {"cudaDeviceP2PAttr",   {"hipDeviceP2PAttribute", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_DRIVER ANALOGUE (CUdevice_P2PAttribute)
  {"cudaComputeMode",     {"hipComputeMode",        CONV_TYPE, API_RUNTIME}}, // API_DRIVER ANALOGUE (CUcomputemode)
  {"cudaFuncCache",       {"hipFuncCache_t",        CONV_TYPE, API_RUNTIME}}, // API_Driver ANALOGUE (CUfunc_cache)
  {"cudaSharedMemConfig", {"hipSharedMemConfig",    CONV_TYPE, API_RUNTIME}},
  {"cudaLimit",           {"hipLimit_t",            CONV_TYPE, API_RUNTIME}},                  // API_Driver ANALOGUE (CUlimit)

  {"cudaOutputMode", {"hipOutputMode", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // Texture reference management
  {"cudaTextureReadMode",   {"hipTextureReadMode",   CONV_TYPE, API_RUNTIME}},
  {"cudaTextureFilterMode", {"hipTextureFilterMode", CONV_TYPE, API_RUNTIME}}, // API_DRIVER ANALOGUE (CUfilter_mode)

  {"cudaChannelFormatKind", {"hipChannelFormatKind", CONV_TYPE, API_RUNTIME}},

  // Texture Object Management
  {"cudaResourceDesc",     {"hipResourceDesc",     CONV_TYPE, API_RUNTIME}},
  {"cudaResourceViewDesc", {"hipResourceViewDesc", CONV_TYPE, API_RUNTIME}},
  {"cudaTextureDesc",      {"hipTextureDesc",      CONV_TYPE, API_RUNTIME}},
  {"surfaceReference",     {"hipSurfaceReference", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  // Left unchanged
  //     {"textureReference", {"textureReference", CONV_TYPE, API_RUNTIME}},

  // typedefs
  {"cudaTextureObject_t",  {"hipTextureObject_t",  CONV_TYPE,     API_RUNTIME}},
  {"cudaSurfaceObject_t",  {"hipSurfaceObject_t",  CONV_TYPE, API_RUNTIME}},

  // enums
  {"cudaResourceType",        {"hipResourceType",        CONV_TYPE, API_RUNTIME}}, // API_Driver ANALOGUE (CUresourcetype)
  {"cudaResourceViewFormat",  {"hipResourceViewFormat",  CONV_TYPE, API_RUNTIME}}, // API_Driver ANALOGUE (CUresourceViewFormat)
  {"cudaTextureAddressMode",  {"hipTextureAddressMode",  CONV_TYPE, API_RUNTIME}},
  {"cudaSurfaceBoundaryMode", {"hipSurfaceBoundaryMode", CONV_TYPE, API_RUNTIME}},

  {"cudaSurfaceFormatMode", {"hipSurfaceFormatMode", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // defines
  {"cudaTextureType1D",             {"hipTextureType1D",             CONV_TYPE, API_RUNTIME}},
  {"cudaTextureType2D",             {"hipTextureType2D",             CONV_TYPE, API_RUNTIME}},
  {"cudaTextureType3D",             {"hipTextureType3D",             CONV_TYPE, API_RUNTIME}},
  {"cudaTextureTypeCubemap",        {"hipTextureTypeCubemap",        CONV_TYPE, API_RUNTIME}},
  {"cudaTextureType1DLayered",      {"hipTextureType1DLayered",      CONV_TYPE, API_RUNTIME}},
  {"cudaTextureType2DLayered",      {"hipTextureType2DLayered",      CONV_TYPE, API_RUNTIME}},
  {"cudaTextureTypeCubemapLayered", {"hipTextureTypeCubemapLayered", CONV_TYPE, API_RUNTIME}},

  // Graphics Interoperability
  {"cudaGraphicsCubeFace",      {"hipGraphicsCubeFace",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGraphicsMapFlags",      {"hipGraphicsMapFlags",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUgraphicsMapResourceFlags)
  {"cudaGraphicsRegisterFlags", {"hipGraphicsRegisterFlags", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUgraphicsRegisterFlags)

  // OpenGL Interoperability
  {"cudaGLDeviceList", {"hipGLDeviceList", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUGLDeviceList)
  {"cudaGLMapFlags",   {"hipGLMapFlags",   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUGLmap_flags)

  // Direct3D 9 Interoperability
  {"cudaD3D9DeviceList",    {"hipD3D9DeviceList",    CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUd3d9DeviceList)
  {"cudaD3D9MapFlags",      {"hipD3D9MapFlags",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUd3d9map_flags)
  {"cudaD3D9RegisterFlags", {"hipD3D9RegisterFlags", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUd3d9Register_flags)

  // Direct3D 10 Interoperability
  {"cudaD3D10DeviceList",    {"hipd3d10DeviceList",    CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUd3d10DeviceList)
  {"cudaD3D10MapFlags",      {"hipD3D10MapFlags",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUd3d10map_flags)
  {"cudaD3D10RegisterFlags", {"hipD3D10RegisterFlags", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUd3d10Register_flags)

  // Direct3D 11 Interoperability
  {"cudaD3D11DeviceList", {"hipd3d11DeviceList", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUd3d11DeviceList)

  // EGL Interoperability
  {"cudaEglStreamConnection", {"hipEglStreamConnection", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}}, // API_Driver ANALOGUE (CUeglStreamConnection)


  // Library property types
  // IMPORTANT: no cuda prefix
  {"MAJOR_VERSION",         {"hipLibraryMajorVersion",   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"MINOR_VERSION",         {"hipLibraryMinorVersion",   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"PATCH_LEVEL",           {"hipLibraryPatchVersion",   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // defines
  {"cudaMemAttachGlobal",                 {"hipMemAttachGlobal",                 CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_MEM_ATTACH_GLOBAL = 0x1)
  {"cudaMemAttachHost",                   {"hipMemAttachHost",                   CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x02 // API_Driver ANALOGUE (CU_MEM_ATTACH_HOST = 0x2)
  {"cudaMemAttachSingle",                 {"hipMemAttachSingle",                 CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x04 // API_Driver ANALOGUE (CU_MEM_ATTACH_SINGLE = 0x4)

  {"cudaOccupancyDefault",                {"hipOccupancyDefault",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x00 // API_Driver ANALOGUE (CU_OCCUPANCY_DEFAULT = 0x0)
  {"cudaOccupancyDisableCachingOverride", {"hipOccupancyDisableCachingOverride", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 0x1)


  // enum cudaDeviceAttr
  {"cudaDevAttrMaxThreadsPerBlock",               {"hipDeviceAttributeMaxThreadsPerBlock",               CONV_TYPE,   API_RUNTIME}},                     //  1 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1)
  {"cudaDevAttrMaxBlockDimX",                     {"hipDeviceAttributeMaxBlockDimX",                     CONV_TYPE,   API_RUNTIME}},                     //  2 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2)
  {"cudaDevAttrMaxBlockDimY",                     {"hipDeviceAttributeMaxBlockDimY",                     CONV_TYPE,   API_RUNTIME}},                     //  3 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3)
  {"cudaDevAttrMaxBlockDimZ",                     {"hipDeviceAttributeMaxBlockDimZ",                     CONV_TYPE,   API_RUNTIME}},                     //  4 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4)
  {"cudaDevAttrMaxGridDimX",                      {"hipDeviceAttributeMaxGridDimX",                      CONV_TYPE,   API_RUNTIME}},                     //  5 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5)
  {"cudaDevAttrMaxGridDimY",                      {"hipDeviceAttributeMaxGridDimY",                      CONV_TYPE,   API_RUNTIME}},                     //  6 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 6)
  {"cudaDevAttrMaxGridDimZ",                      {"hipDeviceAttributeMaxGridDimZ",                      CONV_TYPE,   API_RUNTIME}},                     //  7 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 7)
  {"cudaDevAttrMaxSharedMemoryPerBlock",          {"hipDeviceAttributeMaxSharedMemoryPerBlock",          CONV_TYPE,   API_RUNTIME}},                     //  8 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8)
  {"cudaDevAttrTotalConstantMemory",              {"hipDeviceAttributeTotalConstantMemory",              CONV_TYPE,   API_RUNTIME}},                     //  9 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY =9)
  {"cudaDevAttrWarpSize",                         {"hipDeviceAttributeWarpSize",                         CONV_TYPE,   API_RUNTIME}},                     // 10 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10)
  {"cudaDevAttrMaxPitch",                         {"hipDeviceAttributeMaxPitch",                         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 11 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11)
  {"cudaDevAttrMaxRegistersPerBlock",             {"hipDeviceAttributeMaxRegistersPerBlock",             CONV_TYPE,   API_RUNTIME}},                     // 12 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12)
  {"cudaDevAttrClockRate",                        {"hipDeviceAttributeClockRate",                        CONV_TYPE,   API_RUNTIME}},                     // 13 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13)
  {"cudaDevAttrTextureAlignment",                 {"hipDeviceAttributeTextureAlignment",                 CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 14 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14)
  // Is not deprecated as CUDA Driver's API analogue CU_DEVICE_ATTRIBUTE_GPU_OVERLAP
  {"cudaDevAttrGpuOverlap",                       {"hipDeviceAttributeGpuOverlap",                       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 15 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15)
  {"cudaDevAttrMultiProcessorCount",              {"hipDeviceAttributeMultiprocessorCount",              CONV_TYPE,   API_RUNTIME}},                     // 16 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16)
  {"cudaDevAttrKernelExecTimeout",                {"hipDeviceAttributeKernelExecTimeout",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 17 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17)
  {"cudaDevAttrIntegrated",                       {"hipDeviceAttributeIntegrated",                       CONV_TYPE,   API_RUNTIME}},    // 18 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_INTEGRATED = 18)
  {"cudaDevAttrCanMapHostMemory",                 {"hipDeviceAttributeCanMapHostMemory",                 CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 19 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19)
  {"cudaDevAttrComputeMode",                      {"hipDeviceAttributeComputeMode",                      CONV_TYPE,   API_RUNTIME}},                     // 20 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20)
  {"cudaDevAttrMaxTexture1DWidth",                {"hipDeviceAttributeMaxTexture1DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 21 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21)
  {"cudaDevAttrMaxTexture2DWidth",                {"hipDeviceAttributeMaxTexture2DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 22 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22)
  {"cudaDevAttrMaxTexture2DHeight",               {"hipDeviceAttributeMaxTexture2DHeight",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 23 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23)
  {"cudaDevAttrMaxTexture3DWidth",                {"hipDeviceAttributeMaxTexture3DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 24 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24)
  {"cudaDevAttrMaxTexture3DHeight",               {"hipDeviceAttributeMaxTexture3DHeight",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 25 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25)
  {"cudaDevAttrMaxTexture3DDepth",                {"hipDeviceAttributeMaxTexture3DDepth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 26 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26)
  {"cudaDevAttrMaxTexture2DLayeredWidth",         {"hipDeviceAttributeMaxTexture2DLayeredWidth",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 27 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27)
  {"cudaDevAttrMaxTexture2DLayeredHeight",        {"hipDeviceAttributeMaxTexture2DLayeredHeight",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 28 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28)
  {"cudaDevAttrMaxTexture2DLayeredLayers",        {"hipDeviceAttributeMaxTexture2DLayeredLayers",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 29 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29)
  {"cudaDevAttrSurfaceAlignment",                 {"hipDeviceAttributeSurfaceAlignment",                 CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 30 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30)
  {"cudaDevAttrConcurrentKernels",                {"hipDeviceAttributeConcurrentKernels",                CONV_TYPE,   API_RUNTIME}},                     // 31 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31)
  {"cudaDevAttrEccEnabled",                       {"hipDeviceAttributeEccEnabled",                       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 32 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32)
  {"cudaDevAttrPciBusId",                         {"hipDeviceAttributePciBusId",                         CONV_TYPE,   API_RUNTIME}},                     // 33 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33)
  {"cudaDevAttrPciDeviceId",                      {"hipDeviceAttributePciDeviceId",                      CONV_TYPE,   API_RUNTIME}},                     // 34 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34)
  {"cudaDevAttrTccDriver",                        {"hipDeviceAttributeTccDriver",                        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 35 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35)
  {"cudaDevAttrMemoryClockRate",                  {"hipDeviceAttributeMemoryClockRate",                  CONV_TYPE,   API_RUNTIME}},                     // 36 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36)
  {"cudaDevAttrGlobalMemoryBusWidth",             {"hipDeviceAttributeMemoryBusWidth",                   CONV_TYPE,   API_RUNTIME}},                     // 37 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37)
  {"cudaDevAttrL2CacheSize",                      {"hipDeviceAttributeL2CacheSize",                      CONV_TYPE,   API_RUNTIME}},                     // 38 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38)
  {"cudaDevAttrMaxThreadsPerMultiProcessor",      {"hipDeviceAttributeMaxThreadsPerMultiProcessor",      CONV_TYPE,   API_RUNTIME}},                     // 39 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39)
  {"cudaDevAttrAsyncEngineCount",                 {"hipDeviceAttributeAsyncEngineCount",                 CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 40 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40)
  {"cudaDevAttrUnifiedAddressing",                {"hipDeviceAttributeUnifiedAddressing",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 41 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41)
  {"cudaDevAttrMaxTexture1DLayeredWidth",         {"hipDeviceAttributeMaxTexture1DLayeredWidth",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 42 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42)
  {"cudaDevAttrMaxTexture1DLayeredLayers",        {"hipDeviceAttributeMaxTexture1DLayeredLayers",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 43 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43)
  // 44 - no
  {"cudaDevAttrMaxTexture2DGatherWidth",          {"hipDeviceAttributeMaxTexture2DGatherWidth",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 45 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45)
  {"cudaDevAttrMaxTexture2DGatherHeight",         {"hipDeviceAttributeMaxTexture2DGatherHeight",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 46 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46)
  {"cudaDevAttrMaxTexture3DWidthAlt",             {"hipDeviceAttributeMaxTexture3DWidthAlternate",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 47 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47)
  {"cudaDevAttrMaxTexture3DHeightAlt",            {"hipDeviceAttributeMaxTexture3DHeightAlternate",      CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 48 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48)
  {"cudaDevAttrMaxTexture3DDepthAlt",             {"hipDeviceAttributeMaxTexture3DDepthAlternate",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 49 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49)
  {"cudaDevAttrPciDomainId",                      {"hipDeviceAttributePciDomainId",                      CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 50 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50)
  {"cudaDevAttrTexturePitchAlignment",            {"hipDeviceAttributeTexturePitchAlignment",            CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 51 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51)
  {"cudaDevAttrMaxTextureCubemapWidth",           {"hipDeviceAttributeMaxTextureCubemapWidth",           CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 52 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52)
  {"cudaDevAttrMaxTextureCubemapLayeredWidth",    {"hipDeviceAttributeMaxTextureCubemapLayeredWidth",    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 53 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53)
  {"cudaDevAttrMaxTextureCubemapLayeredLayers",   {"hipDeviceAttributeMaxTextureCubemapLayeredLayers",   CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 54 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54)
  {"cudaDevAttrMaxSurface1DWidth",                {"hipDeviceAttributeMaxSurface1DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 55 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55)
  {"cudaDevAttrMaxSurface2DWidth",                {"hipDeviceAttributeMaxSurface2DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 56 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56)
  {"cudaDevAttrMaxSurface2DHeight",               {"hipDeviceAttributeMaxSurface2DHeight",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 57 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57)
  {"cudaDevAttrMaxSurface3DWidth",                {"hipDeviceAttributeMaxSurface3DWidth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 58 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58)
  {"cudaDevAttrMaxSurface3DHeight",               {"hipDeviceAttributeMaxSurface3DHeight",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 59 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59)
  {"cudaDevAttrMaxSurface3DDepth",                {"hipDeviceAttributeMaxSurface3DDepth",                CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 60 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60)
  {"cudaDevAttrMaxSurface1DLayeredWidth",         {"hipDeviceAttributeMaxSurface1DLayeredWidth",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 61 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61)
  {"cudaDevAttrMaxSurface1DLayeredLayers",        {"hipDeviceAttributeMaxSurface1DLayeredLayers",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 62 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62)
  {"cudaDevAttrMaxSurface2DLayeredWidth",         {"hipDeviceAttributeMaxSurface2DLayeredWidth",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 63 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63)
  {"cudaDevAttrMaxSurface2DLayeredHeight",        {"hipDeviceAttributeMaxSurface2DLayeredHeight",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 64 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64)
  {"cudaDevAttrMaxSurface2DLayeredLayers",        {"hipDeviceAttributeMaxSurface2DLayeredLayers",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 65 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65)
  {"cudaDevAttrMaxSurfaceCubemapWidth",           {"hipDeviceAttributeMaxSurfaceCubemapWidth",           CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 66 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66)
  {"cudaDevAttrMaxSurfaceCubemapLayeredWidth",    {"hipDeviceAttributeMaxSurfaceCubemapLayeredWidth",    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 67 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67)
  {"cudaDevAttrMaxSurfaceCubemapLayeredLayers",   {"hipDeviceAttributeMaxSurfaceCubemapLayeredLayers",   CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 68 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68)
  {"cudaDevAttrMaxTexture1DLinearWidth",          {"hipDeviceAttributeMaxTexture1DLinearWidth",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 69 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69)
  {"cudaDevAttrMaxTexture2DLinearWidth",          {"hipDeviceAttributeMaxTexture2DLinearWidth",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 70 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70)
  {"cudaDevAttrMaxTexture2DLinearHeight",         {"hipDeviceAttributeMaxTexture2DLinearHeight",         CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 71 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71)
  {"cudaDevAttrMaxTexture2DLinearPitch",          {"hipDeviceAttributeMaxTexture2DLinearPitch",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 72 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72)
  {"cudaDevAttrMaxTexture2DMipmappedWidth",       {"hipDeviceAttributeMaxTexture2DMipmappedWidth",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 73 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73)
  {"cudaDevAttrMaxTexture2DMipmappedHeight",      {"hipDeviceAttributeMaxTexture2DMipmappedHeight",      CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 74 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74)
  {"cudaDevAttrComputeCapabilityMajor",           {"hipDeviceAttributeComputeCapabilityMajor",           CONV_TYPE,   API_RUNTIME}},                     // 75 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75)
  {"cudaDevAttrComputeCapabilityMinor",           {"hipDeviceAttributeComputeCapabilityMinor",           CONV_TYPE,   API_RUNTIME}},                     // 76 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76)
  {"cudaDevAttrMaxTexture1DMipmappedWidth",       {"hipDeviceAttributeMaxTexture1DMipmappedWidth",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 77 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77)
  {"cudaDevAttrStreamPrioritiesSupported",        {"hipDeviceAttributeStreamPrioritiesSupported",        CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 78 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78)
  {"cudaDevAttrGlobalL1CacheSupported",           {"hipDeviceAttributeGlobalL1CacheSupported",           CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 79 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79)
  {"cudaDevAttrLocalL1CacheSupported",            {"hipDeviceAttributeLocalL1CacheSupported",            CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 80 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80)
  {"cudaDevAttrMaxSharedMemoryPerMultiprocessor", {"hipDeviceAttributeMaxSharedMemoryPerMultiprocessor", CONV_TYPE,   API_RUNTIME}},                     // 81 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81)
  {"cudaDevAttrMaxRegistersPerMultiprocessor",    {"hipDeviceAttributeMaxRegistersPerMultiprocessor",    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 82 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82)
  {"cudaDevAttrManagedMemory",                    {"hipDeviceAttributeManagedMemory",                    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 83 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83)
  {"cudaDevAttrIsMultiGpuBoard",                  {"hipDeviceAttributeIsMultiGpuBoard",                  CONV_TYPE,   API_RUNTIME}},                     // 84 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84)
  {"cudaDevAttrMultiGpuBoardGroupID",             {"hipDeviceAttributeMultiGpuBoardGroupID",             CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 85 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85)
  {"cudaDevAttrHostNativeAtomicSupported",         {"hipDeviceAttributeHostNativeAtomicSupported",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 86 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86)
  {"cudaDevAttrSingleToDoublePrecisionPerfRatio",  {"hipDeviceAttributeSingleToDoublePrecisionPerfRatio",  CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 87 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87)
  {"cudaDevAttrPageableMemoryAccess",              {"hipDeviceAttributePageableMemoryAccess",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 88 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88)
  {"cudaDevAttrConcurrentManagedAccess",           {"hipDeviceAttributeConcurrentManagedAccess",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 89 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89)
  {"cudaDevAttrComputePreemptionSupported",        {"hipDeviceAttributeComputePreemptionSupported",        CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 90 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90)
  {"cudaDevAttrCanUseHostPointerForRegisteredMem", {"hipDeviceAttributeCanUseHostPointerForRegisteredMem", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 91 // API_DRIVER ANALOGUE (CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91)


  // Memory advise values
  {"cudaMemAdviseSetReadMostly",                {"hipMemAdviseSetReadMostly",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_MEM_ADVISE_SET_READ_MOSTLY = 1)
  {"cudaMemAdviseUnsetReadMostly",              {"hipMemAdviseUnsetReadMostly",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2)
  {"cudaMemAdviseSetPreferredLocation",         {"hipMemAdviseSetPreferredLocation",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3)
  {"cudaMemAdviseUnsetPreferredLocation",       {"hipMemAdviseUnsetPreferredLocation",       CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 4 // API_Driver ANALOGUE (CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4)
  {"cudaMemAdviseSetAccessedBy",                {"hipMemAdviseSetAccessedBy",                CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 5 // API_Driver ANALOGUE (CU_MEM_ADVISE_SET_ACCESSED_BY = 5)
  {"cudaMemAdviseUnsetAccessedBy",              {"hipMemAdviseUnsetAccessedBy",              CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 6 // API_Driver ANALOGUE (CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6)

  // CUmem_range_attribute
  {"cudaMemRangeAttributeReadMostly",           {"hipMemRangeAttributeReadMostly",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1)
  {"cudaMemRangeAttributePreferredLocation",    {"hipMemRangeAttributePreferredLocation",    CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2)
  {"cudaMemRangeAttributeAccessedBy",           {"hipMemRangeAttributeAccessedBy",           CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3)
  {"cudaMemRangeAttributeLastPrefetchLocation", {"hipMemRangeAttributeLastPrefetchLocation", CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},    // 4 // API_Driver ANALOGUE (CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4)

  // memcpy kind
  {"cudaMemcpyHostToHost",     {"hipMemcpyHostToHost",     CONV_MEM, API_RUNTIME}},
  {"cudaMemcpyHostToDevice",   {"hipMemcpyHostToDevice",   CONV_MEM, API_RUNTIME}},
  {"cudaMemcpyDeviceToHost",   {"hipMemcpyDeviceToHost",   CONV_MEM, API_RUNTIME}},
  {"cudaMemcpyDeviceToDevice", {"hipMemcpyDeviceToDevice", CONV_MEM, API_RUNTIME}},
  {"cudaMemcpyDefault",        {"hipMemcpyDefault",        CONV_MEM, API_RUNTIME}},


  // Stream Flags (defines)
  {"cudaStreamDefault",     {"hipStreamDefault",     CONV_TYPE, API_RUNTIME}},
  {"cudaStreamNonBlocking", {"hipStreamNonBlocking", CONV_TYPE, API_RUNTIME}},

  // P2P Attributes (enum cudaDeviceP2PAttr)
  {"cudaDevP2PAttrPerformanceRank",       {"hipDeviceP2PAttributePerformanceRank",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_DRIVER ANALOGUE (CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 0x01)
  {"cudaDevP2PAttrAccessSupported",       {"hipDeviceP2PAttributeAccessSupported",       CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x02 // API_DRIVER ANALOGUE (CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 0x02)
  {"cudaDevP2PAttrNativeAtomicSupported", {"hipDeviceP2PAttributeNativeAtomicSupported", CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x03 // API_DRIVER ANALOGUE (CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 0x03)
  //
  {"cudaDeviceGetP2PAttribute",           {"hipDeviceGetP2PAttribute",                   CONV_DEVICE, API_RUNTIME, HIP_UNSUPPORTED}},    // API_DRIVER ANALOGUE (cuDeviceGetP2PAttribute)

  // enum cudaComputeMode
  {"cudaComputeModeDefault",          {"hipComputeModeDefault",          CONV_TYPE, API_RUNTIME}},    // 0 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_DEFAULT = 0)
  {"cudaComputeModeExclusive",        {"hipComputeModeExclusive",        CONV_TYPE, API_RUNTIME}},    // 1 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_EXCLUSIVE = 1)
  {"cudaComputeModeProhibited",       {"hipComputeModeProhibited",       CONV_TYPE, API_RUNTIME}},    // 2 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_PROHIBITED = 2)
  {"cudaComputeModeExclusiveProcess", {"hipComputeModeExclusiveProcess", CONV_TYPE, API_RUNTIME}},    // 3 // API_DRIVER ANALOGUE (CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3)

  // Device stuff (#defines)
  {"cudaDeviceScheduleAuto",         {"hipDeviceScheduleAuto",         CONV_TYPE, API_RUNTIME}},
  {"cudaDeviceScheduleSpin",         {"hipDeviceScheduleSpin",         CONV_TYPE, API_RUNTIME}},
  {"cudaDeviceScheduleYield",        {"hipDeviceScheduleYield",        CONV_TYPE, API_RUNTIME}},
  // Deprecated since CUDA 4.0 and replaced with cudaDeviceScheduleBlockingSync
  {"cudaDeviceBlockingSync",         {"hipDeviceScheduleBlockingSync", CONV_TYPE, API_RUNTIME}},
  {"cudaDeviceScheduleBlockingSync", {"hipDeviceScheduleBlockingSync", CONV_TYPE, API_RUNTIME}},
  {"cudaDeviceScheduleMask",         {"hipDeviceScheduleMask",         CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaDeviceMapHost",              {"hipDeviceMapHost",              CONV_TYPE, API_RUNTIME}},
  {"cudaDeviceLmemResizeToMax",      {"hipDeviceLmemResizeToMax",      CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaDeviceMask",                 {"hipDeviceMask",                 CONV_TYPE, API_RUNTIME, HIP_UNSUPPORTED}},

  // #define cudaIpcMemLazyEnablePeerAccess 0x01
  {"cudaIpcMemLazyEnablePeerAccess", {"hipIpcMemLazyEnablePeerAccess", CONV_TYPE,   API_RUNTIME}},    // 0x01 // API_Driver ANALOGUE (CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1)

  // enum cudaSharedMemConfig
  {"cudaSharedMemBankSizeDefault",   {"hipSharedMemBankSizeDefault",   CONV_TYPE, API_RUNTIME}},
  {"cudaSharedMemBankSizeFourByte",  {"hipSharedMemBankSizeFourByte",  CONV_TYPE, API_RUNTIME}},
  {"cudaSharedMemBankSizeEightByte", {"hipSharedMemBankSizeEightByte", CONV_TYPE, API_RUNTIME}},

  // enum cudaLimit
  {"cudaLimitStackSize",                    {"hipLimitStackSize",                    CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x00 // API_Driver ANALOGUE (CU_LIMIT_STACK_SIZE = 0x00)
  {"cudaLimitPrintfFifoSize",               {"hipLimitPrintfFifoSize",               CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_LIMIT_PRINTF_FIFO_SIZE = 0x01)
  {"cudaLimitMallocHeapSize",               {"hipLimitMallocHeapSize",               CONV_TYPE,   API_RUNTIME}},                     // 0x02 // API_Driver ANALOGUE (CU_LIMIT_MALLOC_HEAP_SIZE = 0x02)
  {"cudaLimitDevRuntimeSyncDepth",          {"hipLimitDevRuntimeSyncDepth",          CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x03 // API_Driver ANALOGUE (CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 0x03)
  {"cudaLimitDevRuntimePendingLaunchCount", {"hipLimitDevRuntimePendingLaunchCount", CONV_TYPE,   API_RUNTIME, HIP_UNSUPPORTED}},    // 0x04 // API_Driver ANALOGUE (CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04)

  // enum cudaGraphicsMapFlags
  {"cudaGraphicsMapFlagsNone",         {"hipGraphicsMapFlagsNone",         CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00)
  {"cudaGraphicsMapFlagsReadOnly",     {"hipGraphicsMapFlagsReadOnly",     CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01)
  {"cudaGraphicsMapFlagsWriteDiscard", {"hipGraphicsMapFlagsWriteDiscard", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 0x02)

  // enum cudaGraphicsRegisterFlags
  {"cudaGraphicsRegisterFlagsNone",             {"hipGraphicsRegisterFlagsNone",             CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0x00)
  {"cudaGraphicsRegisterFlagsReadOnly",         {"hipGraphicsRegisterFlagsReadOnly",         CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 0x01)
  {"cudaGraphicsRegisterFlagsWriteDiscard",     {"hipGraphicsRegisterFlagsWriteDiscard",     CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 0x02)
  {"cudaGraphicsRegisterFlagsSurfaceLoadStore", {"hipGraphicsRegisterFlagsSurfaceLoadStore", CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 4 // API_Driver ANALOGUE (CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 0x04)
  {"cudaGraphicsRegisterFlagsTextureGather",    {"hipGraphicsRegisterFlagsTextureGather",    CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},    // 8 // API_Driver ANALOGUE (CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 0x08)

  // enum cudaMemoryType
  {"cudaMemoryTypeHost",   {"hipMemoryTypeHost",   CONV_MEM, API_RUNTIME}},
  {"cudaMemoryTypeDevice", {"hipMemoryTypeDevice", CONV_MEM, API_RUNTIME}},


  // Execution control
  // CUDA function cache configurations (enum cudaFuncCache)
  {"cudaFuncCachePreferNone",   {"hipFuncCachePreferNone",   CONV_CACHE, API_RUNTIME}},    // 0 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_NONE = 0x00)
  {"cudaFuncCachePreferShared", {"hipFuncCachePreferShared", CONV_CACHE, API_RUNTIME}},    // 1 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_SHARED = 0x01)
  {"cudaFuncCachePreferL1",     {"hipFuncCachePreferL1",     CONV_CACHE, API_RUNTIME}},    // 2 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_L1 = 0x02)
  {"cudaFuncCachePreferEqual",  {"hipFuncCachePreferEqual",  CONV_CACHE, API_RUNTIME}},    // 3 // API_Driver ANALOGUE (CU_FUNC_CACHE_PREFER_EQUAL = 0x03)


  // enum cudaOutputMode
  {"cudaKeyValuePair", {"hipKeyValuePair", CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaCSV",          {"hipCSV",          CONV_OTHER, API_RUNTIME, HIP_UNSUPPORTED}},

  // Texture Reference Management
  // enum cudaTextureReadMode
  {"cudaReadModeElementType",         {"hipReadModeElementType",         CONV_TEX, API_RUNTIME}},
  {"cudaReadModeNormalizedFloat",     {"hipReadModeNormalizedFloat",     CONV_TEX, API_RUNTIME}},

  // enum cudaTextureFilterMode
  {"cudaFilterModePoint",             {"hipFilterModePoint",             CONV_TEX, API_RUNTIME}},    // 0 // API_DRIVER ANALOGUE (CU_TR_FILTER_MODE_POINT = 0)
  {"cudaFilterModeLinear",            {"hipFilterModeLinear",            CONV_TEX, API_RUNTIME}},    // 1 // API_DRIVER ANALOGUE (CU_TR_FILTER_MODE_POINT = 1)


  // Channel (enum cudaChannelFormatKind)
  {"cudaChannelFormatKindSigned",   {"hipChannelFormatKindSigned",   CONV_TEX, API_RUNTIME}},
  {"cudaChannelFormatKindUnsigned", {"hipChannelFormatKindUnsigned", CONV_TEX, API_RUNTIME}},
  {"cudaChannelFormatKindFloat",    {"hipChannelFormatKindFloat",    CONV_TEX, API_RUNTIME}},
  {"cudaChannelFormatKindNone",     {"hipChannelFormatKindNone",     CONV_TEX, API_RUNTIME}},

  // enum cudaResourceType
  {"cudaResourceTypeArray",          {"hipResourceTypeArray",          CONV_TEX, API_RUNTIME}},    // 0x00 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_ARRAY = 0x00)
  {"cudaResourceTypeMipmappedArray", {"hipResourceTypeMipmappedArray", CONV_TEX, API_RUNTIME}},    // 0x01 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01)
  {"cudaResourceTypeLinear",         {"hipResourceTypeLinear",         CONV_TEX, API_RUNTIME}},    // 0x02 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_LINEAR = 0x02)
  {"cudaResourceTypePitch2D",        {"hipResourceTypePitch2D",        CONV_TEX, API_RUNTIME}},    // 0x03 // API_Driver ANALOGUE (CU_RESOURCE_TYPE_PITCH2D = 0x03)

  // enum cudaResourceViewFormat
  {"cudaResViewFormatNone",                       {"hipResViewFormatNone",                       CONV_TEX,      API_RUNTIME}},    // 0x00 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_NONE = 0x00)
  {"cudaResViewFormatUnsignedChar1",              {"hipResViewFormatUnsignedChar1",              CONV_TEX,      API_RUNTIME}},    // 0x01 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_1X8 = 0x01)
  {"cudaResViewFormatUnsignedChar2",              {"hipResViewFormatUnsignedChar2",              CONV_TEX,      API_RUNTIME}},    // 0x02 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_2X8 = 0x02)
  {"cudaResViewFormatUnsignedChar4",              {"hipResViewFormatUnsignedChar4",              CONV_TEX,      API_RUNTIME}},    // 0x03 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_4X8 = 0x03)
  {"cudaResViewFormatSignedChar1",                {"hipResViewFormatSignedChar1",                CONV_TEX,      API_RUNTIME}},    // 0x04 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_1X8 = 0x04)
  {"cudaResViewFormatSignedChar2",                {"hipResViewFormatSignedChar2",                CONV_TEX,      API_RUNTIME}},    // 0x05 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_2X8 = 0x05)
  {"cudaResViewFormatSignedChar4",                {"hipResViewFormatSignedChar4",                CONV_TEX,      API_RUNTIME}},    // 0x06 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_4X8 = 0x06)
  {"cudaResViewFormatUnsignedShort1",             {"hipResViewFormatUnsignedShort1",             CONV_TEX,      API_RUNTIME}},    // 0x07 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_1X16 = 0x07)
  {"cudaResViewFormatUnsignedShort2",             {"hipResViewFormatUnsignedShort2",             CONV_TEX,      API_RUNTIME}},    // 0x08 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_2X16 = 0x08)
  {"cudaResViewFormatUnsignedShort4",             {"hipResViewFormatUnsignedShort4",             CONV_TEX,      API_RUNTIME}},    // 0x09 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_4X16 = 0x09)
  {"cudaResViewFormatSignedShort1",               {"hipResViewFormatSignedShort1",               CONV_TEX,      API_RUNTIME}},    // 0x0a // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_1X16 = 0x0a)
  {"cudaResViewFormatSignedShort2",               {"hipResViewFormatSignedShort2",               CONV_TEX,      API_RUNTIME}},    // 0x0b // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_2X16 = 0x0b)
  {"cudaResViewFormatSignedShort4",               {"hipResViewFormatSignedShort4",               CONV_TEX,      API_RUNTIME}},    // 0x0c // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_4X16 = 0x0c)
  {"cudaResViewFormatUnsignedInt1",               {"hipResViewFormatUnsignedInt1",               CONV_TEX,      API_RUNTIME}},    // 0x0d // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_1X32 = 0x0d)
  {"cudaResViewFormatUnsignedInt2",               {"hipResViewFormatUnsignedInt2",               CONV_TEX,      API_RUNTIME}},    // 0x0e // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_2X32 = 0x0e)
  {"cudaResViewFormatUnsignedInt4",               {"hipResViewFormatUnsignedInt4",               CONV_TEX,      API_RUNTIME}},    // 0x0f // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UINT_4X32 = 0x0f)
  {"cudaResViewFormatSignedInt1",                 {"hipResViewFormatSignedInt1",                 CONV_TEX,      API_RUNTIME}},    // 0x10 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_1X32 = 0x10)
  {"cudaResViewFormatSignedInt2",                 {"hipResViewFormatSignedInt2",                 CONV_TEX,      API_RUNTIME}},    // 0x11 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_2X32 = 0x11)
  {"cudaResViewFormatSignedInt4",                 {"hipResViewFormatSignedInt4",                 CONV_TEX,      API_RUNTIME}},    // 0x12 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SINT_4X32 = 0x12)
  {"cudaResViewFormatHalf1",                      {"hipResViewFormatHalf1",                      CONV_TEX,      API_RUNTIME}},    // 0x13 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_1X16 = 0x13)
  {"cudaResViewFormatHalf2",                      {"hipResViewFormatHalf2",                      CONV_TEX,      API_RUNTIME}},    // 0x14 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_2X16 = 0x14)
  {"cudaResViewFormatHalf4",                      {"hipResViewFormatHalf4",                      CONV_TEX,      API_RUNTIME}},    // 0x15 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_4X16 = 0x15)
  {"cudaResViewFormatFloat1",                     {"hipResViewFormatFloat1",                     CONV_TEX,      API_RUNTIME}},    // 0x16 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_1X32 = 0x16)
  {"cudaResViewFormatFloat2",                     {"hipResViewFormatFloat2",                     CONV_TEX,      API_RUNTIME}},    // 0x17 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_2X32 = 0x17)
  {"cudaResViewFormatFloat4",                     {"hipResViewFormatFloat4",                     CONV_TEX,      API_RUNTIME}},    // 0x18 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_FLOAT_4X32 = 0x18)
  {"cudaResViewFormatUnsignedBlockCompressed1",   {"hipResViewFormatUnsignedBlockCompressed1",   CONV_TEX,      API_RUNTIME}},    // 0x19 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19)
  {"cudaResViewFormatUnsignedBlockCompressed2",   {"hipResViewFormatUnsignedBlockCompressed2",   CONV_TEX,      API_RUNTIME}},    // 0x1a // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a)
  {"cudaResViewFormatUnsignedBlockCompressed3",   {"hipResViewFormatUnsignedBlockCompressed3",   CONV_TEX,      API_RUNTIME}},    // 0x1b // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b)
  {"cudaResViewFormatUnsignedBlockCompressed4",   {"hipResViewFormatUnsignedBlockCompressed4",   CONV_TEX,      API_RUNTIME}},    // 0x1c // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c)
  {"cudaResViewFormatSignedBlockCompressed4",     {"hipResViewFormatSignedBlockCompressed4",     CONV_TEX,      API_RUNTIME}},    // 0x1d // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d)
  {"cudaResViewFormatUnsignedBlockCompressed5",   {"hipResViewFormatUnsignedBlockCompressed5",   CONV_TEX,      API_RUNTIME}},    // 0x1e // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e)
  {"cudaResViewFormatSignedBlockCompressed5",     {"hipResViewFormatSignedBlockCompressed5",     CONV_TEX,      API_RUNTIME}},    // 0x1f // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f)
  {"cudaResViewFormatUnsignedBlockCompressed6H",  {"hipResViewFormatUnsignedBlockCompressed6H",  CONV_TEX,      API_RUNTIME}},    // 0x20 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20)
  {"cudaResViewFormatSignedBlockCompressed6H",    {"hipResViewFormatSignedBlockCompressed6H",    CONV_TEX,      API_RUNTIME}},    // 0x21 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21)
  {"cudaResViewFormatUnsignedBlockCompressed7",   {"hipResViewFormatUnsignedBlockCompressed7",   CONV_TEX,      API_RUNTIME}},    // 0x22 // API_Driver ANALOGUE (CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22)


  // enum cudaGraphicsCubeFace
  {"cudaGraphicsCubeFacePositiveX",               {"hipGraphicsCubeFacePositiveX",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGraphicsCubeFaceNegativeX",               {"hipGraphicsCubeFaceNegativeX",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGraphicsCubeFacePositiveY",               {"hipGraphicsCubeFacePositiveY",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGraphicsCubeFaceNegativeY",               {"hipGraphicsCubeFaceNegativeY",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGraphicsCubeFacePositiveZ",               {"hipGraphicsCubeFacePositiveZ",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaGraphicsCubeFaceNegativeZ",               {"hipGraphicsCubeFaceNegativeZ",               CONV_GRAPHICS, API_RUNTIME, HIP_UNSUPPORTED}},

  // OpenGL Interoperability
  // enum cudaGLDeviceList
  {"cudaGLDeviceListAll",          {"HIP_GL_DEVICE_LIST_ALL",           CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_GL_DEVICE_LIST_ALL)
  {"cudaGLDeviceListCurrentFrame", {"HIP_GL_DEVICE_LIST_CURRENT_FRAME", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x02 // API_Driver ANALOGUE (CU_GL_DEVICE_LIST_CURRENT_FRAME)
  {"cudaGLDeviceListNextFrame",    {"HIP_GL_DEVICE_LIST_NEXT_FRAME",    CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x03 // API_Driver ANALOGUE (CU_GL_DEVICE_LIST_NEXT_FRAME)

  // enum cudaSurfaceBoundaryMode
  {"cudaBoundaryModeZero",    {"hipBoundaryModeZero",    CONV_SURFACE, API_RUNTIME}},
  {"cudaBoundaryModeClamp",   {"hipBoundaryModeClamp",   CONV_SURFACE, API_RUNTIME}},
  {"cudaBoundaryModeTrap",    {"hipBoundaryModeTrap",    CONV_SURFACE, API_RUNTIME}},

  // enum cudaSurfaceFormatMode
  {"cudaFormatModeForced",    {"hipFormatModeForced",    CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},
  {"cudaFormatModeAuto",      {"hipFormatModeAuto",      CONV_SURFACE, API_RUNTIME, HIP_UNSUPPORTED}},


  // enum cudaGLMapFlags
  {"cudaGLMapFlagsNone",            {"HIP_GL_MAP_RESOURCE_FLAGS_NONE",          CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x00 // API_Driver ANALOGUE (CU_GL_MAP_RESOURCE_FLAGS_NONE)
  {"cudaGLMapFlagsReadOnly",        {"HIP_GL_MAP_RESOURCE_FLAGS_READ_ONLY",     CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x01 // API_Driver ANALOGUE (CU_GL_MAP_RESOURCE_FLAGS_READ_ONLY)
  {"cudaGLMapFlagsWriteDiscard",    {"HIP_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD", CONV_GL, API_RUNTIME, HIP_UNSUPPORTED}},    // 0x02 // API_Driver ANALOGUE (CU_GL_MAP_RESOURCE_FLAGS_WRITE_DISCARD)

  // enum CUd3d9DeviceList
  {"cudaD3D9DeviceListAll",            {"HIP_D3D9_DEVICE_LIST_ALL",           CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D9_DEVICE_LIST_ALL)
  {"cudaD3D9DeviceListCurrentFrame",   {"HIP_D3D9_DEVICE_LIST_CURRENT_FRAME", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D9_DEVICE_LIST_CURRENT_FRAME)
  {"cudaD3D9DeviceListNextFrame",      {"HIP_D3D9_DEVICE_LIST_NEXT_FRAME",    CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_D3D9_DEVICE_LIST_NEXT_FRAME)

  // cudaD3D9MapFlags enum values
  {"cudaD3D9MapFlags",             {"hipD3D9MapFlags",                         CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // API_Driver ANALOGUE (CUd3d9map_flags)
  {"cudaD3D9MapFlagsNone",         {"HIP_D3D9_MAPRESOURCE_FLAGS_NONE",         CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_D3D9_MAPRESOURCE_FLAGS_NONE)
  {"cudaD3D9MapFlagsReadOnly",     {"HIP_D3D9_MAPRESOURCE_FLAGS_READONLY",     CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D9_MAPRESOURCE_FLAGS_READONLY)
  {"cudaD3D9MapFlagsWriteDiscard", {"HIP_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D9_MAPRESOURCE_FLAGS_WRITEDISCARD)
	

  // enum cudaD3D9RegisterFlags
  {"cudaD3D9RegisterFlagsNone",            {"HIP_D3D9_REGISTER_FLAGS_NONE",        CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_D3D9_REGISTER_FLAGS_NONE)
  {"cudaD3D9RegisterFlagsArray",           {"HIP_D3D9_REGISTER_FLAGS_ARRAY",       CONV_D3D9, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D9_REGISTER_FLAGS_ARRAY)

  // enum cudaD3D10DeviceList
  {"cudaD3D10DeviceListAll",            {"HIP_D3D10_DEVICE_LIST_ALL",           CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D10_DEVICE_LIST_ALL)
  {"cudaD3D10DeviceListCurrentFrame",   {"HIP_D3D10_DEVICE_LIST_CURRENT_FRAME", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D10_DEVICE_LIST_CURRENT_FRAME)
  {"cudaD3D10DeviceListNextFrame",      {"HIP_D3D10_DEVICE_LIST_NEXT_FRAME",    CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_D3D10_DEVICE_LIST_NEXT_FRAME)

  // enum cudaD3D10MapFlags
  {"cudaD3D10MapFlagsNone",                 {"HIP_D3D10_MAPRESOURCE_FLAGS_NONE",         CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_D3D10_MAPRESOURCE_FLAGS_NONE)
  {"cudaD3D10MapFlagsReadOnly",             {"HIP_D3D10_MAPRESOURCE_FLAGS_READONLY",     CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D10_MAPRESOURCE_FLAGS_READONLY)
  {"cudaD3D10MapFlagsWriteDiscard",         {"HIP_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD", CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D10_MAPRESOURCE_FLAGS_WRITEDISCARD)

  // enum cudaD3D10RegisterFlags
  {"cudaD3D10RegisterFlagsNone",            {"HIP_D3D10_REGISTER_FLAGS_NONE",            CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 0 // API_Driver ANALOGUE (CU_D3D10_REGISTER_FLAGS_NONE)
  {"cudaD3D10RegisterFlagsArray",           {"HIP_D3D10_REGISTER_FLAGS_ARRAY",           CONV_D3D10, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D10_REGISTER_FLAGS_ARRAY)

  // enum cudaD3D11DeviceList
  {"cudaD3D11DeviceListAll",            {"HIP_D3D11_DEVICE_LIST_ALL",           CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // 1 // API_Driver ANALOGUE (CU_D3D11_DEVICE_LIST_ALL)
  {"cudaD3D11DeviceListCurrentFrame",   {"HIP_D3D11_DEVICE_LIST_CURRENT_FRAME", CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // 2 // API_Driver ANALOGUE (CU_D3D11_DEVICE_LIST_CURRENT_FRAME)
  {"cudaD3D11DeviceListNextFrame",      {"HIP_D3D11_DEVICE_LIST_NEXT_FRAME",    CONV_D3D11, API_RUNTIME, HIP_UNSUPPORTED}},    // 3 // API_Driver ANALOGUE (CU_D3D11_DEVICE_LIST_NEXT_FRAME)

};

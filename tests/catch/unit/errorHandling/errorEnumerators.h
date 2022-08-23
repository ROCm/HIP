/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime_api.h>

constexpr hipError_t kErrorEnumerators[] = {hipSuccess,
                                            hipErrorInvalidValue,
                                            hipErrorOutOfMemory,
                                            hipErrorNotInitialized,
                                            hipErrorDeinitialized,
                                            hipErrorProfilerDisabled,
                                            hipErrorProfilerNotInitialized,
                                            hipErrorProfilerAlreadyStarted,
                                            hipErrorProfilerAlreadyStopped,
                                            hipErrorInvalidConfiguration,
                                            hipErrorInvalidPitchValue,
                                            hipErrorInvalidSymbol,
                                            hipErrorInvalidDevicePointer,
                                            hipErrorInvalidMemcpyDirection,
                                            hipErrorInsufficientDriver,
                                            hipErrorMissingConfiguration,
                                            hipErrorPriorLaunchFailure,
                                            hipErrorInvalidDeviceFunction,
                                            hipErrorNoDevice,
                                            hipErrorInvalidDevice,
                                            hipErrorInvalidImage,
                                            hipErrorInvalidContext,
                                            hipErrorContextAlreadyCurrent,
                                            hipErrorMapFailed,
                                            hipErrorUnmapFailed,
                                            hipErrorArrayIsMapped,
                                            hipErrorAlreadyMapped,
                                            hipErrorNoBinaryForGpu,
                                            hipErrorAlreadyAcquired,
                                            hipErrorNotMapped,
                                            hipErrorNotMappedAsArray,
                                            hipErrorNotMappedAsPointer,
                                            hipErrorECCNotCorrectable,
                                            hipErrorUnsupportedLimit,
                                            hipErrorContextAlreadyInUse,
                                            hipErrorPeerAccessUnsupported,
                                            hipErrorInvalidKernelFile,
                                            hipErrorInvalidGraphicsContext,
                                            hipErrorInvalidSource,
                                            hipErrorFileNotFound,
                                            hipErrorSharedObjectSymbolNotFound,
                                            hipErrorSharedObjectInitFailed,
                                            hipErrorOperatingSystem,
                                            hipErrorInvalidHandle,
                                            hipErrorIllegalState,
                                            hipErrorNotFound,
                                            hipErrorNotReady,
                                            hipErrorIllegalAddress,
                                            hipErrorLaunchOutOfResources,
                                            hipErrorLaunchTimeOut,
                                            hipErrorPeerAccessAlreadyEnabled,
                                            hipErrorPeerAccessNotEnabled,
                                            hipErrorSetOnActiveProcess,
                                            hipErrorContextIsDestroyed,
                                            hipErrorAssert,
                                            hipErrorHostMemoryAlreadyRegistered,
                                            hipErrorHostMemoryNotRegistered,
                                            hipErrorLaunchFailure,
                                            hipErrorCooperativeLaunchTooLarge,
                                            hipErrorNotSupported,
                                            hipErrorStreamCaptureUnsupported,
                                            hipErrorStreamCaptureInvalidated,
                                            hipErrorStreamCaptureMerge,
                                            hipErrorStreamCaptureUnmatched,
                                            hipErrorStreamCaptureUnjoined,
                                            hipErrorStreamCaptureIsolation,
                                            hipErrorStreamCaptureImplicit,
                                            hipErrorCapturedEvent,
                                            hipErrorStreamCaptureWrongThread,
                                            hipErrorGraphExecUpdateFailure,
                                            hipErrorUnknown,
                                            hipErrorRuntimeMemory,
                                            hipErrorRuntimeOther};
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

#include "hip_internal.hpp"

hipError_t hipGetLastError()
{
  HIP_INIT_API(hipGetLastError);
  hipError_t err = hip::g_lastError;
  hip::g_lastError = hipSuccess;
  return err;
}

hipError_t hipPeekAtLastError()
{
  HIP_INIT_API(hipPeekAtLastError);
  hipError_t err = hip::g_lastError;
  HIP_RETURN(err);
}

const char *hipGetErrorName(hipError_t hip_error)
{
  switch (hip_error) {
    case hipSuccess:
        return "hipSuccess";
    case hipErrorInvalidValue:
        return "hipErrorInvalidValue";
    case hipErrorOutOfMemory:
        return "hipErrorOutOfMemory";
    case hipErrorNotInitialized:
        return "hipErrorNotInitialized";
    case hipErrorDeinitialized:
        return "hipErrorDeinitialized";
    case hipErrorProfilerDisabled:
        return "hipErrorProfilerDisabled";
    case hipErrorProfilerNotInitialized:
        return "hipErrorProfilerNotInitialized";
    case hipErrorProfilerAlreadyStarted:
        return "hipErrorProfilerAlreadyStarted";
    case hipErrorProfilerAlreadyStopped:
        return "hipErrorProfilerAlreadyStopped";
    case hipErrorInvalidConfiguration:
        return "hipErrorInvalidConfiguration";
    case hipErrorInvalidSymbol:
        return "hipErrorInvalidSymbol";
    case hipErrorInvalidDevicePointer:
        return "hipErrorInvalidDevicePointer";
    case hipErrorInvalidMemcpyDirection:
        return "hipErrorInvalidMemcpyDirection";
    case hipErrorInsufficientDriver:
        return "hipErrorInsufficientDriver";
    case hipErrorMissingConfiguration:
        return "hipErrorMissingConfiguration";
    case hipErrorPriorLaunchFailure:
        return "hipErrorPriorLaunchFailure";
    case hipErrorInvalidDeviceFunction:
        return "hipErrorInvalidDeviceFunction";
    case hipErrorNoDevice:
        return "hipErrorNoDevice";
    case hipErrorInvalidDevice:
        return "hipErrorInvalidDevice";
    case hipErrorInvalidPitchValue:
        return "hipErrorInvalidPitchValue";
    case hipErrorInvalidImage:
        return "hipErrorInvalidImage";
    case hipErrorInvalidContext:
        return "hipErrorInvalidContext";
    case hipErrorContextAlreadyCurrent:
        return "hipErrorContextAlreadyCurrent";
    case hipErrorMapFailed:
        return "hipErrorMapFailed";
    case hipErrorUnmapFailed:
        return "hipErrorUnmapFailed";
    case hipErrorArrayIsMapped:
        return "hipErrorArrayIsMapped";
    case hipErrorAlreadyMapped:
        return "hipErrorAlreadyMapped";
    case hipErrorNoBinaryForGpu:
        return "hipErrorNoBinaryForGpu";
    case hipErrorAlreadyAcquired:
        return "hipErrorAlreadyAcquired";
    case hipErrorNotMapped:
        return "hipErrorNotMapped";
    case hipErrorNotMappedAsArray:
        return "hipErrorNotMappedAsArray";
    case hipErrorNotMappedAsPointer:
        return "hipErrorNotMappedAsPointer";
    case hipErrorECCNotCorrectable:
        return "hipErrorECCNotCorrectable";
    case hipErrorUnsupportedLimit:
        return "hipErrorUnsupportedLimit";
    case hipErrorContextAlreadyInUse:
        return "hipErrorContextAlreadyInUse";
    case hipErrorPeerAccessUnsupported:
        return "hipErrorPeerAccessUnsupported";
    case hipErrorInvalidKernelFile:
        return "hipErrorInvalidKernelFile";
    case hipErrorInvalidGraphicsContext:
        return "hipErrorInvalidGraphicsContext";
    case hipErrorInvalidSource:
        return "hipErrorInvalidSource";
    case hipErrorFileNotFound:
        return "hipErrorFileNotFound";
    case hipErrorSharedObjectSymbolNotFound:
        return "hipErrorSharedObjectSymbolNotFound";
    case hipErrorSharedObjectInitFailed:
        return "hipErrorSharedObjectInitFailed";
    case hipErrorOperatingSystem:
        return "hipErrorOperatingSystem";
    case hipErrorInvalidHandle:
        return "hipErrorInvalidHandle";
    case hipErrorNotFound:
        return "hipErrorNotFound";
    case hipErrorNotReady:
        return "hipErrorNotReady";
    case hipErrorIllegalAddress:
        return "hipErrorIllegalAddress";
    case hipErrorLaunchOutOfResources:
        return "hipErrorLaunchOutOfResources";
    case hipErrorLaunchTimeOut:
        return "hipErrorLaunchTimeOut";
    case hipErrorPeerAccessAlreadyEnabled:
        return "hipErrorPeerAccessAlreadyEnabled";
    case hipErrorPeerAccessNotEnabled:
        return "hipErrorPeerAccessNotEnabled";
    case hipErrorSetOnActiveProcess:
        return "hipErrorSetOnActiveProcess";
    case hipErrorAssert:
        return "hipErrorAssert";
    case hipErrorHostMemoryAlreadyRegistered:
        return "hipErrorHostMemoryAlreadyRegistered";
    case hipErrorHostMemoryNotRegistered:
        return "hipErrorHostMemoryNotRegistered";
    case hipErrorLaunchFailure:
        return "hipErrorLaunchFailure";
    case hipErrorNotSupported:
        return "hipErrorNotSupported";
    case hipErrorUnknown:
        return "hipErrorUnknown";
    case hipErrorRuntimeMemory:
        return "hipErrorRuntimeMemory";
    case hipErrorRuntimeOther:
        return "hipErrorRuntimeOther";
    case hipErrorCooperativeLaunchTooLarge:
        return "hipErrorCooperativeLaunchTooLarge";
    case hipErrorTbd:
        return "hipErrorTbd";
    default:
        return "hipErrorUnknown";
    };
}

const char *hipGetErrorString(hipError_t hip_error)
{
  return hipGetErrorName(hip_error);
}


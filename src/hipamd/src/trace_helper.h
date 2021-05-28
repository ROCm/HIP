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

#pragma once

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
//---
// Helper functions to convert HIP function arguments into strings.
// Handles POD data types as well as enumerations (ie hipMemcpyKind).
// The implementation uses C++11 variadic templates and template specialization.
// The hipMemcpyKind example below is a good example that shows how to implement conversion for a
// new HSA type.


// Handy macro to convert an enumeration to a stringified version of same:
#define CASE_STR(x)                                                                                \
    case x:                                                                                        \
        return #x;

inline const char* ihipErrorString(hipError_t hip_error) {
    switch (hip_error) {
        CASE_STR(hipSuccess);
        CASE_STR(hipErrorOutOfMemory);
        CASE_STR(hipErrorNotInitialized);
        CASE_STR(hipErrorDeinitialized);
        CASE_STR(hipErrorProfilerDisabled);
        CASE_STR(hipErrorProfilerNotInitialized);
        CASE_STR(hipErrorProfilerAlreadyStarted);
        CASE_STR(hipErrorProfilerAlreadyStopped);
        CASE_STR(hipErrorInvalidImage);
        CASE_STR(hipErrorInvalidContext);
        CASE_STR(hipErrorContextAlreadyCurrent);
        CASE_STR(hipErrorMapFailed);
        CASE_STR(hipErrorUnmapFailed);
        CASE_STR(hipErrorArrayIsMapped);
        CASE_STR(hipErrorAlreadyMapped);
        CASE_STR(hipErrorNoBinaryForGpu);
        CASE_STR(hipErrorAlreadyAcquired);
        CASE_STR(hipErrorNotMapped);
        CASE_STR(hipErrorNotMappedAsArray);
        CASE_STR(hipErrorNotMappedAsPointer);
        CASE_STR(hipErrorECCNotCorrectable);
        CASE_STR(hipErrorUnsupportedLimit);
        CASE_STR(hipErrorContextAlreadyInUse);
        CASE_STR(hipErrorPeerAccessUnsupported);
        CASE_STR(hipErrorInvalidKernelFile);
        CASE_STR(hipErrorInvalidGraphicsContext);
        CASE_STR(hipErrorInvalidSource);
        CASE_STR(hipErrorFileNotFound);
        CASE_STR(hipErrorSharedObjectSymbolNotFound);
        CASE_STR(hipErrorSharedObjectInitFailed);
        CASE_STR(hipErrorOperatingSystem);
        CASE_STR(hipErrorSetOnActiveProcess);
        CASE_STR(hipErrorInvalidHandle);
        CASE_STR(hipErrorNotFound);
        CASE_STR(hipErrorIllegalAddress);
        CASE_STR(hipErrorMissingConfiguration);
        CASE_STR(hipErrorLaunchFailure);
        CASE_STR(hipErrorPriorLaunchFailure);
        CASE_STR(hipErrorLaunchTimeOut);
        CASE_STR(hipErrorLaunchOutOfResources);
        CASE_STR(hipErrorInvalidDeviceFunction);
        CASE_STR(hipErrorInvalidConfiguration);
        CASE_STR(hipErrorInvalidDevice);
        CASE_STR(hipErrorInvalidValue);
        CASE_STR(hipErrorInvalidPitchValue);
        CASE_STR(hipErrorInvalidDevicePointer);
        CASE_STR(hipErrorInvalidMemcpyDirection);
        CASE_STR(hipErrorUnknown);
        CASE_STR(hipErrorNotReady);
        CASE_STR(hipErrorNoDevice);
        CASE_STR(hipErrorPeerAccessAlreadyEnabled);
        CASE_STR(hipErrorPeerAccessNotEnabled);
        CASE_STR(hipErrorRuntimeMemory);
        CASE_STR(hipErrorRuntimeOther);
        CASE_STR(hipErrorHostMemoryAlreadyRegistered);
        CASE_STR(hipErrorHostMemoryNotRegistered);
        CASE_STR(hipErrorTbd);
        default:
            return "hipErrorUnknown";
    };
};

// Building block functions:
template <typename T>
inline std::string ToHexString(T v) {
    std::ostringstream ss;
    ss << "0x" << std::hex << v;
    return ss.str();
};

template <typename T>
inline std::string ToString(T* v) {
    std::ostringstream ss;
    if (v == NULL) {
        ss << "char array:<null>";
    } else {
        ss << v;
    }
    return ss.str();
};

template <typename T>
inline std::string ToString(T** v) {
    std::ostringstream ss;
    if (v == NULL) {
        ss << "char array:<null>";
    } else {
        ss << v;
    }
    return ss.str();
};

//---
// Template overloads for ToString to handle specific types

// This is the default which works for most types:
template <typename T>
inline std::string ToString(T v) {
    std::ostringstream ss;
    ss << v;
    return ss.str();
};

template <>
inline std::string ToString(hipFunction_t v) {
    std::ostringstream ss;
    ss << "0x" << std::hex << static_cast<void*>(v);
    return ss.str();
};

//  hipEvent_t specialization. TODO - maybe add an event ID for debug?
template <>
inline std::string ToString(hipEvent_t v) {
    std::ostringstream ss;
    ss << "event:" << std::hex << static_cast<void*>(v);
    return ss.str();
};
//  hipStream_t
template <>
inline std::string ToString(hipStream_t v) {
    std::ostringstream ss;
    if (v == NULL) {
        ss << "stream:<null>";
    } else {
        ss << "stream:" << std::hex << static_cast<void*>(v);
    }

    return ss.str();
};

//  hipCtx_t
template <>
inline std::string ToString(hipCtx_t v) {
    std::ostringstream ss;
    if (v == NULL) {
        ss << "context:<null>";
    } else {
        ss << "context:" << std::hex << static_cast<void*>(v);
    }

    return ss.str();
};

//  hipPitchedPtr
template <>
inline std::string ToString(hipPitchedPtr v) {
    std::ostringstream ss;
    ss << "pitchPtr:" << std::hex << static_cast<void*>(v.ptr);
    return ss.str();
};

//  hipMemcpyKind specialization
template <>
inline std::string ToString(hipMemcpyKind v) {
    switch (v) {
        CASE_STR(hipMemcpyHostToHost);
        CASE_STR(hipMemcpyHostToDevice);
        CASE_STR(hipMemcpyDeviceToHost);
        CASE_STR(hipMemcpyDeviceToDevice);
        CASE_STR(hipMemcpyDefault);
        default:
            return ToHexString(v);
    };
};

template <>
inline std::string ToString(hipFuncCache_t v) {
    switch (v) {
        CASE_STR(hipFuncCachePreferNone);
        CASE_STR(hipFuncCachePreferShared);
        CASE_STR(hipFuncCachePreferL1);
        CASE_STR(hipFuncCachePreferEqual);
        default:
            return ToHexString(v);
    };
};

template <>
inline std::string ToString(hipSharedMemConfig v) {
    switch (v) {
        CASE_STR(hipSharedMemBankSizeDefault);
        CASE_STR(hipSharedMemBankSizeFourByte);
        CASE_STR(hipSharedMemBankSizeEightByte);
        default:
            return ToHexString(v);
    };
};

template <>
inline std::string ToString(hipError_t v) {
    return ihipErrorString(v);
};

// Catch empty arguments case
inline std::string ToString() { return (""); }


//---
// C++11 variadic template - peels off first argument, converts to string, and calls itself again to
// peel the next arg. Strings are automatically separated by comma+space.
template <typename T, typename... Args>
inline std::string ToString(T first, Args... args) {
    return ToString(first) + ", " + ToString(args...);
}


/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

//#pragma once

#ifndef TRACE_HELPER_H
#define TRACE_HELPER_H

#include <iostream>
#include <iomanip>
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


// Building block functions:
template <typename T>
inline std::string ToHexString(T v) {
    std::ostringstream ss;
    ss << "0x" << std::hex << v;
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


//  hipEvent_t specialization. TODO - maybe add an event ID for debug?
template <>
inline std::string ToString(hipEvent_t v) {
    std::ostringstream ss;
    ss << v;
    return ss.str();
};
//  hipStream_t
template <>
inline std::string ToString(hipStream_t v) {
    std::ostringstream ss;
    if (v == NULL) {
        ss << "stream:<null>";
    } else {
        ss << *v;
    }

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

#endif

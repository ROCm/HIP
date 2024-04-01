
/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
// We haven't checked which filesystem to include yet
#ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
// Check for feature test macro for <filesystem>
#if defined(__cpp_lib_filesystem)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
// Check for feature test macro for <experimental/filesystem>
#elif defined(__cpp_lib_experimental_filesystem)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1
// We can't check if headers exist...
// Let's assume experimental to be safe
#elif !defined(__has_include)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1
// Check if the header "<filesystem>" exists
#elif __has_include(<filesystem>)
// If we're compiling on Visual Studio and are not compiling with C++17,
// we need to use experimental
#ifdef _MSC_VER
// Check and include header that defines "_HAS_CXX17"
#if __has_include(<yvals_core.h>)
#include <yvals_core.h>

// Check for enabled C++17 support
#if defined(_HAS_CXX17) && _HAS_CXX17
// We're using C++17, so let's use the normal version
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
#endif

#endif

// If the marco isn't defined yet, that means any of the other
// VS specific checks failed, so we need to use experimental
#ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1
#endif

// Not on Visual Studio. Let's use the normal version
#else  // #ifdef _MSC_VER
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
#endif

// Check if the header "<filesystem>" exists
#elif __has_include(<experimental/filesystem>)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// Fail if neither header is available with a nice error message
#else
#error Could not find system header "<filesystem>" ||
       "<experimental/filesystem>"
#endif

// We priously determined that we need the exprimental version
#if INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
// Include it
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING 1;
#include <experimental/filesystem>
// We need the alias from std::experimental::filesystem to std::filesystem
namespace fs = std::experimental::filesystem;
// We have a decent compiler and can use the normal version
#else
// Include it
#include <filesystem>
namespace fs = std::filesystem;
#endif

#endif  // #ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
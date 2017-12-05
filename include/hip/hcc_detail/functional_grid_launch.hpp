/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "code_object_bundle.hpp"
#include "concepts.hpp"
#include "helpers.hpp"
#include "program_state.hpp"

#include "hc.hpp"
#include "hip/hip_hcc.h"
#include "hip_runtime.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hip_impl
{
    template<
        typename T,
        typename std::enable_if<std::is_integral<T>{}>::type* = nullptr>
    inline
    T round_up_to_next_multiple_nonnegative(T x, T y)
    {
        T tmp = x + y - 1;
        return tmp - tmp % y;
    }

    inline
    std::vector<std::uint8_t> make_kernarg()
    {
        return {};
    }

    inline
    std::vector<std::uint8_t> make_kernarg(std::vector<std::uint8_t> kernarg)
    {
        return kernarg;
    }

    template<typename T>
    inline
    std::vector<std::uint8_t> make_kernarg(std::vector<uint8_t> kernarg, T x)
    {
        kernarg.resize(
            round_up_to_next_multiple_nonnegative(kernarg.size(), alignof(T)) +
            sizeof(T));

        new (kernarg.data() + kernarg.size() - sizeof(T)) T{std::move(x)};

        return kernarg;
    }

    template<typename T, typename... Ts>
    inline
    std::vector<std::uint8_t> make_kernarg(
        std::vector<std::uint8_t> kernarg, T x, Ts... xs)
    {
        return make_kernarg(
            make_kernarg(std::move(kernarg), std::move(x)), std::move(xs)...);
    }

    template<typename... Ts>
    inline
    std::vector<std::uint8_t> make_kernarg(Ts... xs)
    {
        std::vector<std::uint8_t> kernarg;
        kernarg.reserve(sizeof(std::tuple<Ts...>));

        return make_kernarg(std::move(kernarg), std::move(xs)...);
    }

    void hipLaunchKernelGGLImpl(
        std::uintptr_t function_address,
        const dim3& numBlocks,
        const dim3& dimBlocks,
        std::uint32_t sharedMemBytes,
        hipStream_t stream,
        void** kernarg);
} // Namespace hip_impl.

template<typename... Args, typename F = void (*)(Args...)>
inline
void hipLaunchKernelGGL(
    F kernel,
    const dim3& numBlocks,
    const dim3& dimBlocks,
    std::uint32_t sharedMemBytes,
    hipStream_t stream,
    Args... args)
{
    auto kernarg = hip_impl::make_kernarg(std::move(args)...);
    std::size_t kernarg_size = kernarg.size();

    void* config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(),
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernarg_size,
      HIP_LAUNCH_PARAM_END
    };

    hip_impl::hipLaunchKernelGGLImpl(
        reinterpret_cast<std::uintptr_t>(kernel),
        numBlocks,
        dimBlocks,
        sharedMemBytes,
        stream,
        &config[0]);
}

template<typename... Args, typename F = void (*)(hipLaunchParm, Args...)>
inline
void hipLaunchKernel(
    F kernel,
    const dim3& numBlocks,
    const dim3& dimBlocks,
    std::uint32_t groupMemBytes,
    hipStream_t stream,
    Args... args)
{
    hipLaunchKernelGGL(
        kernel,
        numBlocks,
        dimBlocks,
        groupMemBytes,
        stream,
        hipLaunchParm{},
        std::move(args)...);
}


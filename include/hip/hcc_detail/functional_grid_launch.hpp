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
#include <cstring>
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

namespace hip_impl {
template <typename T, typename std::enable_if<std::is_integral<T>{}>::type* = nullptr>
inline T round_up_to_next_multiple_nonnegative(T x, T y) {
    T tmp = x + y - 1;
    return tmp - tmp % y;
}

template <
    std::size_t n,
    typename... Ts,
    typename std::enable_if<n == sizeof...(Ts)>::type* = nullptr>
inline std::vector<std::uint8_t> make_kernarg(
    const std::tuple<Ts...>&,
    const std::vector<std::pair<std::size_t, std::size_t>>&,
    std::vector<std::uint8_t> kernarg) {
    return kernarg;
}

template <
    std::size_t n,
    typename... Ts,
    typename std::enable_if<n != sizeof...(Ts)>::type* = nullptr>
inline std::vector<std::uint8_t> make_kernarg(
    const std::tuple<Ts...>& formals,
    const std::vector<std::pair<std::size_t, std::size_t>>& size_align,
    std::vector<std::uint8_t> kernarg) {
    using T = typename std::tuple_element<n, std::tuple<Ts...>>::type;

    static_assert(
        !std::is_reference<T>{},
        "A __global__ function cannot have a reference as one of its "
            "arguments.");
    #if defined(HIP_STRICT)
        static_assert(
            std::is_trivially_copyable<T>{},
            "Only TriviallyCopyable types can be arguments to a __global__ "
                "function");
    #endif

    kernarg.resize(round_up_to_next_multiple_nonnegative(
        kernarg.size(), size_align[n].second) + size_align[n].first);

    std::memcpy(
        kernarg.data() + kernarg.size() - size_align[n].first,
        &std::get<n>(formals),
        size_align[n].first);

    return make_kernarg<n + 1>(formals, size_align, std::move(kernarg));
}

template <typename... Formals, typename... Actuals>
inline std::vector<std::uint8_t> make_kernarg(
    void (*kernel)(Formals...), std::tuple<Actuals...> actuals) {
    static_assert(sizeof...(Formals) == sizeof...(Actuals),
        "The count of formal arguments must match the count of actuals.");

    if (sizeof...(Formals) == 0) return {};

    auto it = function_names().find(reinterpret_cast<std::uintptr_t>(kernel));
    if (it == function_names().cend()) {
        it =
            function_names(true).find(reinterpret_cast<std::uintptr_t>(kernel));
        if (it == function_names().cend()) {
            throw std::runtime_error{"Undefined __global__ function."};
        }
    }

    auto it1 = kernargs().find(it->second);
    if (it1 == kernargs().end()) {
        it1 = kernargs(true).find(it->second);

        if (it1 == kernargs().end()) {
            throw std::runtime_error{
                "Missing metadata for __global__ function: " + it->second};
        }
    }

    std::tuple<Formals...> to_formals{std::move(actuals)};
    std::vector<std::uint8_t> kernarg;
    kernarg.reserve(sizeof(to_formals));

    return make_kernarg<0>(to_formals, it1->second, std::move(kernarg));
}

void hipLaunchKernelGGLImpl(std::uintptr_t function_address, const dim3& numBlocks,
                            const dim3& dimBlocks, std::uint32_t sharedMemBytes, hipStream_t stream,
                            void** kernarg);
}  // Namespace hip_impl.

template <typename... Args, typename F = void (*)(Args...)>
inline void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                               std::uint32_t sharedMemBytes, hipStream_t stream, Args... args) {
    auto kernarg = hip_impl::make_kernarg(
        kernel, std::tuple<Args...>{std::move(args)...});
    std::size_t kernarg_size = kernarg.size();

    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &kernarg_size, HIP_LAUNCH_PARAM_END};

    hip_impl::hipLaunchKernelGGLImpl(reinterpret_cast<std::uintptr_t>(kernel), numBlocks, dimBlocks,
                                     sharedMemBytes, stream, &config[0]);
}

template <typename... Args, typename F = void (*)(hipLaunchParm, Args...)>
[[deprecated("hipLaunchKernel is deprecated and will be removed in the next "
             "version of HIP; please upgrade to hipLaunchKernelGGL.")]]
inline void hipLaunchKernel(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                            std::uint32_t groupMemBytes, hipStream_t stream, Args... args) {
    hipLaunchKernelGGL(kernel, numBlocks, dimBlocks, groupMemBytes, stream, hipLaunchParm{},
                       std::move(args)...);
}
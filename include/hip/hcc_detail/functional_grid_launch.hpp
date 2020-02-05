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

#include "concepts.hpp"
#include "helpers.hpp"
#include "program_state.hpp"
#include "hip_runtime_api.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

hipError_t ihipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList, int numDevices,
                                               unsigned int flags, hip_impl::program_state& ps);

hipError_t ihipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX, void** kernelParams,
                unsigned int sharedMemBytes, hipStream_t stream, hip_impl::program_state& ps);

hipError_t ihipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int  numDevices,
                unsigned int  flags, hip_impl::program_state& ps);




#pragma GCC visibility push(hidden)

namespace hip_impl {
template <typename T, typename std::enable_if<std::is_integral<T>{}>::type* = nullptr>
inline T round_up_to_next_multiple_nonnegative(T x, T y) {
    T tmp = x + y - 1;
    return tmp - tmp % y;
}

template <std::size_t n>
inline std::size_t make_kernarg(const kernargs_size_align&, char*,
                                std::size_t sz) {
    return sz;
}

template <unsigned int n, typename T, typename... Ts>
inline std::size_t make_kernarg(const kernargs_size_align& size_align,
                                char* kernarg, std::size_t sz, T x, Ts... xs) {
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

    sz = round_up_to_next_multiple_nonnegative(sz, size_align.alignment(n)) +
         size_align.size(n);

    std::memcpy(kernarg + sz - size_align.size(n), &x, size_align.size(n));

    return make_kernarg<n + 1>(size_align, kernarg, sz, std::move(xs)...);
}

template <size_t n, typename... Formals, typename... Actuals>
inline
std::size_t make_kernarg(void (*kernel)(Formals...), char (&kernarg)[n],
                         Actuals... actuals) {
    static_assert(sizeof...(Formals) == sizeof...(Actuals),
        "The count of formal arguments must match the count of actuals.");

    if (sizeof...(Formals) == 0) return {};

    auto& ps = hip_impl::get_program_state();
    return make_kernarg<0u>(
        ps.get_kernargs_size_align(reinterpret_cast<std::uintptr_t>(kernel)),
        kernarg,
        0u,
        (Formals)actuals...);
}

HIP_INTERNAL_EXPORTED_API hsa_agent_t target_agent(hipStream_t stream);

inline
__attribute__((visibility("hidden")))
void hipLaunchKernelGGLImpl(
    std::uintptr_t function_address,
    const dim3& numBlocks,
    const dim3& dimBlocks,
    std::uint32_t sharedMemBytes,
    hipStream_t stream,
    void** kernarg) {

    const auto& kd = get_program_state()
        .kernel_descriptor(function_address, target_agent(stream));

    hipModuleLaunchKernel(kd, numBlocks.x, numBlocks.y, numBlocks.z,
                          dimBlocks.x, dimBlocks.y, dimBlocks.z, sharedMemBytes,
                          stream, nullptr, kernarg);
}
} // Namespace hip_impl.


template <typename F>
inline
hipError_t hipOccupancyMaxPotentialBlockSize(uint32_t* gridSize, uint32_t* blockSize,
    F kernel, size_t dynSharedMemPerBlk, uint32_t blockSizeLimit) {

    using namespace hip_impl;

    hip_impl::hip_init();
    auto f = get_program_state().kernel_descriptor(
        reinterpret_cast<std::uintptr_t>(kernel), target_agent(nullptr));

    return hipOccupancyMaxPotentialBlockSize(gridSize, blockSize, f,
                                      dynSharedMemPerBlk, blockSizeLimit);
}

template <typename F>
inline
hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(uint32_t* numBlocks, F kernel,
    uint32_t blockSize, size_t dynSharedMemPerBlk) {

    using namespace hip_impl;

    hip_impl::hip_init();
    auto f = get_program_state().kernel_descriptor(
        reinterpret_cast<std::uintptr_t>(kernel), target_agent(nullptr));

    return hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize, dynSharedMemPerBlk);
}

template <typename... Args, typename F = void (*)(Args...)>
inline
void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                        std::uint32_t sharedMemBytes, hipStream_t stream,
                        Args... args) {
    hip_impl::hip_init();

    char kernarg[sizeof(std::tuple<Args...>)];
    auto n = hip_impl::make_kernarg(kernel, kernarg, std::move(args)...);

    void* config[]{
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernarg,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &n,
        HIP_LAUNCH_PARAM_END};

    hip_impl::hipLaunchKernelGGLImpl(reinterpret_cast<std::uintptr_t>(kernel),
                                     numBlocks, dimBlocks, sharedMemBytes,
                                     stream, &config[0]);
}

inline
__attribute__((visibility("hidden")))
hipError_t hipExtLaunchMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                              int  numDevices, unsigned int  flags) {
    hip_impl::hip_init();
    auto& ps = hip_impl::get_program_state();
    return ihipExtLaunchMultiKernelMultiDevice(launchParamsList, numDevices, flags, ps);

}

template <typename F>
inline
__attribute__((visibility("hidden")))
hipError_t hipLaunchCooperativeKernel(F f, dim3 gridDim, dim3 blockDimX, void** kernelParams,
                unsigned int sharedMemBytes, hipStream_t stream) {

    hip_impl::hip_init();
    auto& ps = hip_impl::get_program_state();
    return ihipLaunchCooperativeKernel(reinterpret_cast<void*>(f), gridDim, blockDimX, kernelParams, sharedMemBytes, stream, ps);
}

inline
__attribute__((visibility("hidden")))
hipError_t hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int  numDevices,
                unsigned int  flags) {

    hip_impl::hip_init();
    auto& ps = hip_impl::get_program_state();
    return ihipLaunchCooperativeKernelMultiDevice(launchParamsList, numDevices, flags, ps);
}

#pragma GCC visibility pop

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

#ifndef HIP_INCLUDE_HIP_HIP_EXT_H
#define HIP_INCLUDE_HIP_HIP_EXT_H
#include "hip/hip_runtime.h"
#if defined(__cplusplus)
#include <tuple>
#include <type_traits>
#endif

/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
 to kernelparams or extra
 *
 * @param [in[ f     Kernel to launch.
 * @param [in] gridDimX  X grid dimension specified in work-items
 * @param [in] gridDimY  Y grid dimension specified in work-items
 * @param [in] gridDimZ  Z grid dimension specified in work-items
 * @param [in] blockDimX X block dimensions specified in work-items
 * @param [in] blockDimY Y grid dimension specified in work-items
 * @param [in] blockDimZ Z grid dimension specified in work-items
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel.  The
 kernel can access this with HIP_DYNAMIC_SHARED.
 * @param [in] stream Stream where the kernel should be dispatched.  May be 0, in which case th
 default stream is used with associated synchronization rules.
 * @param [in] kernelParams
 * @param [in] extra     Pointer to kernel arguments.   These are passed directly to the kernel and
 must be in the memory layout and alignment expected by the kernel.
 * @param [in] startEvent  If non-null, specified event will be updated to track the start time of
 the kernel launch.  The event must be created before calling this API.
 * @param [in] stopEvent   If non-null, specified event will be updated to track the stop time of
 the kernel launch.  The event must be created before calling this API.
 *
 * @returns hipSuccess, hipInvalidDevice, hipErrorNotInitialized, hipErrorInvalidValue
 *
 * @warning kernellParams argument is not yet implemented in HIP. Please use extra instead. Please
 refer to hip_porting_driver_api.md for sample usage.
 * HIP/ROCm actually updates the start event when the associated kernel completes.
 * Currently, timing between startEvent and stopEvent does not include the time it takes to perform
 * a system scope release / cache flush - only the time it takes to issues writes to cache.
 */
HIP_PUBLIC_API
hipError_t hipExtModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent = nullptr,
                                    hipEvent_t stopEvent = nullptr,
                                    uint32_t flags = 0);

HIP_PUBLIC_API
hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent = nullptr,
                                    hipEvent_t stopEvent = nullptr)
                                    __attribute__((deprecated("use hipExtModuleLaunchKernel instead")));

#if defined(__HIP_ROCclr__) && defined(__cplusplus)

extern "C" hipError_t hipExtLaunchKernel(const void* function_address, dim3 numBlocks,
                                         dim3 dimBlocks, void** args, size_t sharedMemBytes,
                                         hipStream_t stream, hipEvent_t startEvent,
                                         hipEvent_t stopEvent, int flags);

template <typename... Args, typename F = void (*)(Args...)>
inline void hipExtLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                                  std::uint32_t sharedMemBytes, hipStream_t stream,
                                  hipEvent_t startEvent, hipEvent_t stopEvent, std::uint32_t flags,
                                  Args... args) {
    constexpr size_t count = sizeof...(Args);
    auto tup_ = std::tuple<Args...>{args...};
    auto tup = validateArgsCountType(kernel, tup_);
    void* _Args[count];
    pArgs<0>(tup, _Args);

    auto k = reinterpret_cast<void*>(kernel);
    hipExtLaunchKernel(k, numBlocks, dimBlocks, _Args, sharedMemBytes, stream, startEvent,
                       stopEvent, (int)flags);
}
#elif defined(__HIP_PLATFORM_HCC__) && GENERIC_GRID_LAUNCH == 1 && defined(__HCC__)
//kernel_descriptor and hip_impl::make_kernarg are in "grid_launch_GGL.hpp"

namespace hip_impl {
inline
__attribute__((visibility("hidden")))
void hipExtLaunchKernelGGLImpl(
    std::uintptr_t function_address,
    const dim3& numBlocks,
    const dim3& dimBlocks,
    std::uint32_t sharedMemBytes,
    hipStream_t stream,
    hipEvent_t startEvent,
    hipEvent_t stopEvent,
    std::uint32_t flags,
    void** kernarg) {

    const auto& kd = hip_impl::get_program_state()
        .kernel_descriptor(function_address, target_agent(stream));

    hipExtModuleLaunchKernel(kd, numBlocks.x * dimBlocks.x,
                             numBlocks.y * dimBlocks.y,
                             numBlocks.z * dimBlocks.z,
                             dimBlocks.x, dimBlocks.y, dimBlocks.z,
                             sharedMemBytes, stream, nullptr, kernarg,
                             startEvent, stopEvent, flags);
}
}  // namespace hip_impl

template <typename... Args, typename F = void (*)(Args...)>
inline
void hipExtLaunchKernelGGL(F kernel, const dim3& numBlocks,
                           const dim3& dimBlocks, std::uint32_t sharedMemBytes,
                           hipStream_t stream, hipEvent_t startEvent,
                           hipEvent_t stopEvent, std::uint32_t flags,
                           Args... args) {
    hip_impl::hip_init();
    auto kernarg =
        hip_impl::make_kernarg(kernel, std::tuple<Args...>{std::move(args)...});
    std::size_t kernarg_size = kernarg.size();

    void* config[]{
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kernarg.data(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &kernarg_size,
        HIP_LAUNCH_PARAM_END};

    hip_impl::hipExtLaunchKernelGGLImpl(reinterpret_cast<std::uintptr_t>(kernel),
                                        numBlocks, dimBlocks, sharedMemBytes,
                                        stream, startEvent, stopEvent, flags,
                                        &config[0]);
}
#endif // !__HIP_ROCclr__ && defined(__cplusplus)

// doxygen end AMD-specific features
/**
 * @}
 */
#endif  // #iidef HIP_INCLUDE_HIP_HIP_EXT_H

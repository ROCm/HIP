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

#pragma once

#include <hip_test_common.hh>

namespace hip {
inline namespace stream {

const hipStream_t nullStream = nullptr;
const hipStream_t streamPerThread = hipStreamPerThread;

/**
 * @brief Kernel that signals a semaphore to change value from 0 to 1.
 *
 * @param semaphore the semaphore that needs to be signaled.
 */
__global__ void signaling_kernel(int* semaphore = nullptr);

/**
 * @brief Kernel that busy waits until the specified semaphore goes from 0 to 1.
 *
 * @param semaphore the semaphore to wait for.
 */
__global__ void waiting_kernel(int* semaphore = nullptr);

/**
 * @brief Creates a thread that runs a signaling_kernel on a non-blocking stream.
 * hipStreamNonBlocking is used here to avoid interfering with tests for the Null Stream.
 *
 * @param semaphore memory location to signal
 * @return std::thread thread that has to be joined after the testing is done.
 */
std::thread startSignalingThread(int* semaphore = nullptr);

// Checks stream for valid values of flags and priority
bool checkStream(hipStream_t stream);

// Checks stream for valid flags and a particular value of priority
bool checkStreamPriorityAndFlags(hipStream_t stream, int priority,
                                 unsigned int flags = hipStreamDefault);

}  // namespace stream
}  // namespace hip

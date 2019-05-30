/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

#define TAG_TO_ZERO 5

static uint
process_packets(hostcall_buffer_t *buffer, ulong F)
{
    WHEN_DEBUG(std::cout << "processing cptr: " << F << std::endl);
    ulong packet_index = get_ptr_index(F, buffer->index_size);
    WHEN_DEBUG(std::cout << "packet index: " << packet_index << std::endl);
    auto header = get_header(buffer, packet_index);
    WHEN_DEBUG(std::cout << "packet header: " << header << std::endl);

    if (get_ready_flag(header->control) == 0) {
        return __LINE__;
    }

    if (header->service != TEST_SERVICE)
        return __LINE__;

    __atomic_store_n(&header->control, reset_ready_flag(header->control),
                     std::memory_order_release);
    return 0;
}

static ulong
grab_ready_stack(hostcall_buffer_t *buffer)
{
    ulong *top = &buffer->ready_stack;
    ulong F = __atomic_load_n(top, std::memory_order_acquire);
    if (!F)
        return F;

    do {
        WHEN_DEBUG(std::cout << "trying to grab ready stack: " << std::hex << F
                             << std::endl);
    } while (!__atomic_compare_exchange_n(top, &F, 0,
                                          /* weak = */ false,
                                          std::memory_order_acquire,
                                          std::memory_order_relaxed));
    return F;
}

static void
consume_packets(uint *status, hostcall_buffer_t *buffer)
{
    *status = 0;

    WHEN_DEBUG(std::cout << "launched consumer" << std::endl);
    ulong signal_value = SIGNAL_INIT;
    ulong timeout = 1024 * 1024;

    while (true) {
        signal_value = wait_on_signal(buffer->doorbell, timeout, signal_value);

        ulong F = grab_ready_stack(buffer);
        WHEN_DEBUG(std::cout << "grabbed ready stack: " << F << std::endl);
        if (F) {
            *status = process_packets(buffer, F);
            if (*status != 0)
                return;
        }

        if (signal_value == SIGNAL_DONE) {
            *status = 0;
            return;
        }
    }

    *status = __LINE__;
    return;
}

__global__ void
kernelAbaWraparound(void *buffer)
{
    uint tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    ulong arg0 = tid;
    ulong arg1 = 0;
    ulong arg2 = 0;
    ulong arg3 = 0;
    ulong arg4 = 0;
    ulong arg5 = 0;
    ulong arg6 = 0;
    ulong arg7 = 0;

    for (int i = 0; i != TAG_TO_ZERO + 2; ++i) {
        __ockl_hostcall_internal(buffer, TEST_SERVICE, arg0, arg1, arg2, arg3,
                                arg4, arg5, arg6, arg7);
    }
}

uint
testAbaWraparound()
{
    uint num_blocks = 1;
    uint threads_per_block = 64;
    uint num_packets = 1;

    hsa_signal_t signal;
    if (hsa_signal_create(SIGNAL_INIT, 0, NULL, &signal) != HSA_STATUS_SUCCESS)
        return __LINE__;

    hostcall_buffer_t *buffer = createBuffer(num_packets, signal);
    if (!buffer)
        return __LINE__;
    buffer->free_stack =
        set_tag(buffer->free_stack,
                UINT64_MAX - TAG_TO_ZERO, buffer->index_size);

    hipLaunchKernelGGL(kernelAbaWraparound, dim3(num_blocks),
                       dim3(threads_per_block), 0, 0, buffer);
    hipEvent_t mark;
    HIPCHECK(hipEventCreate(&mark));
    HIPCHECK(hipEventRecord(mark));

    uint consumer_status;
    std::thread consumer_thread(consume_packets, &consumer_status,
                                buffer);

    bool timed_out = timeout(mark, 500);
    if (consumer_status)
        return consumer_status;
    if (timed_out) {
        std::cout << "timed out" << std::endl;
        return __LINE__;
    }

    work_done(buffer);
    consumer_thread.join();

    return 0;
}

int
main(int argc, char **argv)
{
    set_flags(argc, argv);
    runTest(testAbaWraparound);
    test_passed(__FILE__);
    return 0;
}

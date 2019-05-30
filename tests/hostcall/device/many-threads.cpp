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

#include <vector>

static void
push(ulong *top, ulong ptr, hostcall_buffer_t *buffer)
{
    ulong F = __atomic_load_n((ulong *)top, std::memory_order_relaxed);
    header_t *P = get_header(buffer, ptr);
    P->next = F;

    while (!__atomic_compare_exchange_n(top, &F, ptr,
                                        /* weak = */ false,
                                        std::memory_order_release,
                                        std::memory_order_relaxed)) {
        P->next = F;
    }

    WHEN_DEBUG(std::cout << "pushed free packet: " << ptr << std::endl);
}

static uint
handle_packet(ulong *retval, const ulong *payload)
{
    *retval = *payload + 1;
    return 0;
}

static uint
process_packets(hostcall_buffer_t *buffer, ulong F)
{
    WHEN_DEBUG(std::cout << "process packets starting with " << F << std::endl);
    std::vector<ulong> R;
    while (F) {
        WHEN_DEBUG(std::cout << "found cptr: " << F << std::endl);
        auto *P = get_header(buffer, F);
        R.push_back(F);
        F = P->next;
    }

    while (!R.empty()) {
        auto II = R.back();
        R.pop_back();

        WHEN_DEBUG(std::cout << "processing cptr: " << II << std::endl);
        ulong packet_index = get_ptr_index(II, buffer->index_size);
        WHEN_DEBUG(std::cout << "packet index: " << packet_index << std::endl);
        auto header = get_header(buffer, II);

        if (get_ready_flag(header->control) == 0) {
            return __LINE__;
        }

        if (header->service != TEST_SERVICE)
            return __LINE__;

        ulong activemask = header->activemask;
        WHEN_DEBUG(std::cout << "activemask: " << std::hex << activemask
                             << std::dec << std::endl);

        payload_t *payload = get_payload(buffer, II);
        for (uint wi = 0; wi != 64; ++wi) {
            ulong flag = activemask & ((ulong)1 << wi);
            if (flag == 0)
                continue;
            auto slot = payload->slots[wi];
            ulong retval;
            uint status = handle_packet(&retval, slot);
            if (status != 0) {
                return status;
            }
            *slot = retval;
        }

        WHEN_DEBUG(std::cout << "finished processing" << std::endl);

        __atomic_store_n(&header->control, reset_ready_flag(header->control),
                         std::memory_order_release);
    }

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
        WHEN_DEBUG(std::cout << "trying to grab ready stack: " << F
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
    WHEN_DEBUG(std::cout << "launched consumer" << std::endl);
    ulong signal_value = SIGNAL_INIT;
    ulong timeout = 1024 * 1024;

    while (true) {
        signal_value = wait_on_signal(buffer->doorbell, timeout, signal_value);

        ulong F = grab_ready_stack(buffer);
        WHEN_DEBUG(std::cout << "grabbed ready stack: " << F << std::endl);

        if (F) {
            *status = process_packets(buffer, F);
            if (*status != 0) {
                return;
            }
        }

        if (signal_value == SIGNAL_DONE) {
            *status = 0;
            return;
        }
    }

    return;
}

__global__ void
kernelManyThreads(void *buffer, ulong *retval)
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

    long2 result = {0, 0};
    if (tid % 71 != 1) {
        result.data = __ockl_hostcall_internal(buffer, TEST_SERVICE, arg0, arg1, arg2,
                                               arg3, arg4, arg5, arg6, arg7);
        retval[tid] = result.x;
    }
}

uint
testManyThreads()
{
    uint num_blocks = 5;
    uint threads_per_block = 1000;
    uint warps_per_block = (threads_per_block + 63) / 64;
    uint num_threads = num_blocks * threads_per_block;
    uint num_packets = warps_per_block * num_blocks;

    hsa_signal_t signal;
    if (hsa_signal_create(SIGNAL_INIT, 0, NULL, &signal) != HSA_STATUS_SUCCESS)
        return __LINE__;

    hostcall_buffer_t *buffer = createBuffer(num_packets, signal);
    if (!buffer)
        return __LINE__;

    void *retval_void;
    if (hipHostMalloc(&retval_void, 8 * num_threads) != hipSuccess)
        return __LINE__;
    uint64_t *retval = (uint64_t *)retval_void;
    for (uint i = 0; i != num_threads; ++i) {
        retval[i] = 0x23232323;
    }

    hipLaunchKernelGGL(kernelManyThreads, dim3(num_blocks),
                       dim3(threads_per_block), 0, 0, buffer, retval);
    hipEvent_t mark;
    HIPCHECK(hipEventCreate(&mark));
    HIPCHECK(hipEventRecord(mark));

    uint consumer_status;
    std::thread consumer_thread(consume_packets, &consumer_status, buffer);

    uint timeout_status = 0;
    using std::chrono::system_clock;
    system_clock::time_point start = system_clock::now();
    while (hipEventQuery(mark) != hipSuccess) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        system_clock::time_point now = system_clock::now();
        if (now - start > std::chrono::milliseconds(500)) {
            WHEN_DEBUG(std::cout << "host timed out" << std::endl);
            timeout_status = __LINE__;
            break;
        }
    }

    work_done(buffer);
    consumer_thread.join();

    if (consumer_status)
        return consumer_status;
    if (timeout_status)
        return timeout_status;

    for (uint ii = 0; ii != num_threads; ++ii) {
        ulong value = retval[ii];
        if (ii % 71 == 1) {
            if (value != 0x23232323)
                return __LINE__;
        } else {
            if (value != ii + 1)
                return __LINE__;
        }
    }

    return 0;
}

int
main(int argc, char **argv)
{
    set_flags(argc, argv);
    runTest(testManyThreads);
    test_passed(__FILE__);
    return 0;
}

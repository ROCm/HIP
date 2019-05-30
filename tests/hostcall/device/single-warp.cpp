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

__global__ void
kernelSinglePacketSingleWarpModThree(void *buffer, ulong *retval0,
                                     ulong *retval1)
{
    uint tid = hipThreadIdx_x;
    if (tid % 3 != 0)
        return;

    uint count = hipThreadIdx_x * 8 + 1;
    ulong arg0 = count++;
    ulong arg1 = count++;
    ulong arg2 = count++;
    ulong arg3 = count++;
    ulong arg4 = count++;
    ulong arg5 = count++;
    ulong arg6 = count++;
    ulong arg7 = count++;

    long2 result;
    result.data =
        __ockl_hostcall_internal(buffer, TEST_SERVICE, arg0, arg1, arg2, arg3,
                                 arg4, arg5, arg6, arg7);
    retval0[tid] = result.x;
    retval1[tid] = result.y;
}

static uint
checkSinglePacketSingleWarpModThree(hostcall_buffer_t *buffer)
{
    wait_on_signal(buffer->doorbell, 1024 * 1024, SIGNAL_INIT);
    ulong cptr =
        __atomic_load_n(&buffer->ready_stack, std::memory_order_acquire);
    if (cptr == 0) {
        return __LINE__;
    }
    WHEN_DEBUG(std::cout << "received packet: " << std::hex << cptr << std::dec
                         << std::endl);
    ulong fptr =
        __atomic_load_n(&buffer->free_stack, std::memory_order_relaxed);
    WHEN_DEBUG(std::cout << "free stack: " << std::hex << fptr << std::dec
                         << std::endl);
    if (fptr != 0)
        return __LINE__;

    header_t *header = get_header(buffer, cptr);
    if (header->next != 0)
        return __LINE__;

    if (get_ready_flag(header->control) == 0)
        return __LINE__;

    // If every third bit is set, we get a series of octal 1's
    if (header->activemask != 01111111111111111111111)
        return __LINE__;
    if (header->service != TEST_SERVICE)
        return __LINE__;

    payload_t *payload = get_payload(buffer, cptr);
    for (int tid = 0; tid != 64; ++tid) {
        if (tid % 3 != 0)
            continue;

        WHEN_DEBUG(std::cout << "workitem: " << std::dec << tid << std::endl);
        auto slot = payload->slots[tid];
        for (int ii = 0; ii != 8; ++ii) {
            WHEN_DEBUG(std::cout << "payload: " << std::hex << slot[ii] << std::dec
                                 << std::endl);
            if (slot[ii] != tid * 8 + ii + 1)
                return __LINE__;
        }
        slot[0] = tid % 5 + 1;
        slot[1] = tid % 7 + 1;
    }

    __atomic_store_n(&header->control, reset_ready_flag(header->control),
                     std::memory_order_release);

    // wait for the single wave to return its packet
    ulong F = __atomic_load_n(&buffer->free_stack, std::memory_order_acquire);
    while (F == fptr) {
        std::this_thread::sleep_for(std::chrono::microseconds(5));
        F = __atomic_load_n(&buffer->free_stack, std::memory_order_acquire);
    }
    WHEN_DEBUG(std::cout << "new free stack: " << std::hex << F << std::endl);
    if (F != inc_ptr_tag(cptr, buffer->index_size)) {
        return __LINE__;
    }

    return 0;
}

uint
testSinglePacketSingleWarpModThree()
{
    unsigned int numThreads = 64;
    unsigned int numBlocks = 1;

    unsigned int numPackets = 1;

    hsa_signal_t signal;
    if (hsa_signal_create(SIGNAL_INIT, 0, NULL, &signal) != HSA_STATUS_SUCCESS)
        return __LINE__;

    hostcall_buffer_t *buffer = createBuffer(numPackets, signal);
    if (!buffer)
        return __LINE__;

    void *retval0_void;
    if (hipHostMalloc(&retval0_void, 8 * numThreads) != hipSuccess)
        return __LINE__;
    uint64_t *retval0 = (uint64_t *)retval0_void;
    for (int i = 0; i != numThreads; ++i) {
        retval0[i] = 0x23232323;
    }

    void *retval1_void;
    if (hipHostMalloc(&retval1_void, 8 * numThreads) != hipSuccess)
        return __LINE__;
    uint64_t *retval1 = (uint64_t *)retval1_void;
    for (int i = 0; i != numThreads; ++i) {
        retval1[i] = 0x17171717;
    }

    hipLaunchKernelGGL(kernelSinglePacketSingleWarpModThree, dim3(numBlocks),
                       dim3(numThreads, 1, 1), 0, 0, buffer, retval0, retval1);
    hipEvent_t start;
    HIPCHECK(hipEventCreate(&start));
    HIPCHECK(hipEventRecord(start));

    uint status = checkSinglePacketSingleWarpModThree(buffer);
    WHEN_DEBUG(std::cout << "check status: " << std::dec << status << std::endl);
    if (status != 0)
        return status;

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (hipEventQuery(start) != hipSuccess) {
        return __LINE__;
    }

    HIPCHECK(hipDeviceSynchronize());
    for (int i = 0; i != numThreads; ++i) {
        switch (i % 3) {
        case 0:
            if (retval0[i] != i % 5 + 1)
                return __LINE__;
            if (retval1[i] != i % 7 + 1)
                return __LINE__;
            break;
        default:
            if (retval0[i] != 0x23232323)
                return __LINE__;
            if (retval1[i] != 0x17171717)
                return __LINE__;
            break;
        }
    }

    return 0;
}

int
main(int argc, char **argv)
{
    set_flags(argc, argv);
    runTest(testSinglePacketSingleWarpModThree);
    test_passed(__FILE__);
    return 0;
}

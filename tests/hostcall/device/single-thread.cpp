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
kernelSinglePacketSingleWorkitem(void *buffer, ulong *retval0, ulong *retval1)
{
    uint count = 1;
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

    *retval0 = result.x;
    *retval1 = result.y;
}

static uint
checkSinglePacketSingleWorkitem(hostcall_buffer_t *buffer)
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

    if (header->activemask != 1)
        return __LINE__;
    if (header->service != TEST_SERVICE)
        return __LINE__;

    payload_t *payload = get_payload(buffer, cptr);
    auto p = payload->slots[0];

    for (int ii = 0; ii != 8; ++ii) {
        WHEN_DEBUG(std::cout << "payload: " << p[ii] << std::endl);
        if (p[ii] != ii + 1)
            return __LINE__;
    }
    p[0] = 42;
    p[1] = 17;

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

static uint
testSinglePacketSingleWorkitem()
{
    unsigned int numThreads = 1;
    unsigned int numBlocks = 1;

    unsigned int numPackets = 1;

    hsa_signal_t signal;
    if (hsa_signal_create(SIGNAL_INIT, 0, NULL, &signal) != HSA_STATUS_SUCCESS)
        return __LINE__;

    hostcall_buffer_t *buffer = createBuffer(numPackets, signal);
    if (!buffer)
        return __LINE__;

    void *retval0_void;
    if (hipHostMalloc(&retval0_void, 8) != hipSuccess)
        return __LINE__;
    uint64_t *retval0 = (uint64_t *)retval0_void;
    *retval0 = 0x23232323;

    void *retval1_void;
    if (hipHostMalloc(&retval1_void, 8) != hipSuccess)
        return __LINE__;
    uint64_t *retval1 = (uint64_t *)retval1_void;
    *retval1 = 0x17171717;

    hipLaunchKernelGGL(kernelSinglePacketSingleWorkitem, dim3(numBlocks),
                       dim3(numThreads), 0, 0, buffer, retval0, retval1);
    hipEvent_t start;
    HIPCHECK(hipEventCreate(&start));
    HIPCHECK(hipEventRecord(start));

    uint status = checkSinglePacketSingleWorkitem(buffer);
    WHEN_DEBUG(std::cout << "check status: " << std::dec << status << std::endl);
    if (status != 0)
        return status;

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (hipEventQuery(start) != hipSuccess) {
        return __LINE__;
    }

    HIPCHECK(hipDeviceSynchronize());
    if (*retval0 != 42)
        return __LINE__;

    if (*retval1 != 17)
        return __LINE__;

    return 0;
}

int
main(int argc, char **argv)
{
    set_flags(argc, argv);
    runTest(testSinglePacketSingleWorkitem);
    test_passed(__FILE__);
    return 0;
}

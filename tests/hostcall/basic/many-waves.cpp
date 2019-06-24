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

#include <amd_hostcall.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>

#include "common.h"

static int
handler(void *state, uint32_t service, ulong *payload)
{
    *payload = *payload + 1;
    return 0;
}

__global__ void
kernel(ulong *retval)
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
        result.data = __ockl_hostcall_preview(TEST_SERVICE, arg0, arg1, arg2,
                                              arg3, arg4, arg5, arg6, arg7);
        retval[tid] = result.x;
    }
}

static void
test()
{
    uint num_blocks = 5;
    uint threads_per_block = 1000;
    uint num_threads = num_blocks * threads_per_block;

    void *retval_void;
    HIPCHECK(hipHostMalloc(&retval_void, 8 * num_threads));
    uint64_t *retval = (uint64_t *)retval_void;
    for (uint i = 0; i != num_threads; ++i) {
        retval[i] = 0x23232323;
    }

    amd_hostcall_register_service(TEST_SERVICE, handler, nullptr);

    hipLaunchKernelGGL(kernel, dim3(num_blocks), dim3(threads_per_block), 0, 0,
                       retval);
    hipEvent_t mark;
    HIPCHECK(hipEventCreate(&mark));
    HIPCHECK(hipEventRecord(mark));

    ASSERT(!timeout(mark, 500));

    hipStreamSynchronize(0);

    for (uint ii = 0; ii != num_threads; ++ii) {
        ulong value = retval[ii];
        if (ii % 71 == 1) {
            ASSERT(value == 0x23232323);
        } else {
            ASSERT(value == ii + 1);
        }
    }
}

int
main(int argc, char **argv)
{
    ASSERT(set_flags(argc, argv));
    test();
    test_passed();
}

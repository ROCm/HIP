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

#include "hip/hcc_detail/grid_launch_GGL.hpp"

// Internal header, do not percolate upwards.
#include "hip_hcc_internal.h"
#include "hc.hpp"
#include "trace_helper.h"

namespace hip_impl
{
    hc::accelerator_view lock_stream_hip_(
        hipStream_t& stream, void*& locked_stream)
    {   // This allocated but does not take ownership of locked_stream. If it is
        // not deleted elsewhere it will leak.
        using L = decltype(stream->lockopen_preKernelCommand());

        HIP_INIT();

        stream = ihipSyncAndResolveStream(stream);
        locked_stream = new L{stream->lockopen_preKernelCommand()};
        return (*static_cast<L*>(locked_stream))->_av;
    }

    void unlock_stream_hip_(
        hipStream_t stream,
        void* locked_stream,
        const char* kernel_name,
        hc::accelerator_view* acc_v)
    {   // Precondition: acc_v is the accelerator_view associated with stream
        //               which is guarded by locked_stream;
        //               locked_stream is deletable.
        using L = decltype(stream->lockopen_preKernelCommand());

        stream->lockclose_postKernelCommand(kernel_name, acc_v);

        delete static_cast<L*>(locked_stream);
        locked_stream = nullptr;
    }
}

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

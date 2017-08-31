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

#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Stream
//

//---
hipError_t ihipStreamCreate(hipStream_t *stream, unsigned int flags)
{
    ihipCtx_t *ctx = ihipGetTlsDefaultCtx();

    hipError_t e = hipSuccess;

    if (ctx) {

        if (HIP_FORCE_NULL_STREAM) {
            *stream = 0; 
        } else {
            hc::accelerator acc = ctx->getWriteableDevice()->_acc;

            // TODO - se try-catch loop to detect memory exception?
            //
            //Note this is an execute_in_order queue, so all kernels submitted will atuomatically wait for prev to complete:
            //This matches CUDA stream behavior:

            {
                // Obtain mutex access to the device critical data, release by destructor
                LockedAccessor_CtxCrit_t  ctxCrit(ctx->criticalData());

                auto istream = new ihipStream_t(ctx, acc.create_view(), flags);

                ctxCrit->addStream(istream);
                *stream = istream;
            }
        }

        tprintf(DB_SYNC, "hipStreamCreate, %s\n", ToString(*stream).c_str());
    } else {
        e = hipErrorInvalidDevice;
    }

    return e;
}


//---
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
{
    HIP_INIT_API(stream, flags);

    return ihipLogStatus(ihipStreamCreate(stream, flags));

}

//---
hipError_t hipStreamCreate(hipStream_t *stream)
{
    HIP_INIT_API(stream);

    return ihipLogStatus(ihipStreamCreate(stream, hipStreamDefault));
}


hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    HIP_INIT_SPECIAL_API(TRACE_SYNC, stream, event, flags);

    hipError_t e = hipSuccess;

    if (event == nullptr) {
        e = hipErrorInvalidResourceHandle;

    } else if (event->_state != hipEventStatusUnitialized) {

        if (HIP_SYNC_STREAM_WAIT || (HIP_SYNC_NULL_STREAM && (stream == 0))) {
            // conservative wait on host for the specified event to complete:
            event->locked_waitComplete((event->_flags & hipEventBlockingSync) ? hc::hcWaitModeBlocked : hc::hcWaitModeActive);
        } else {
            stream = ihipSyncAndResolveStream(stream);
            // This will user create_blocking_marker to wait on the specified queue.
            stream->locked_streamWaitEvent(event);
        }

    } // else event not recorded, return immediately and don't create marker.

    return ihipLogStatus(e);
};


//---
hipError_t hipStreamQuery(hipStream_t stream)
{
    HIP_INIT_SPECIAL_API(TRACE_QUERY, stream);

    // Use default stream if 0 specified:
    if (stream == hipStreamNull) {
        ihipCtx_t *device = ihipGetTlsDefaultCtx();
        stream =  device->_defaultStream;
    }

    bool isEmpty = 0;

    {
        LockedAccessor_StreamCrit_t crit(stream->_criticalData);
        isEmpty = crit->_av.get_is_empty();
    }

    hipError_t e = isEmpty ? hipSuccess : hipErrorNotReady ;

    return ihipLogStatus(e);
}


//---
hipError_t hipStreamSynchronize(hipStream_t stream)
{
    HIP_INIT_API(stream);
    HIP_INIT_SPECIAL_API(TRACE_SYNC, stream);

    hipError_t e = hipSuccess;

    if (stream == hipStreamNull) {
        ihipCtx_t *ctx = ihipGetTlsDefaultCtx();
        ctx->locked_syncDefaultStream(true/*waitOnSelf*/, true/*syncToHost*/);
    } else {
		// note this does not synchornize with the NULL stream:
        stream->locked_wait();
        e = hipSuccess;
    }


    return ihipLogStatus(e);
};


//---
/**
 * @return #hipSuccess, #hipErrorInvalidResourceHandle
 */
hipError_t hipStreamDestroy(hipStream_t stream)
{
    HIP_INIT_API(stream);

    hipError_t e = hipSuccess;

    //--- Drain the stream:
    if (stream == NULL) {
        if (!HIP_FORCE_NULL_STREAM) {
            e = hipErrorInvalidResourceHandle; 
        } 
    } else {
        stream->locked_wait();

        ihipCtx_t *ctx = stream->getCtx();

        if (ctx) {
            ctx->locked_removeStream(stream);
            delete stream;
        } else {
            e = hipErrorInvalidResourceHandle;
        }
    }

    return ihipLogStatus(e);
}


//---
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags)
{
    HIP_INIT_API(stream, flags);

    if (flags == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    } else if (stream == hipStreamNull) {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    } else {
        *flags = stream->_flags;
        return ihipLogStatus(hipSuccess);
    }
}


//---
hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void *userData, unsigned int flags)
{
    HIP_INIT_API(stream, callback, userData, flags);
    hipError_t e = hipSuccess;
    //--- explicitly synchronize stream to add callback routines
    hipStreamSynchronize(stream);
    callback(stream, e, userData);
    return ihipLogStatus(e);
}

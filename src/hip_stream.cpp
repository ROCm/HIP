/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip_runtime.h"
#include "hcc_detail/hip_hcc.h"
#include "hcc_detail/trace_helper.h"


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Stream
//

//---
hipError_t ihipStreamCreate(hipStream_t *stream, unsigned int flags)
{
    ihipCtx_t *ctx = ihipGetTlsDefaultCtx();
    hc::accelerator acc = ctx->_acc;

    // TODO - se try-catch loop to detect memory exception?
    //
    //
    //Note this is an execute_in_order queue, so all kernels submitted will atuomatically wait for prev to complete:
    //This matches CUDA stream behavior:

    auto istream = new ihipStream_t(ctx, acc.create_view(), flags);

    ctx->locked_addStream(istream);

    *stream = istream;
    tprintf(DB_SYNC, "hipStreamCreate, stream=%p\n", *stream);

    return hipSuccess;
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


//---
/**
 * @bug This function conservatively waits for all work in the specified stream to complete.
 */
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
{
    HIP_INIT_API(stream, event, flags);

    hipError_t e = hipSuccess;

    {
        // TODO-hcc Convert to use create_blocking_marker(...) functionality.
        // Currently we have a super-conservative version of this - block on host, and drain the queue.
        // This should create a barrier packet in the target queue.
        stream->locked_wait();
        e = hipSuccess;
    }

    return ihipLogStatus(e);
};


//---
hipError_t hipStreamSynchronize(hipStream_t stream)
{
    HIP_INIT_API(stream);

    hipError_t e = hipSuccess;

    if (stream == NULL) {
        ihipCtx_t *ctx = ihipGetTlsDefaultCtx();
        ctx->locked_syncDefaultStream(true/*waitOnSelf*/);
    } else {
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
        ihipCtx_t *ctx = ihipGetTlsDefaultCtx();
        ctx->locked_syncDefaultStream(true/*waitOnSelf*/);
    } else {
        stream->locked_wait();
        e = hipSuccess;
    }

    ihipCtx_t *ctx = stream->getDevice();

    if (ctx) {
        ctx->locked_removeStream(stream);
        delete stream;
    } else {
        e = hipErrorInvalidResourceHandle;
    }

    return ihipLogStatus(e);
}


//---
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags)
{
    HIP_INIT_API(stream, flags);

    if (flags == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    } else if (stream == NULL) {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    } else {
        *flags = stream->_flags;
        return ihipLogStatus(hipSuccess);
    }
}




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
// Events
//---


hipError_t ihipEventCreate(hipEvent_t* event, unsigned flags)
{
    hipError_t e = hipSuccess;

    // TODO - support hipEventDefault, hipEventBlockingSync, hipEventDisableTiming
    if (flags == 0) {
        ihipEvent_t *eh = new ihipEvent_t();

        eh->_state  = hipEventStatusCreated;
        eh->_stream = NULL;
        eh->_flags  = flags;
        eh->_timestamp  = 0;
        eh->_copySeqId  = 0;
        *event = eh;
    } else {
        e = hipErrorInvalidValue;
    }

    return e;
}

/**
 * @warning : flags must be 0.
 */
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags)
{
    HIP_INIT_API(event, flags);

    return ihipLogStatus(ihipEventCreate(event, flags));
}


hipError_t hipEventCreate(hipEvent_t* event)
{
    HIP_INIT_API(event);

    return ihipLogStatus(ihipEventCreate(event, 0));
}


//---
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)
{
    HIP_INIT_API(event, stream);

    ihipEvent_t *eh = event;
    if (eh && eh->_state != hipEventStatusUnitialized)   {
        eh->_stream = stream;

        if (stream == NULL) {
            // If stream == NULL, wait on all queues.
            // TODO-HCC fix this - is this conservative or still uses device timestamps?
            // TODO-HCC can we use barrier or event marker to implement better solution?
            ihipCtx_t *ctx = ihipGetTlsDefaultCtx();
            ctx->locked_syncDefaultStream(true);

            eh->_timestamp = hc::get_system_ticks();
            eh->_state = hipEventStatusRecorded;
            return ihipLogStatus(hipSuccess);
        } else {
            eh->_state  = hipEventStatusRecording;
            // Clear timestamps
            eh->_timestamp = 0;
            eh->_marker = stream->_av.create_marker();
            
            eh->_copySeqId = stream->locked_lastCopySeqId();

            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}


//---
hipError_t hipEventDestroy(hipEvent_t event)
{
    HIP_INIT_API(event);

    event->_state  = hipEventStatusUnitialized;

    delete event;
    event = NULL;

    // TODO - examine return additional error codes
    return ihipLogStatus(hipSuccess);
}


//---
hipError_t hipEventSynchronize(hipEvent_t event)
{
    HIP_INIT_API(event);

    ihipEvent_t *eh = event;

    if (eh) {
        if (eh->_state == hipEventStatusUnitialized) {
            return ihipLogStatus(hipErrorInvalidResourceHandle);
        } else if (eh->_state == hipEventStatusCreated ) {
            // Created but not actually recorded on any device:
            return ihipLogStatus(hipSuccess);
        } else if (eh->_stream == NULL) {
            auto *ctx = ihipGetTlsDefaultCtx();
            ctx->locked_syncDefaultStream(true);
            return ihipLogStatus(hipSuccess);
        } else {
            eh->_marker.wait((eh->_flags & hipEventBlockingSync) ? hc::hcWaitModeBlocked : hc::hcWaitModeActive);
            eh->_stream->locked_reclaimSignals(eh->_copySeqId);

            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}


//---
hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop)
{
    HIP_INIT_API(ms, start, stop);

    ihipEvent_t *start_eh = start;
    ihipEvent_t *stop_eh = stop;

    ihipSetTs(start);
    ihipSetTs(stop);

    hipError_t status = hipSuccess;
    *ms = 0.0f;

    if (start_eh && stop_eh) {
        if ((start_eh->_state == hipEventStatusRecorded) && (stop_eh->_state == hipEventStatusRecorded)) {
            // Common case, we have good information for both events.

            int64_t tickDiff = (stop_eh->_timestamp - start_eh->_timestamp);

            // TODO-move this to a variable saved with each agent.
            uint64_t freqHz;
            hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freqHz);
            if (freqHz) {
                *ms = ((double)(tickDiff) /  (double)(freqHz)) * 1000.0f;
                status = hipSuccess;
            } else {
                * ms = 0.0f;
                status = hipErrorInvalidValue;
            }


        } else if ((start_eh->_state == hipEventStatusRecording) ||
                   (stop_eh->_state  == hipEventStatusRecording)) {
            status = hipErrorNotReady;
        } else if ((start_eh->_state == hipEventStatusUnitialized) ||
                   (stop_eh->_state  == hipEventStatusUnitialized)) {
            status = hipErrorInvalidResourceHandle;
        }
    }

    return ihipLogStatus(status);
}


//---
hipError_t hipEventQuery(hipEvent_t event)
{
    HIP_INIT_API(event);

    ihipEvent_t *eh = event;

    // TODO-stream - need to read state of signal here:  The event may have become ready after recording..
    // TODO-HCC - use get_hsa_signal here.

    if (eh->_state == hipEventStatusRecording) {
        return ihipLogStatus(hipErrorNotReady);
    } else {
        return ihipLogStatus(hipSuccess);
    }
}



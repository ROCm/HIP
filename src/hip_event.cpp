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
// Events
//---


ihipEvent_t::ihipEvent_t(unsigned flags)
{
    _state  = hipEventStatusCreated;
    _stream = NULL;
    _flags  = flags;
    _timestamp  = 0;
    _type   = hipEventTypeIndependent;
};



// Attach to an existing completion future:
void ihipEvent_t::attachToCompletionFuture(const hc::completion_future *cf, ihipEventType_t eventType)
{
    _state  = hipEventStatusRecording;
    _marker = *cf;
    _type   = eventType;
}



void ihipEvent_t::setTimestamp()
{
    if (_state == hipEventStatusRecorded) {
        // already recorded, done:
        return;
    } else {
        // TODO - use completion-future functions to obtain ticks and timestamps:
        hsa_signal_t *sig  = static_cast<hsa_signal_t*> (_marker.get_native_handle());
        if (sig) {
            if (hsa_signal_load_acquire(*sig) == 0) {

                if ((_type == hipEventTypeIndependent) || (_type == hipEventTypeStopCommand)) {
                    _timestamp =  _marker.get_end_tick();
                } else if (_type == hipEventTypeStartCommand) {
                    _timestamp =  _marker.get_begin_tick();
                } else {
                    assert(0); // TODO - move to debug assert
                    _timestamp =  0;
                }

                _state = hipEventStatusRecorded;
            }
        }
    }
}


hipError_t ihipEventCreate(hipEvent_t* event, unsigned flags)
{
    hipError_t e = hipSuccess;

    // TODO-IPC - support hipEventInterprocess.
    unsigned supportedFlags = hipEventDefault | hipEventBlockingSync | hipEventDisableTiming;
    if ((flags & ~supportedFlags) == 0) {
        ihipEvent_t *eh = new ihipEvent_t(flags);

        *event = eh;
    } else {
        e = hipErrorInvalidValue;
    }

    return e;
}

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

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)
{
    HIP_INIT_API(event, stream);

    if (event && event->_state != hipEventStatusUnitialized)   {
        event->_stream = stream;

        if (stream == NULL) {
            // If stream == NULL, wait on all queues.
            // TODO-HCC fix this - is this conservative or still uses device timestamps?
            // TODO-HCC can we use barrier or event marker to implement better solution?
            ihipCtx_t *ctx = ihipGetTlsDefaultCtx();
            ctx->locked_syncDefaultStream(true);

            event->_timestamp = hc::get_system_ticks();
            event->_state = hipEventStatusRecorded;
            return ihipLogStatus(hipSuccess);
        } else {
            event->_state  = hipEventStatusRecording;
            // Clear timestamps
            event->_timestamp = 0;

            // Record the event in the stream:
            stream->locked_recordEvent(event);

            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}

hipError_t hipEventDestroy(hipEvent_t event)
{
    HIP_INIT_API(event);

    event->_state  = hipEventStatusUnitialized;

    delete event;
    event = NULL;

    // TODO - examine return additional error codes
    return ihipLogStatus(hipSuccess);
}

hipError_t hipEventSynchronize(hipEvent_t event)
{
    HIP_INIT_API(event);

    if (event) {
        if (event->_state == hipEventStatusUnitialized) {
            return ihipLogStatus(hipErrorInvalidResourceHandle);
        } else if (event->_state == hipEventStatusCreated ) {
            // Created but not actually recorded on any device:
            return ihipLogStatus(hipSuccess);
        } else if (event->_stream == NULL) {
            auto *ctx = ihipGetTlsDefaultCtx();
            ctx->locked_syncDefaultStream(true);
            return ihipLogStatus(hipSuccess);
        } else {
            event->_marker.wait((event->_flags & hipEventBlockingSync) ? hc::hcWaitModeBlocked : hc::hcWaitModeActive);

            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}

hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop)
{
    HIP_INIT_API(ms, start, stop);

    ihipEvent_t *start_eh = start;
    ihipEvent_t *stop_eh = stop;

    start->setTimestamp();
    stop->setTimestamp();

    hipError_t status = hipSuccess;
    *ms = 0.0f;

    if (start_eh && stop_eh) {
        if ((start_eh->_state == hipEventStatusRecorded) && (stop_eh->_state == hipEventStatusRecorded)) {
            // Common case, we have good information for both events.

            int64_t tickDiff = (stop_eh->timestamp() - start_eh->timestamp());

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

hipError_t hipEventQuery(hipEvent_t event)
{
    HIP_INIT_API(event);

    if ((event->_state == hipEventStatusRecording) && (!event->_marker.is_ready())) {
        return ihipLogStatus(hipErrorNotReady);
    } else {
        return ihipLogStatus(hipSuccess);
    }
}

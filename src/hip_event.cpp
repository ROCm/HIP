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


ihipEvent_t::ihipEvent_t(unsigned flags) : _criticalData(this) { _flags = flags; };


// Attach to an existing completion future:
void ihipEvent_t::attachToCompletionFuture(const hc::completion_future* cf, hipStream_t stream,
                                           ihipEventType_t eventType) {
    LockedAccessor_EventCrit_t crit(_criticalData);
    crit->_eventData.marker(*cf);
    crit->_eventData._type = eventType;
    crit->_eventData._stream = stream;
    crit->_eventData._state = hipEventStatusRecording;
}


static std::pair<hipEventStatus_t, uint64_t> refreshEventStatus(ihipEventData_t &ecd) {
    if (ecd._state == hipEventStatusRecording && ecd.marker().is_ready()) {
        if ((ecd._type == hipEventTypeIndependent) ||
            (ecd._type == hipEventTypeStopCommand)) {
            ecd._timestamp = ecd.marker().get_end_tick();
        } else if (ecd._type == hipEventTypeStartCommand) {
            ecd._timestamp = ecd.marker().get_begin_tick();
        } else {
            ecd._timestamp = 0;
            assert(0);  // TODO - move to debug assert
        }

        ecd._state = hipEventStatusComplete;

        return std::pair<hipEventStatus_t, uint64_t>(ecd._state,
                                                     ecd._timestamp);
    }

    // Not complete path here:
    return std::pair<hipEventStatus_t, uint64_t>(ecd._state, ecd._timestamp);
}


hipError_t ihipEventCreate(hipEvent_t* event, unsigned flags) {
    hipError_t e = hipSuccess;

    // TODO-IPC - support hipEventInterprocess.
    unsigned supportedFlags = hipEventDefault | hipEventBlockingSync | hipEventDisableTiming |
                              hipEventReleaseToDevice | hipEventReleaseToSystem;
    const unsigned releaseFlags = (hipEventReleaseToDevice | hipEventReleaseToSystem);

    const bool illegalFlags =
        (flags & ~supportedFlags) ||             // can't set any unsupported flags.
        (flags & releaseFlags) == releaseFlags;  // can't set both release flags

    if (event && !illegalFlags) {
        *event = new ihipEvent_t(flags);
    } else {
        e = hipErrorInvalidValue;
    }

    return e;
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
    HIP_INIT_API(hipEventCreateWithFlags, event, flags);

    return ihipLogStatus(ihipEventCreate(event, flags));
}

hipError_t hipEventCreate(hipEvent_t* event) {
    HIP_INIT_API(hipEventCreate, event);

    return ihipLogStatus(ihipEventCreate(event, 0));
}


hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    HIP_INIT_SPECIAL_API(hipEventRecord, TRACE_SYNC, event, stream);
    if (!event) return ihipLogStatus(hipErrorInvalidResourceHandle);
    stream = ihipSyncAndResolveStream(stream);
    LockedAccessor_EventCrit_t eCrit(event->criticalData());
    if (eCrit->_eventData._state == hipEventStatusUnitialized) return ihipLogStatus(hipErrorInvalidResourceHandle);
    if (HIP_SYNC_NULL_STREAM && stream->isDefaultStream()) {
        // TODO-HIP_SYNC_NULL_STREAM : can remove this code when HIP_SYNC_NULL_STREAM = 0
        // If default stream , then wait on all queues.
        ihipCtx_t* ctx = ihipGetTlsDefaultCtx();
        ctx->locked_syncDefaultStream(true, true);
        eCrit->_eventData.marker(hc::completion_future());  // reset event
        eCrit->_eventData._stream = stream;
        eCrit->_eventData._timestamp = hc::get_system_ticks();
        eCrit->_eventData._state = hipEventStatusComplete;
    }
    else {
        // Record the event in the stream:
        eCrit->_eventData.marker(stream->locked_recordEvent(event));
        eCrit->_eventData._stream = stream;
        eCrit->_eventData._timestamp = 0;
        eCrit->_eventData._state = hipEventStatusRecording;
    }
    return ihipLogStatus(hipSuccess);
}


hipError_t hipEventDestroy(hipEvent_t event) {
    HIP_INIT_API(hipEventDestroy, event);

    if (event) {
        delete event;

        return ihipLogStatus(hipSuccess);
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}

hipError_t hipEventSynchronize(hipEvent_t event) {
    HIP_INIT_SPECIAL_API(hipEventSynchronize, TRACE_SYNC, event);

    if (!(event->_flags & hipEventReleaseToSystem)) {
        tprintf(DB_WARN,
                "hipEventSynchronize on event without system-scope fence ; consider creating with "
                "hipEventReleaseToSystem\n");
    }
    auto ecd = event->locked_copyCrit();

    if (event) {
        if (ecd._state == hipEventStatusUnitialized) {
            return ihipLogStatus(hipErrorInvalidResourceHandle);
        } else if (ecd._state == hipEventStatusCreated) {
            // Created but not actually recorded on any device:
            return ihipLogStatus(hipSuccess);
        } else if (HIP_SYNC_NULL_STREAM && (ecd._stream->isDefaultStream())) {
            auto* ctx = ihipGetTlsDefaultCtx();
            // TODO-HIP_SYNC_NULL_STREAM - can remove this code
            ctx->locked_syncDefaultStream(true, true);
            return ihipLogStatus(hipSuccess);
        } else {
            ecd.marker().wait((event->_flags & hipEventBlockingSync) ? hc::hcWaitModeBlocked
                                                                     : hc::hcWaitModeActive);
            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    HIP_INIT_API(hipEventElapsedTime, ms, start, stop);

    if (ms == nullptr) return ihipLogStatus(hipErrorInvalidValue);
    if ((start == nullptr) || (stop == nullptr)) return ihipLogStatus(hipErrorInvalidResourceHandle);

    *ms = 0.0f;
    auto startEcd = start->locked_copyCrit();
    auto stopEcd = stop->locked_copyCrit();

    if ((start->_flags & hipEventDisableTiming) ||
        (startEcd._state == hipEventStatusUnitialized) ||
        (startEcd._state == hipEventStatusCreated) ||
        (stop->_flags & hipEventDisableTiming) ||
        (stopEcd._state == hipEventStatusUnitialized) ||
        (stopEcd._state == hipEventStatusCreated)) {
        // Both events must be at least recorded else return hipErrorInvalidResourceHandle
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }

    // Refresh status, if still recording...

    auto startStatus = refreshEventStatus(startEcd);  // pair < state, timestamp >
    auto stopStatus = refreshEventStatus(stopEcd);    // pair < state, timestamp >

    if ((startStatus.first == hipEventStatusComplete) &&
        (stopStatus.first == hipEventStatusComplete)) {
        // Common case, we have good information for both events.  'second' is the timestamp:
        int64_t tickDiff = (stopStatus.second - startStatus.second);
        uint64_t freqHz;
        hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freqHz);
        if (freqHz) {
            *ms = ((double)(tickDiff) / (double)(freqHz)) * 1000.0f;
            return ihipLogStatus(hipSuccess);
        } else {
            *ms = 0.0f;
            return ihipLogStatus(hipErrorInvalidValue);
        }
    } else if ((startStatus.first == hipEventStatusRecording) ||
               (stopStatus.first == hipEventStatusRecording)) {
        return ihipLogStatus(hipErrorNotReady);
    } else {
        assert(0); // TODO should we return hipErrorUnknown ?
    }

    return ihipLogStatus(hipSuccess);
}

hipError_t hipEventQuery(hipEvent_t event) {
    HIP_INIT_SPECIAL_API(hipEventQuery, TRACE_QUERY, event);
 
    hipError_t status = hipSuccess;

    if ( NULL == event)
    {
        status = hipErrorInvalidResourceHandle;
    } else {
        if (!(event->_flags & hipEventReleaseToSystem)) {
            tprintf(DB_WARN,
                    "hipEventQuery on event without system-scope fence ; consider creating with "
                    "hipEventReleaseToSystem\n");
        }

        auto ecd = event->locked_copyCrit();

        if ((ecd._state == hipEventStatusRecording) && !ecd._stream->locked_eventIsReady(event)) {
            status = hipErrorNotReady;
        } else {
            status = hipSuccess;
        }
    }
    return ihipLogStatus(status);
}

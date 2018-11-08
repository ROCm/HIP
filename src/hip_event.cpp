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


std::pair<hipEventStatus_t, uint64_t> ihipEvent_t::refreshEventStatus() {
    auto ecd = locked_copyCrit();
    if (ecd._state == hipEventStatusRecording) {
        bool isReady1 = ecd._stream->locked_eventIsReady(this);
        if (isReady1) {
            LockedAccessor_EventCrit_t eCrit(_criticalData);

            if ((eCrit->_eventData._type == hipEventTypeIndependent) ||
                (eCrit->_eventData._type == hipEventTypeStopCommand)) {
                eCrit->_eventData._timestamp = eCrit->_eventData.marker().get_end_tick();
            } else if (eCrit->_eventData._type == hipEventTypeStartCommand) {
                eCrit->_eventData._timestamp = eCrit->_eventData.marker().get_begin_tick();
            } else {
                eCrit->_eventData._timestamp = 0;
                assert(0);  // TODO - move to debug assert
            }

            eCrit->_eventData._state = hipEventStatusComplete;

            return std::pair<hipEventStatus_t, uint64_t>(eCrit->_eventData._state,
                                                         eCrit->_eventData._timestamp);
        }
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

    if (!illegalFlags) {
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

    auto ecd = event->locked_copyCrit();

    if (event && ecd._state != hipEventStatusUnitialized) {
        stream = ihipSyncAndResolveStream(stream);

        if (HIP_SYNC_NULL_STREAM && stream->isDefaultStream()) {
            // TODO-HIP_SYNC_NULL_STREAM : can remove this code when HIP_SYNC_NULL_STREAM = 0
            //
            // If default stream , then wait on all queues.
            ihipCtx_t* ctx = ihipGetTlsDefaultCtx();
            ctx->locked_syncDefaultStream(true, true);

            {
                LockedAccessor_EventCrit_t eCrit(event->criticalData());
                eCrit->_eventData.marker(hc::completion_future());  // reset event
                eCrit->_eventData._stream = stream;
                eCrit->_eventData._timestamp = hc::get_system_ticks();
                eCrit->_eventData._state = hipEventStatusComplete;
            }
            return ihipLogStatus(hipSuccess);
        } else {
            // Record the event in the stream:
            // Keep a copy outside the critical section so we lock stream first, then event - to
            // avoid deadlock
            hc::completion_future cf = stream->locked_recordEvent(event);

            {
                LockedAccessor_EventCrit_t eCrit(event->criticalData());
                eCrit->_eventData.marker(cf);
                eCrit->_eventData._stream = stream;
                eCrit->_eventData._timestamp = 0;
                eCrit->_eventData._state = hipEventStatusRecording;
            }

            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
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
            ecd._stream->locked_eventWaitComplete(
                ecd.marker(), (event->_flags & hipEventBlockingSync) ? hc::hcWaitModeBlocked
                                                                     : hc::hcWaitModeActive);

            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
    HIP_INIT_API(hipEventElapsedTime, ms, start, stop);

    hipError_t status = hipSuccess;

    *ms = 0.0f;

    if ((start == nullptr) || (stop == nullptr)) {
        status = hipErrorInvalidResourceHandle;
    } else {
        auto startEcd = start->locked_copyCrit();
        auto stopEcd = stop->locked_copyCrit();

        if ((start->_flags & hipEventDisableTiming) ||
            (startEcd._state == hipEventStatusUnitialized) ||
            (startEcd._state == hipEventStatusCreated) || (stop->_flags & hipEventDisableTiming) ||
            (stopEcd._state == hipEventStatusUnitialized) ||
            (stopEcd._state == hipEventStatusCreated)) {
            // Both events must be at least recorded else return hipErrorInvalidResourceHandle

            status = hipErrorInvalidResourceHandle;

        } else {
            // Refresh status, if still recording...

            auto startStatus = start->refreshEventStatus();  // pair < state, timestamp >
            auto stopStatus = stop->refreshEventStatus();    // pair < state, timestamp >

            if ((startStatus.first == hipEventStatusComplete) &&
                (stopStatus.first == hipEventStatusComplete)) {
                // Common case, we have good information for both events.  'second" is the
                // timestamp:
                int64_t tickDiff = (stopStatus.second - startStatus.second);

                uint64_t freqHz;
                hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freqHz);
                if (freqHz) {
                    *ms = ((double)(tickDiff) / (double)(freqHz)) * 1000.0f;
                    status = hipSuccess;
                } else {
                    *ms = 0.0f;
                    status = hipErrorInvalidValue;
                }


            } else if ((startStatus.first == hipEventStatusRecording) ||
                       (stopStatus.first == hipEventStatusRecording)) {
                status = hipErrorNotReady;
            } else {
                assert(0);
            }
        }
    }

    return ihipLogStatus(status);
}

hipError_t hipEventQuery(hipEvent_t event) {
    HIP_INIT_SPECIAL_API(hipEventQuery, TRACE_QUERY, event);

    if (!(event->_flags & hipEventReleaseToSystem)) {
        tprintf(DB_WARN,
                "hipEventQuery on event without system-scope fence ; consider creating with "
                "hipEventReleaseToSystem\n");
    }

    auto ecd = event->locked_copyCrit();

    if ((ecd._state == hipEventStatusRecording) && !ecd._stream->locked_eventIsReady(event)) {
        return ihipLogStatus(hipErrorNotReady);
    } else {
        return ihipLogStatus(hipSuccess);
    }
}

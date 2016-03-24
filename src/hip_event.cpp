#include "hip_runtime.h"
#include "hcc_detail/hip_hcc.h"
#include "hcc_detail/trace_helper.h"

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Events
//---
/**
 * @warning : flags must be 0.
 */
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags)
{
    // TODO - support hipEventDefault, hipEventBlockingSync, hipEventDisableTiming
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    if (flags == 0) {
        ihipEvent_t *eh = event->_handle = new ihipEvent_t();

        eh->_state  = hipEventStatusCreated;
        eh->_stream = NULL;
        eh->_flags  = flags;
        eh->_timestamp  = 0;
        eh->_copy_seq_id  = 0;
    } else {
        e = hipErrorInvalidValue;
    }


    return ihipLogStatus(e);
}


//---
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)
{
    std::call_once(hip_initialized, ihipInit);

    ihipEvent_t *eh = event._handle;
    if (eh && eh->_state != hipEventStatusUnitialized)   {
        eh->_stream = stream;

        if (stream == NULL) {
            // If stream == NULL, wait on all queues.
            // Behavior matches "use standard default semantics".
            // TODO-HCC fix this - conservative or use device timestamps?
            // TODO-HCC can we use barrier or event marker to implement better solution?
            ihipDevice_t *device = ihipGetTlsDefaultDevice();
            device->syncDefaultStream(true);

            eh->_timestamp = hc::get_system_ticks();
            eh->_state = hipEventStatusRecorded;
            return ihipLogStatus(hipSuccess);
        } else {
            eh->_state  = hipEventStatusRecording;
            // Clear timestamps
            eh->_timestamp = 0;
            eh->_marker = stream->_av.create_marker();
            eh->_copy_seq_id = stream->lastCopySeqId();

            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}


//---
hipError_t hipEventDestroy(hipEvent_t event)
{
    std::call_once(hip_initialized, ihipInit);

    event._handle->_state  = hipEventStatusUnitialized;

    delete event._handle;
    event._handle = NULL;

    // TODO - examine return additional error codes
    return ihipLogStatus(hipSuccess);
}


//---
hipError_t hipEventSynchronize(hipEvent_t event)
{
    std::call_once(hip_initialized, ihipInit);

    ihipEvent_t *eh = event._handle;

    if (eh) {
        if (eh->_state == hipEventStatusUnitialized) {
            return ihipLogStatus(hipErrorInvalidResourceHandle);
        } else if (eh->_state == hipEventStatusCreated ) {
            // Created but not actually recorded on any device:
            return ihipLogStatus(hipSuccess);
        } else if (eh->_stream == NULL) {
            ihipDevice_t *device = ihipGetTlsDefaultDevice();
            device->syncDefaultStream(true);
            return ihipLogStatus(hipSuccess);
        } else {
#if __hcc_workweek__ >= 16033
            eh->_marker.wait((eh->_flags & hipEventBlockingSync) ? hc::hcWaitModeBlocked : hc::hcWaitModeActive);
#else
            eh->_marker.wait();
#endif
            eh->_stream->reclaimSignals_ts(eh->_copy_seq_id);

            return ihipLogStatus(hipSuccess);
        }
    } else {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    }
}


//---
hipError_t hipEventElapsedTime(float *ms, hipEvent_t start, hipEvent_t stop)
{
    std::call_once(hip_initialized, ihipInit);

    ihipEvent_t *start_eh = start._handle;
    ihipEvent_t *stop_eh = stop._handle;

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
    std::call_once(hip_initialized, ihipInit);

    ihipEvent_t *eh = event._handle;

    // TODO-stream - need to read state of signal here:  The event may have become ready after recording..
    // TODO-HCC - use get_hsa_signal here.

    if (eh->_state == hipEventStatusRecording) {
        return ihipLogStatus(hipErrorNotReady);
    } else {
        return ihipLogStatus(hipSuccess);
    }
}



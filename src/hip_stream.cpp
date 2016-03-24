
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// Stream
//

//---
hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
{
    std::call_once(hip_initialized, ihipInit);

    ihipDevice_t *device = ihipGetTlsDefaultDevice();
    hc::accelerator acc = device->_acc;

    // TODO - se try-catch loop to detect memory exception?
    //
    //
    //Note this is an execute_in_order queue, so all kernels submitted will atuomatically wait for prev to complete:
    //This matches CUDA stream behavior:

    auto istream = new ihipStream_t(device->_device_index, acc.create_view(), device->_stream_id, flags);
    device->_streams.push_back(istream);
    *stream = istream;
    tprintf(DB_SYNC, "hipStreamCreate, stream=%p\n", *stream);

    return ihipLogStatus(hipSuccess);
}


//---
/**
 * @bug This function conservatively waits for all work in the specified stream to complete.
 */
hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags)
{

    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    {
        // TODO-hcc Convert to use create_blocking_marker(...) functionality.
        // Currently we have a super-conservative version of this - block on host, and drain the queue.
        // This should create a barrier packet in the target queue.
        stream->wait();
        e = hipSuccess;
    }

    return ihipLogStatus(e);
};


//---
hipError_t hipStreamSynchronize(hipStream_t stream)
{
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    if (stream == NULL) {
        ihipDevice_t *device = ihipGetTlsDefaultDevice();
        device->syncDefaultStream(true/*waitOnSelf*/);
    } else {
        stream->wait();
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
    std::call_once(hip_initialized, ihipInit);

    hipError_t e = hipSuccess;

    //--- Drain the stream:
    if (stream == NULL) {
        ihipDevice_t *device = ihipGetTlsDefaultDevice();
        device->syncDefaultStream(true/*waitOnSelf*/);
    } else {
        stream->wait();
        e = hipSuccess;
    }

    ihipDevice_t *device = stream->getDevice();

    if (device) {
        device->_streams.remove(stream);
        delete stream;
    } else {
        e = hipErrorInvalidResourceHandle;
    }

    return ihipLogStatus(e);
}


//---
hipError_t hipStreamGetFlags(hipStream_t stream, unsigned int *flags)
{
    std::call_once(hip_initialized, ihipInit);

    if (flags == NULL) {
        return ihipLogStatus(hipErrorInvalidValue);
    } else if (stream == NULL) {
        return ihipLogStatus(hipErrorInvalidResourceHandle);
    } else {
        *flags = stream->_flags;
        return ihipLogStatus(hipSuccess);
    }
}

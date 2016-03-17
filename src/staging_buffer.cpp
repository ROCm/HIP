#include "hcc_detail/staging_buffer.h"


//-------------------------------------------------------------------------------------------------
StagingBuffer::StagingBuffer(ihipDevice_t *device, size_t bufferSize, int numBuffers) :
    _device(device),
    _bufferSize(bufferSize),
    _numBuffers(numBuffers > _max_buffers ? _max_buffers : numBuffers)
{



    for (int i=0; i<_numBuffers; i++) {
        // TODO - experiment with alignment here.
        _pinnedStagingBuffer[i] = hc::am_alloc(_bufferSize, device->_acc, amHostPinned);
        if (_pinnedStagingBuffer[i] == NULL) {
            throw ihipException(hipErrorMemoryAllocation);
        }
        hsa_signal_create(0, 0, NULL, &_completion_signal[i]);
    }
};

//---
StagingBuffer::~StagingBuffer()
{
    for (int i=0; i<_numBuffers; i++) {
        if (_pinnedStagingBuffer[i]) {
            hc::am_free(_pinnedStagingBuffer[i]);
            _pinnedStagingBuffer[i] = NULL;
        }
        hsa_signal_destroy(_completion_signal[i]);
    }
}



//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from host CPU.
//IN: src - src pointer for copy.  Must be accessible from agent this buffer is associated with (via _device)
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyHostToDevicePinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    const char *srcp = static_cast<const char*> (src);
    char *dstp = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
    }

    if (sizeBytes >= UINT64_MAX/2) {
        throw (ihipException(hipErrorInvalidValue));
    }
    int bufferIndex = 0;
    for (int64_t bytesRemaining=sizeBytes; bytesRemaining>0 ;  bytesRemaining -= _bufferSize) {

        size_t theseBytes = (bytesRemaining > _bufferSize) ? _bufferSize : bytesRemaining;

        tprintf (DB_COPY2, "H2D: waiting... on completion signal handle=%lu\n", _completion_signal[bufferIndex].handle);
        hsa_signal_wait_acquire(_completion_signal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);



        void * masked_srcp = (void*) ((uintptr_t)srcp & (uintptr_t)(~0x3f)) ; // TODO
        void *locked_srcp;
        hsa_status_t hsa_status = hsa_amd_memory_lock(masked_srcp, theseBytes, &_device->_hsa_agent, 1, &locked_srcp);
        //hsa_status_t hsa_status = hsa_amd_memory_lock(const_cast<char*> (srcp), theseBytes, &_device->_hsa_agent, 1, &locked_srcp);
        tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: pin-in-place:%p+%zu bufferIndex[%d]\n", bytesRemaining, srcp, theseBytes, bufferIndex);
        printf ("status=%x srcp=%p, masked_srcp=%p, locked_srcp=%p\n", hsa_status, srcp, masked_srcp, locked_srcp);

        if (hsa_status != HSA_STATUS_SUCCESS) {
            throw (ihipException(hipErrorUnknown));
        }

        hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);

        hsa_status = hsa_amd_memory_async_copy(dstp, _device->_hsa_agent, locked_srcp, _device->_hsa_agent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
        tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: async_copy %zu bytes %p to %p status=%x\n", bytesRemaining, theseBytes, _pinnedStagingBuffer[bufferIndex], dstp, hsa_status);

        if (hsa_status != HSA_STATUS_SUCCESS) {
            throw (ihipException(hipErrorUnknown));
        }

        srcp += theseBytes;
        dstp += theseBytes;
        if (++bufferIndex >= _numBuffers) {
            bufferIndex = 0;
        }

        if (HIP_ONESHOT_COPY_DEP) {
            waitFor = NULL; // TODO - don't need dependency after first copy submitted?
        }
    }

    // TODO -
    printf ("unpin the memory\n");


    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_wait_acquire(_completion_signal[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    }
}




//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from host CPU.
//IN: src - src pointer for copy.  Must be accessible from agent this buffer is associated with (via _device)
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyHostToDevice(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    const char *srcp = static_cast<const char*> (src);
    char *dstp = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
    }

    assert(sizeBytes < UINT64_MAX/2); // TODO
    int bufferIndex = 0;
    for (int64_t bytesRemaining=sizeBytes; bytesRemaining>0 ;  bytesRemaining -= _bufferSize) {

        size_t theseBytes = (bytesRemaining > _bufferSize) ? _bufferSize : bytesRemaining;

        tprintf (DB_COPY2, "H2D: waiting... on completion signal handle=%lu\n", _completion_signal[bufferIndex].handle);
        hsa_signal_wait_acquire(_completion_signal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

        tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: copy %zu bytes %p to stagingBuf[%d]:%p\n", bytesRemaining, theseBytes, srcp, bufferIndex, _pinnedStagingBuffer[bufferIndex]);
        // TODO - use uncached memcpy, someday.
        memcpy(_pinnedStagingBuffer[bufferIndex], srcp, theseBytes);


        hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);

        hsa_status_t hsa_status = hsa_amd_memory_async_copy(dstp, _device->_hsa_agent, _pinnedStagingBuffer[bufferIndex], _device->_hsa_agent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
        tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: async_copy %zu bytes %p to %p status=%x\n", bytesRemaining, theseBytes, _pinnedStagingBuffer[bufferIndex], dstp, hsa_status);

        if (hsa_status != HSA_STATUS_SUCCESS) {
            throw (ihipException(hipErrorUnknown));
        }

        srcp += theseBytes;
        dstp += theseBytes;
        if (++bufferIndex >= _numBuffers) {
            bufferIndex = 0;
        }

        if (HIP_ONESHOT_COPY_DEP) {
            waitFor = NULL; // TODO - don't need dependency after first copy submitted?
        }
    }


    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_wait_acquire(_completion_signal[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    }
}

//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from agent this buffer is assocaited with (via _device).
//IN: src - src pointer for copy.  Must be accessible from host CPU.
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyDeviceToHost(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    const char *srcp0 = static_cast<const char*> (src);
    char *dstp1 = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
    }

    assert(sizeBytes < UINT64_MAX/2); // TODO

    int64_t bytesRemaining0 = sizeBytes; // bytes to copy from dest into staging buffer.
    int64_t bytesRemaining1 = sizeBytes; // bytes to copy from staging buffer into final dest

    while (bytesRemaining1 > 0) {
        // First launch the async copies to copy from dest to host
        for (int bufferIndex = 0; (bytesRemaining0>0) && (bufferIndex < _numBuffers);  bytesRemaining0 -= _bufferSize, bufferIndex++) {

            size_t theseBytes = (bytesRemaining0 > _bufferSize) ? _bufferSize : bytesRemaining0;

            tprintf (DB_COPY2, "D2H: bytesRemaining0=%zu  async_copy %zu bytes src:%p to staging:%p\n", bytesRemaining0, theseBytes, srcp0, _pinnedStagingBuffer[bufferIndex]);
            hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(_pinnedStagingBuffer[bufferIndex], _device->_hsa_agent, srcp0, _device->_hsa_agent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
            if (hsa_status != HSA_STATUS_SUCCESS) {
                throw (ihipException(hipErrorUnknown));
            }

            srcp0 += theseBytes;


            if (HIP_ONESHOT_COPY_DEP) {
                waitFor = NULL; // TODO - don't need dependency after first copy submitted?
            }
        }

        // Now unload the staging buffers:
        for (int bufferIndex=0; (bytesRemaining1>0) && (bufferIndex < _numBuffers);  bytesRemaining1 -= _bufferSize, bufferIndex++) {

            size_t theseBytes = (bytesRemaining1 > _bufferSize) ? _bufferSize : bytesRemaining1;

            tprintf (DB_COPY2, "D2H: wait_completion[%d] bytesRemaining=%zu\n", bufferIndex, bytesRemaining1);
            hsa_signal_wait_acquire(_completion_signal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

            tprintf (DB_COPY2, "D2H: bytesRemaining1=%zu copy %zu bytes stagingBuf[%d]:%p to dst:%p\n", bytesRemaining1, theseBytes, bufferIndex, _pinnedStagingBuffer[bufferIndex], dstp1);
            memcpy(dstp1, _pinnedStagingBuffer[bufferIndex], theseBytes);

            dstp1 += theseBytes;
        }
    }


    //for (int i=0; i<_numBuffers; i++) {
    //    hsa_signal_wait_acquire(_completion_signal[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    //}
}

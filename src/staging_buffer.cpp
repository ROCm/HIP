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

#include <hc_am.hpp>

#include "hsa_ext_amd.h"

#include "hcc_detail/staging_buffer.h"

#ifdef HIP_HCC
#include "hcc_detail/hip_runtime.h"
#include "hcc_detail/hip_hcc.h"
#define THROW_ERROR(e) throw ihipException(e)
#else
#define THROW_ERROR(e) throw 
#define tprintf(trace_level, ...) 
#endif

//-------------------------------------------------------------------------------------------------
StagingBuffer::StagingBuffer(hsa_agent_t hsaAgent, hsa_region_t systemRegion, size_t bufferSize, int numBuffers) :
    _hsa_agent(hsaAgent),
    _bufferSize(bufferSize),
    _numBuffers(numBuffers > _max_buffers ? _max_buffers : numBuffers)
{
    for (int i=0; i<_numBuffers; i++) {
        // TODO - experiment with alignment here.
        hsa_status_t s1 = hsa_memory_allocate(systemRegion, _bufferSize, (void**) (&_pinnedStagingBuffer[i]) );

        if ((s1 != HSA_STATUS_SUCCESS) || (_pinnedStagingBuffer[i] == NULL)) {
            THROW_ERROR(hipErrorMemoryAllocation);
        }
        hsa_signal_create(0, 0, NULL, &_completion_signal[i]);
        hsa_signal_create(0, 0, NULL, &_completion_signal2[i]);
    }
};


//---
StagingBuffer::~StagingBuffer()
{
    for (int i=0; i<_numBuffers; i++) {
        if (_pinnedStagingBuffer[i]) {
            hsa_memory_free(_pinnedStagingBuffer[i]);
            _pinnedStagingBuffer[i] = NULL;
        }
        hsa_signal_destroy(_completion_signal[i]);
        hsa_signal_destroy(_completion_signal2[i]);
    }
}



//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from host CPU.
//IN: src - src pointer for copy.  Must be accessible from agent this buffer is associated with (via _hsa_agent)
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyHostToDevicePinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    std::lock_guard<std::mutex> l (_copy_lock);

    const char *srcp = static_cast<const char*> (src);
    char *dstp = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
    }

    if (sizeBytes >= UINT64_MAX/2) {
        THROW_ERROR (hipErrorInvalidValue);
    }
    int bufferIndex = 0;
    for (int64_t bytesRemaining=sizeBytes; bytesRemaining>0 ;  bytesRemaining -= _bufferSize) {

        size_t theseBytes = (bytesRemaining > _bufferSize) ? _bufferSize : bytesRemaining;

        tprintf (DB_COPY2, "H2D: waiting... on completion signal handle=%lu\n", _completion_signal[bufferIndex].handle);
        hsa_signal_wait_acquire(_completion_signal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

        void * masked_srcp = (void*) ((uintptr_t)srcp & (uintptr_t)(~0x3f)) ; // TODO
        void *locked_srcp;
        hsa_status_t hsa_status = hsa_amd_memory_lock(masked_srcp, theseBytes, &_hsa_agent, 1, &locked_srcp);
        //hsa_status_t hsa_status = hsa_amd_memory_lock(const_cast<char*> (srcp), theseBytes, &_hsa_agent, 1, &locked_srcp);
        tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: pin-in-place:%p+%zu bufferIndex[%d]\n", bytesRemaining, srcp, theseBytes, bufferIndex);
        printf ("status=%x srcp=%p, masked_srcp=%p, locked_srcp=%p\n", hsa_status, srcp, masked_srcp, locked_srcp);

        if (hsa_status != HSA_STATUS_SUCCESS) {
            THROW_ERROR (hipErrorRuntimeMemory);
        }

        hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);

        hsa_status = hsa_amd_memory_async_copy(dstp, _hsa_agent, locked_srcp, _hsa_agent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
        tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: async_copy %zu bytes %p to %p status=%x\n", bytesRemaining, theseBytes, _pinnedStagingBuffer[bufferIndex], dstp, hsa_status);

        if (hsa_status != HSA_STATUS_SUCCESS) {
            THROW_ERROR (hipErrorRuntimeMemory);
        }

        srcp += theseBytes;
        dstp += theseBytes;
        if (++bufferIndex >= _numBuffers) {
            bufferIndex = 0;
        }

        // Assume subsequent commands are dependent on previous and don't need dependency after first copy submitted, HIP_ONESHOT_COPY_DEP=1 
        waitFor = NULL; 
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
//IN: src - src pointer for copy.  Must be accessible from agent this buffer is associated with (via _hsa_agent)
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyHostToDevice(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    std::lock_guard<std::mutex> l (_copy_lock);

    const char *srcp = static_cast<const char*> (src);
    char *dstp = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
    }

    if (sizeBytes >= UINT64_MAX/2) {
        THROW_ERROR (hipErrorInvalidValue);
    }
    int bufferIndex = 0;
    for (int64_t bytesRemaining=sizeBytes; bytesRemaining>0 ;  bytesRemaining -= _bufferSize) {

        size_t theseBytes = (bytesRemaining > _bufferSize) ? _bufferSize : bytesRemaining;

        tprintf (DB_COPY2, "H2D: waiting... on completion signal handle=%lu\n", _completion_signal[bufferIndex].handle);
        hsa_signal_wait_acquire(_completion_signal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);

        tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: copy %zu bytes %p to stagingBuf[%d]:%p\n", bytesRemaining, theseBytes, srcp, bufferIndex, _pinnedStagingBuffer[bufferIndex]);
        // TODO - use uncached memcpy, someday.
        memcpy(_pinnedStagingBuffer[bufferIndex], srcp, theseBytes);


        hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);

        hsa_status_t hsa_status = hsa_amd_memory_async_copy(dstp, _hsa_agent, _pinnedStagingBuffer[bufferIndex], _hsa_agent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
        tprintf (DB_COPY2, "H2D: bytesRemaining=%zu: async_copy %zu bytes %p to %p status=%x\n", bytesRemaining, theseBytes, _pinnedStagingBuffer[bufferIndex], dstp, hsa_status);

        if (hsa_status != HSA_STATUS_SUCCESS) {
            THROW_ERROR ((hipErrorRuntimeMemory));
        }

        srcp += theseBytes;
        dstp += theseBytes;
        if (++bufferIndex >= _numBuffers) {
            bufferIndex = 0;
        }

        // Assume subsequent commands are dependent on previous and don't need dependency after first copy submitted, HIP_ONESHOT_COPY_DEP=1 
        waitFor = NULL; 
    }


    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_wait_acquire(_completion_signal[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    }
}

//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from agent this buffer is associated with (via _hsa_agent).
//IN: src - src pointer for copy.  Must be accessible from host CPU.
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyDeviceToHost(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor)
{
    std::lock_guard<std::mutex> l (_copy_lock);

    const char *srcp0 = static_cast<const char*> (src);
    char *dstp1 = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
    }

    if (sizeBytes >= UINT64_MAX/2) {
        THROW_ERROR (hipErrorInvalidValue);
    }

    int64_t bytesRemaining0 = sizeBytes; // bytes to copy from dest into staging buffer.
    int64_t bytesRemaining1 = sizeBytes; // bytes to copy from staging buffer into final dest

    while (bytesRemaining1 > 0) {
        // First launch the async copies to copy from dest to host
        for (int bufferIndex = 0; (bytesRemaining0>0) && (bufferIndex < _numBuffers);  bytesRemaining0 -= _bufferSize, bufferIndex++) {

            size_t theseBytes = (bytesRemaining0 > _bufferSize) ? _bufferSize : bytesRemaining0;

            tprintf (DB_COPY2, "D2H: bytesRemaining0=%zu  async_copy %zu bytes src:%p to staging:%p\n", bytesRemaining0, theseBytes, srcp0, _pinnedStagingBuffer[bufferIndex]);
            hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(_pinnedStagingBuffer[bufferIndex], _hsa_agent, srcp0, _hsa_agent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
            if (hsa_status != HSA_STATUS_SUCCESS) {
                THROW_ERROR (hipErrorRuntimeMemory);
            }

            srcp0 += theseBytes;


            // Assume subsequent commands are dependent on previous and don't need dependency after first copy submitted, HIP_ONESHOT_COPY_DEP=1 
            waitFor = NULL; 
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
}


//---
//Copies sizeBytes from src to dst, using either a copy to a staging buffer or a staged pin-in-place strategy
//IN: dst - dest pointer - must be accessible from agent this buffer is associated with (via _hsa_agent).
//IN: src - src pointer for copy.  Must be accessible from host CPU.
//IN: waitFor - hsaSignal to wait for - the copy will begin only when the specified dependency is resolved.  May be NULL indicating no dependency.
void StagingBuffer::CopyPeerToPeer(void* dst, hsa_agent_t dstAgent, const void* src, hsa_agent_t srcAgent, size_t sizeBytes, hsa_signal_t *waitFor)
{
    std::lock_guard<std::mutex> l (_copy_lock);

    const char *srcp0 = static_cast<const char*> (src);
    char *dstp1 = static_cast<char*> (dst);

    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_store_relaxed(_completion_signal[i], 0);
        hsa_signal_store_relaxed(_completion_signal2[i], 0);
    }

    if (sizeBytes >= UINT64_MAX/2) {
        THROW_ERROR (hipErrorInvalidValue);
    }

    int64_t bytesRemaining0 = sizeBytes; // bytes to copy from dest into staging buffer.
    int64_t bytesRemaining1 = sizeBytes; // bytes to copy from staging buffer into final dest

    while (bytesRemaining1 > 0) {
        // First launch the async copies to copy from dest to host
        for (int bufferIndex = 0; (bytesRemaining0>0) && (bufferIndex < _numBuffers);  bytesRemaining0 -= _bufferSize, bufferIndex++) {

            size_t theseBytes = (bytesRemaining0 > _bufferSize) ? _bufferSize : bytesRemaining0;

            tprintf (DB_COPY2, "P2P: bytesRemaining0=%zu  async_copy %zu bytes src:%p to staging:%p\n", bytesRemaining0, theseBytes, srcp0, _pinnedStagingBuffer[bufferIndex]);
            hsa_signal_store_relaxed(_completion_signal[bufferIndex], 1);
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(_pinnedStagingBuffer[bufferIndex], srcAgent, srcp0, srcAgent, theseBytes, waitFor ? 1:0, waitFor, _completion_signal[bufferIndex]);
            if (hsa_status != HSA_STATUS_SUCCESS) {
                THROW_ERROR (hipErrorRuntimeMemory);
            }

            srcp0 += theseBytes;


            // Assume subsequent commands are dependent on previous and don't need dependency after first copy submitted, HIP_ONESHOT_COPY_DEP=1 
            waitFor = NULL; 
        }

        // Now unload the staging buffers:
        for (int bufferIndex=0; (bytesRemaining1>0) && (bufferIndex < _numBuffers);  bytesRemaining1 -= _bufferSize, bufferIndex++) {

            size_t theseBytes = (bytesRemaining1 > _bufferSize) ? _bufferSize : bytesRemaining1;

            tprintf (DB_COPY2, "P2P: wait_completion[%d] bytesRemaining=%zu\n", bufferIndex, bytesRemaining1);

            bool hostWait = 0; // TODO - remove me

            if (hostWait) {
                // Host-side wait, should not be necessary:
                hsa_signal_wait_acquire(_completion_signal[bufferIndex], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
            }

            tprintf (DB_COPY2, "P2P: bytesRemaining1=%zu copy %zu bytes stagingBuf[%d]:%p to device:%p\n", bytesRemaining1, theseBytes, bufferIndex, _pinnedStagingBuffer[bufferIndex], dstp1);
            memcpy(dstp1, _pinnedStagingBuffer[bufferIndex], theseBytes);
            hsa_status_t hsa_status = hsa_amd_memory_async_copy(dstp1, dstAgent, _pinnedStagingBuffer[bufferIndex], dstAgent /*not used*/, theseBytes, 
                                      hostWait ? 0:1, hostWait ? NULL : &_completion_signal[bufferIndex], 
                                      _completion_signal2[bufferIndex]);

            dstp1 += theseBytes;
        }
    }


    // Wait for the staging-buffer to dest copies to complete:
    for (int i=0; i<_numBuffers; i++) {
        hsa_signal_wait_acquire(_completion_signal2[i], HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    }
}

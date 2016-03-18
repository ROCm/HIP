#pragma once

#include "hsa.h"


//-------------------------------------------------------------------------------------------------
// An optimized "staging buffer" used to implement Host-To-Device and Device-To-Host copies.
// Some GPUs may not be able to directly access host memory, and in these cases we need to 
// stage the copy through a pinned staging buffer.  For example, the CopyHostToDevice
// uses the CPU to copy to a pinned "staging buffer", and then use the GPU DMA engine to copy
// from the staging buffer to the final destination.  The copy is broken into buffer-sized chunks
// to limit the size of the buffer and also to provide better performance by overlapping the CPU copies 
// with the DMA copies.
//
// PinInPlace is another algorithm which pins the host memory "in-place", and copies it with the DMA
// engine.  This routine is under development.
//
// Staging buffer provides thread-safe access via a mutex.
struct StagingBuffer {

    static const int _max_buffers = 4;

    StagingBuffer(hsa_agent_t hsaAgent, hsa_region_t systemRegion, size_t bufferSize, int numBuffers) ;
    ~StagingBuffer();

    void CopyHostToDevice(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyHostToDevicePinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);

    void CopyDeviceToHost   (void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyDeviceToHostPinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);


private:
    hsa_agent_t     _hsa_agent;
    size_t          _bufferSize;  // Size of the buffers.
    int             _numBuffers;

    char            *_pinnedStagingBuffer[_max_buffers];
    hsa_signal_t     _completion_signal[_max_buffers];
    std::mutex       _copy_lock;    // provide thread-safe access 
};

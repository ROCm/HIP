#pragma once

#include "hsa.h"

struct ihipDevice_t;

//-------------------------------------------------------------------------------------------------
struct StagingBuffer {

    static const int _max_buffers = 4;

    StagingBuffer(ihipDevice_t *device, size_t bufferSize, int numBuffers) ;
    ~StagingBuffer();

    void CopyHostToDevice(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyHostToDevicePinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);

    void CopyDeviceToHost   (void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyDeviceToHostPinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);


private:
    ihipDevice_t    *_device;
    size_t          _bufferSize;  // Size of the buffers.
    int             _numBuffers;

    char            *_pinnedStagingBuffer[_max_buffers];
    hsa_signal_t     _completion_signal[_max_buffers];
};

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

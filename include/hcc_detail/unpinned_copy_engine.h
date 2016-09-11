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

//#pragma once
#ifndef STAGING_BUFFER_H
#define STAGING_BUFFER_H

#include "hsa/hsa.h"


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
struct UnpinnedCopyEngine {

    enum CopyMode {ChooseBest, UsePinInPlace, UseStaging, UseMemcpy} ; 

    static const int _max_buffers = 4;

    UnpinnedCopyEngine(hsa_agent_t hsaAgent,hsa_agent_t cpuAgent, size_t bufferSize, int numBuffers, 
                       bool isLargeBar, int thresholdH2D_directStaging, int thresholdH2D_stagingPinInPlace, int thresholdD2H) ;
    ~UnpinnedCopyEngine();

    // Use hueristic to choose best copy algorithm 
    void CopyHostToDevice(CopyMode copyMode, void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyDeviceToHost(CopyMode copyMode, void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);


    // Specific H2D copy algorithm implementations:
    void CopyHostToDeviceStaging(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyHostToDevicePinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyHostToDeviceMemcpy(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);


    // Specific D2H copy algorithm implementations:
    void CopyDeviceToHostStaging(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);
    void CopyDeviceToHostPinInPlace(void* dst, const void* src, size_t sizeBytes, hsa_signal_t *waitFor);


    // P2P Copy implementation:
    void CopyPeerToPeer( void* dst, hsa_agent_t dstAgent, const void* src, hsa_agent_t srcAgent, size_t sizeBytes, hsa_signal_t *waitFor);


private:
    hsa_agent_t     _hsaAgent;
    hsa_agent_t     _cpuAgent;
    size_t          _bufferSize;  // Size of the buffers.
    int             _numBuffers;

    // True if system supports large-bar and thus can benefit from CPU directly performing copy operation.
    bool            _isLargeBar;

    char            *_pinnedStagingBuffer[_max_buffers];
    hsa_signal_t     _completionSignal[_max_buffers];
    hsa_signal_t     _completionSignal2[_max_buffers]; // P2P needs another set of signals.
    std::mutex       _copyLock;    // provide thread-safe access
    size_t              _hipH2DTransferThresholdDirectOrStaging;
    size_t              _hipH2DTransferThresholdStagingOrPininplace;
    size_t              _hipD2HTransferThreshold;
};

#endif

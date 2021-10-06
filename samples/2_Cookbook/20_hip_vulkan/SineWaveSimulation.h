/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Modifications Copyright (C)2021 Advanced
 * Micro Devices, Inc. All rights reserved.
 */

#pragma once
#ifndef __SINESIM_H__
#define __SINESIM_H__

#include <vector>
#include <hip/hip_runtime_api.h>
#include <stdint.h>
#include "linmath.h"

class SineWaveSimulation
{
    float *m_heightMap;
    size_t m_width, m_height;
    int m_blocks, m_threads;
public:
    SineWaveSimulation(size_t width, size_t height);
    ~SineWaveSimulation();
    void initSimulation(float *heightMap);
    void stepSimulation(float time, hipStream_t stream = 0);
    void initCudaLaunchConfig(int device);
    int initCuda(uint8_t  *vkDeviceUUID, size_t UUID_SIZE);

    size_t getWidth() const {
        return m_width;
    }
    size_t getHeight() const {
        return m_height;
    }
};

template <typename T>
void check(T result, char const* const func, const char* const file,
  int const line) {
  if (result) {
    fprintf(stderr, "HIP error at %s:%d code=%d \"%s\" \n", file, line, static_cast<unsigned int>(result), func);
     // static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkHIPErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastHIPError(msg) __getLastHIPError(msg, __FILE__, __LINE__)

inline void __getLastHIPError(const char* errorMessage, const char* file,
  const int line) {
  hipError_t err = hipGetLastError();

  if (hipSuccess != err) {
    fprintf(stderr,
      "%s(%i) : getLastHIPError() HIP error :"
      " %s : (%d) %s.\n",
      file, line, errorMessage, static_cast<int>(err),
      hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
}



#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#endif // __SINESIM_H__

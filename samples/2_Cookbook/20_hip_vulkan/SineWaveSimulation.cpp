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

#include "SineWaveSimulation.h"
#include <algorithm>
//#include <helper_cuda.h>
#include "hip/hip_runtime.h"


__global__ void sinewave(float *heightMap, unsigned int width, unsigned int height, float time)
{
    const float freq = 4.0f;
    const size_t stride = gridDim.x * blockDim.x;

    // Iterate through the entire array in a way that is
    // independent of the grid configuration
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < width * height; tid += stride) {
        // Calculate the x, y coordinates
        const size_t y = tid / width;
        const size_t x = tid - y * width;
        // Normalize x, y to [0,1]
        const float u = ((2.0f * x) / width)  - 1.0f;
        const float v = ((2.0f * y) / height) - 1.0f;
        // Calculate the new height value
        const float w = 0.5f * sinf(u * freq + time) * cosf(v * freq + time);
        // Store this new height value
        heightMap[tid] = w;
    }
}

SineWaveSimulation::SineWaveSimulation(size_t width, size_t height) 
                                        : m_heightMap(nullptr), m_width(width), m_height(height)
{
}

void SineWaveSimulation::initCudaLaunchConfig(int device)
{
    hipDeviceProp_t prop = {};
    checkHIPErrors(hipSetDevice(device));
    checkHIPErrors(hipGetDeviceProperties(&prop, device));

    // We don't need large block sizes, since there's not much inter-thread communication
    m_threads = prop.warpSize;

    // Use the occupancy calculator and fill the gpu as best as we can
    checkHIPErrors(hipOccupancyMaxActiveBlocksPerMultiprocessor(&m_blocks, sinewave, prop.warpSize, 0));
    m_blocks *= prop.multiProcessorCount;

    // Go ahead and the clamp the blocks to the minimum needed for this height/width
    m_blocks = std::min(m_blocks, (int)((m_width * m_height + m_threads - 1) / m_threads));
}

int SineWaveSimulation::initCuda(uint8_t  *vkDeviceUUID, size_t UUID_SIZE)
{
    int current_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    hipDeviceProp_t deviceProp;
    checkHIPErrors(hipGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the GPU which is selected by Vulkan
    while (current_device < device_count) {
        hipGetDeviceProperties(&deviceProp, current_device);

        if ((deviceProp.computeMode != hipComputeModeProhibited)) {
            // Compare the cuda device UUID with vulkan UUID
            // FIXME
            int ret = 0; // memcmp((void*)&deviceProp.uuid, vkDeviceUUID, UUID_SIZE);
            if (ret == 0)
            {
                checkHIPErrors(hipSetDevice(current_device));
                checkHIPErrors(hipGetDeviceProperties(&deviceProp, current_device));
                printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                 current_device, deviceProp.name, deviceProp.major,
                 deviceProp.minor);

                return current_device;
            }

        } else {
          devices_prohibited++;
        }

        current_device++;
    }

    if (devices_prohibited == device_count) {
        fprintf(stderr,
                "HIP error:"
                " No Vulkan-HIP Interop capable GPU found.\n");
        exit(EXIT_FAILURE);
    }

    return -1;
}

SineWaveSimulation::~SineWaveSimulation()
{
    m_heightMap = NULL;
}
 
void SineWaveSimulation::initSimulation(float *heights)
{
    m_heightMap = heights;
}

void SineWaveSimulation::stepSimulation(float time, hipStream_t stream)
{
    hipLaunchKernelGGL(sinewave, dim3(m_blocks), dim3(m_threads), 0, stream , m_heightMap, m_width, m_height, time);
    getLastHIPError("Failed to launch CUDA simulation");
    //hipStreamSynchronize(stream);
}

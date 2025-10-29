// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "backprojection.hpp"
#include "filtering.hpp"
#include "log_transform.hpp"
#include "normalization.hpp"
#include "phantom.hpp"
#include "projection.hpp"
#include "utility.hpp"
#include "weighting.hpp"
#include "volume.hpp"

#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numbers>
#include <ostream>
#include <set>
#include <stdexcept>
#include <vector>

auto main() -> int
{
    try
    {
        auto hasTextures = int{0};
        hip_check(hipDeviceGetAttribute(&hasTextures, hipDeviceAttributeImageSupport, 0));

        // [sphinx-async-engine-start]
        // Fetch device properties
        auto devProps = hipDeviceProp_t{};
        hip_check(hipGetDeviceProperties(&devProps, 0));

        auto const numStreams = devProps.asyncEngineCount;

        std::cout << "Device has " << numStreams << " asynchronous engines; preprocessing will use "
                  << numStreams << " parallel streams." << std::endl;

        auto streams = std::vector<hipStream_t>{};
        streams.resize(numStreams);
        for(auto&& stream : streams)
            hip_check(hipStreamCreate(&stream));
        // [sphinx-async-engine-end]

        auto r = static_cast<float*>(nullptr);
        auto R = static_cast<hipfftComplex*>(nullptr);

        auto forwardPlans = std::vector<hipfftHandle>{};
        auto forwardSizes = std::vector<std::size_t>{};
        auto backwardPlans = std::vector<hipfftHandle>{};
        auto backwardSizes = std::vector<std::size_t>{};
        forwardPlans.resize(numStreams);
        forwardSizes.resize(numStreams);
        backwardPlans.resize(numStreams);
        backwardSizes.resize(numStreams);

        auto projections = std::vector<float*>{};
        auto projectionPitches = std::vector<std::size_t>{};
        auto expandedProjections = std::vector<float*>{};
        auto expandedPitches = std::vector<std::size_t>{};
        auto transformedProjections = std::vector<hipfftComplex*>{};
        auto transformedPitches = std::vector<std::size_t>{};
        auto textureProjections = std::vector<hipTextureObject_t>{};

        auto projGeom = phantom::make_projectionGeometry();
        auto volGeom = phantom::make_volumeGeometry();
        auto phantomProjections = phantom::make_projections(projGeom, volGeom, streams);

        std::cout << "Initializing... " << std::flush;

        auto stream = streams.at(0);

        // Create filter kernel
        hip_check(hipMalloc(reinterpret_cast<void**>(&r), projGeom.dimFFT.x * sizeof(float)));
        auto const creationBlocks = std::max((projGeom.dimFFT.x / 1024u), 1u);
        filter_creation_kernel<<<creationBlocks, 1024, 0, stream>>>(r, projGeom.s_dimFFT.x, projGeom.pixelDim.x);

        hip_check(hipMalloc(reinterpret_cast<void**>(&R), projGeom.dimTrans.x * sizeof(hipfftComplex)));
        auto filterPlan = hipfftHandle{};
        hipfft_check(hipfftPlan1d(&filterPlan, projGeom.dimFFT.x, HIPFFT_R2C, 1));
        hipfft_check(hipfftSetStream(filterPlan, stream));
        hipfft_check(hipfftExecR2C(filterPlan, r, R));

        auto absoluteBlocks = (projGeom.dimTrans.x / 1024u) + 1u;
        filter_absolute_kernel<<<absoluteBlocks, 1024, 0, stream>>>(R, projGeom.dimTrans.x, projGeom.pixelDim.x);

        hip_check(hipStreamSynchronize(stream));
                
        hipfft_check(hipfftDestroy(filterPlan));
        hip_check(hipFree(r));

        auto const inputProjSingle = projGeom.dim.x * projGeom.dim.y * sizeof(std::uint16_t);
        auto const inputProjTotal = inputProjSingle * numStreams;
        auto const projSingle = projGeom.dim.x * projGeom.dim.y * sizeof(float);
        auto const projTotal = projSingle * numStreams;
        auto const expandedSingle = projGeom.dimFFT.x * projGeom.dimFFT.y * sizeof(float);
        auto const expandedTotal = expandedSingle * numStreams;
        auto const transformedSingle = projGeom.dimTrans.x * projGeom.dimTrans.y * sizeof(hipfftComplex);
        auto const transformedTotal = transformedSingle * numStreams;
        auto const volumeTotal = volGeom.dim.x * volGeom.dim.y * volGeom.dim.z * sizeof(float);
        auto const memTotal = inputProjTotal + projTotal + expandedTotal + transformedTotal + volumeTotal;
        auto devMemFree = std::size_t{};
        auto devMemTotal = std::size_t{};
        hip_check(hipMemGetInfo(&devMemFree, &devMemTotal));

        auto memRequired = static_cast<std::size_t>(memTotal);
        if(memRequired > devMemFree)
        {
            std::cerr << "Not enough device memory. Required: " << memRequired 
                      << ", available: " << devMemFree << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "Done!" << std::endl;
        std::cout << "Volume dimensions: " << volGeom.dim.x << " x "
                                           << volGeom.dim.y << " x "
                                           << volGeom.dim.z << std::endl;

        // Initialize per-stream data
        for(auto streamIdx = 0u; streamIdx < streams.size(); ++streamIdx)
        {
            std::cout << "Initializing stream " << streamIdx << "... " << std::flush;
            auto stream = streams.at(streamIdx);

            auto proj = static_cast<float*>(nullptr);
            auto projPitch = std::size_t{};
            hip_check(hipMallocPitch(
                reinterpret_cast<void**>(&proj), &projPitch, projGeom.dim.x * sizeof(float), projGeom.dim.y
            ));
            projections.push_back(proj);
            projectionPitches.push_back(projPitch);

            auto expanded = static_cast<float*>(nullptr);
            auto expandedPitch = std::size_t{};
            hip_check(hipMallocPitch(
                reinterpret_cast<void**>(&expanded),
                &expandedPitch,
                projGeom.dimFFT.x * sizeof(float),
                projGeom.dimFFT.y
            ));
            expandedProjections.push_back(expanded);
            expandedPitches.push_back(expandedPitch);
            
            auto transformed = static_cast<hipfftComplex*>(nullptr);
            auto transformedPitch = std::size_t{};
            hip_check(hipMallocPitch(
                reinterpret_cast<void**>(&transformed),
                &transformedPitch,
                projGeom.dimTrans.x * sizeof(hipfftComplex),
                projGeom.dimTrans.y
            ));
            transformedProjections.push_back(transformed);
            transformedPitches.push_back(transformedPitch);

            auto& forward = forwardPlans.at(streamIdx);
            auto& forwardSize = forwardSizes.at(streamIdx);
            auto fw_inembed = static_cast<int>(expandedPitch / sizeof(float));
            auto fw_istride = 1;
            auto fw_idist = fw_inembed;
            auto fw_onembed = static_cast<int>(transformedPitch / sizeof(hipfftComplex));
            auto fw_ostride = 1;
            auto fw_odist = fw_onembed;
            hipfft_check(hipfftCreate(&forward));
            hipfft_check(hipfftMakePlanMany(forward, 1, &projGeom.s_dimFFT.x,
                                            &fw_inembed, 1, fw_idist,
                                            &fw_onembed, 1, fw_odist,
                                            HIPFFT_R2C, projGeom.s_dimFFT.y, &forwardSize));
            hipfft_check(hipfftSetStream(forward, stream));

            auto& backward = backwardPlans.at(streamIdx);
            auto& backwardSize = backwardSizes.at(streamIdx);
            auto bw_inembed = fw_onembed;
            auto bw_istride = fw_ostride;
            auto bw_idist = fw_odist;
            auto bw_onembed = fw_inembed;
            auto bw_ostride = fw_istride;
            auto bw_odist = fw_idist;
            hipfft_check(hipfftCreate(&backward));
            hipfft_check(hipfftMakePlanMany(backward, 1, &projGeom.s_dimFFT.x,
                                            &bw_inembed, bw_istride, bw_idist,
                                            &bw_onembed, bw_ostride, bw_odist,
                                            HIPFFT_C2R, projGeom.s_dimFFT.y, &backwardSize));
            hipfft_check(hipfftSetStream(backward, stream));

            if(hasTextures)
            {
                // create a HIP texture from the projection
                auto resDesc = hipResourceDesc{};
                resDesc.resType = hipResourceTypePitch2D;
                resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();
                resDesc.res.pitch2D.devPtr = static_cast<void*>(proj);
                resDesc.res.pitch2D.width = projGeom.dim.x;
                resDesc.res.pitch2D.height = projGeom.dim.y;
                resDesc.res.pitch2D.pitchInBytes = projPitch;

                auto texDesc = hipTextureDesc{};
                texDesc.addressMode[0] = hipAddressModeBorder;
                texDesc.addressMode[1] = hipAddressModeBorder;
                texDesc.readMode = hipReadModeElementType;
                texDesc.borderColor[0] = 0.f;
                texDesc.borderColor[0] = 0.f;
                texDesc.filterMode = hipFilterModeLinear;
                texDesc.normalizedCoords = 0;

                auto& projTex = textureProjections.emplace_back();
                hip_check(hipCreateTextureObject(&projTex, &resDesc, &texDesc, nullptr));
            }

            std::cout << "Done!" << std::endl;
        }

        create_volume("volume.tif");
        auto hostVolPtr = static_cast<float*>(nullptr);
        hip_check(hipHostMalloc(
            reinterpret_cast<void**>(&hostVolPtr),
            volGeom.dim.x * volGeom.dim.y * volGeom.dim.z * sizeof(float),
            hipHostMallocDefault
        ));
        auto hostVol = make_hipPitchedPtr(
            hostVolPtr, volGeom.dim.x * sizeof(float), volGeom.dim.x, volGeom.dim.y
        );    
        auto vol = hipPitchedPtr{};
        auto volExt = make_hipExtent(volGeom.dim.x * sizeof(float), volGeom.dim.y, volGeom.dim.z);
        hip_check(hipMalloc3D(&vol, volExt));
        hip_check(hipMemset3D(vol, 0, volExt));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // MAIN LOOP
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        auto start = std::chrono::steady_clock::now();

        // [sphinx-batch-start]
        auto projIdx = 0u;
        while(projIdx < projGeom.numProj)
        {
            auto batchSize = std::min(numStreams, static_cast<int>(projGeom.numProj - projIdx));

            // Launch batch in parallel streams
            for(auto streamIdx = 0; streamIdx < batchSize; ++streamIdx, ++projIdx)
            {
                auto stream = streams.at(streamIdx);
                // [sphinx-batch-end]

                auto threadsPerBlock = dim3{32, 32, 1};
                auto blocksPerGrid = dim3{
                    (projGeom.dim.x / threadsPerBlock.x) + 1, (projGeom.dim.y / threadsPerBlock.y) + 1, 1
                };

                auto inputPitchedPtr = phantomProjections.at(projIdx);
                auto input = static_cast<std::uint16_t*>(inputPitchedPtr.ptr);
                auto inputPitch = inputPitchedPtr.pitch;

                // [sphinx-preprocessing-start]
                ////////////////////////////////////////////////////////////////////////////////////////////////////
                // START HERE
                ////////////////////////////////////////////////////////////////////////////////////////////////////
                auto proj = projections.at(streamIdx);
                auto projPitch = projectionPitches.at(streamIdx);
                normalization_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    input, inputPitch, proj, projPitch, projGeom.dim, projGeom.bps
                );

                log_transformation_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(proj, projPitch, projGeom.dim);

                weighting_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    proj,
                    projPitch,
                    projGeom.dim,
                    projGeom.d_sd,
                    projGeom.d_so,
                    projGeom.minCoord,
                    projGeom.pixelDim
                );
                // [sphinx-preprocessing-end]

                // [sphinx-proj-to-expanded-start]
                // Expand projection to filter length
                auto expanded = expandedProjections.at(streamIdx);
                auto expandedPitch = expandedPitches.at(streamIdx);
                hip_check(hipMemset2DAsync(
                    expanded, expandedPitch, 0, projGeom.dimFFT.x * sizeof(float), projGeom.dimFFT.y, stream
                ));
                hip_check(hipMemcpy2DAsync(
                    expanded,
                    expandedPitch,
                    proj,
                    projPitch,
                    projGeom.dim.x * sizeof(float),
                    projGeom.dim.y,
                    hipMemcpyDeviceToDevice,
                    stream
                ));
                // [sphinx-proj-to-expanded-end]

                // [sphinx-forward-start]
                // R2C Fourier-transform projection
                auto transformed = transformedProjections.at(streamIdx);
                auto transformedPitch = transformedPitches.at(streamIdx);
                hip_check(hipMemset2DAsync(
                    transformed,
                    transformedPitch,
                    0,
                    projGeom.dimTrans.x * sizeof(hipfftComplex),
                    projGeom.dimTrans.y,
                    stream
                ));
                auto& forward = forwardPlans.at(streamIdx);
                hipfft_check(hipfftExecR2C(forward, expanded, transformed));
                // [sphinx-forward-end]
                    
                // [sphinx-filter-start]
                // Apply filter
                auto filterBlocksPerGrid = dim3{
                    (projGeom.dimTrans.x / threadsPerBlock.x) + 1,
                    (projGeom.dimTrans.y / threadsPerBlock.y) + 1,
                    1
                };
                filter_application_kernel<<<filterBlocksPerGrid, threadsPerBlock, 0, stream>>>(
                    transformed, transformedPitch, R, projGeom.dimTrans
                );
                    
                auto& backward = backwardPlans.at(streamIdx);
                hipfft_check(hipfftExecC2R(backward, transformed, expanded));
                // [sphinx-filter-end]

                // [sphinx-expanded-to-proj-start]
                // Shrink projection to original size and normalize
                hip_check(hipMemcpy2DAsync(
                    proj,
                    projPitch,
                    expanded,
                    expandedPitch,
                    projGeom.dim.x * sizeof(float),
                    projGeom.dim.y,
                    hipMemcpyDeviceToDevice,
                    stream
                ));
                    
                filter_normalization_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    proj, projPitch, projGeom.dimFFT.x, projGeom.dim
                );
                // [sphinx-expanded-to-proj-end]

                // [sphinx-bp-start]
                // Backprojection
                auto thetaDeg = projGeom.thetaSign * projGeom.thetaStep * projIdx; // Current angle
                auto thetaRad = thetaDeg * std::numbers::pi_v<float> / 180.f; // Convert to radians
                auto sinTheta = std::sin(thetaRad);
                auto cosTheta = std::cos(thetaRad);

                auto bpBlockSize = dim3{32u, 8u, 4u};
                auto bpBlocks = dim3{
                    static_cast<std::uint32_t>(volGeom.dim.x / bpBlockSize.x + 1),
                    static_cast<std::uint32_t>(volGeom.dim.y / bpBlockSize.y + 1),
                    static_cast<std::uint32_t>(volGeom.dim.z / bpBlockSize.z + 1)
                };

                if(hasTextures)
                {
                    auto& projTex = textureProjections.at(streamIdx);
                    backprojection_kernel<<<bpBlocks, bpBlockSize, 0, stream>>>(
                        static_cast<float*>(vol.ptr),
                        vol.pitch,
                        volGeom.dim,
                        volGeom.voxelDim,
                        projTex,
                        projGeom.minCoord,
                        sinTheta,
                        cosTheta,
                        projGeom.pixelDim,
                        projGeom.d_sd,
                        projGeom.d_so
                    );
                }
                else
                {
                    // Fallback for devices without support for texture instructions
                    backprojection_kernel_no_tex<<<bpBlocks, bpBlockSize, 0, stream>>>(
                        static_cast<float*>(vol.ptr),
                        vol.pitch,
                        volGeom.dim,
                        volGeom.voxelDim,
                        proj,
                        projPitch,
                        projGeom.dim,
                        projGeom.minCoord,
                        sinTheta,
                        cosTheta,
                        projGeom.pixelDim,
                        projGeom.d_sd,
                        projGeom.d_so
                    );
                }
                // [sphinx-bp-end]
            }
        }
        
        // [sphinx-sync-start]
        // First stream waits for other streams to complete
        auto completionEvents = std::vector<hipEvent_t>{};
        for(auto streamIdx = 1u; streamIdx < streams.size(); ++streamIdx)
        {
            auto event = hipEvent_t{};
            hip_check(hipEventCreate(&event));
            hip_check(hipEventRecord(event, streams.at(streamIdx)));
            completionEvents.push_back(event);
        }

        for(auto&& event : completionEvents)
            hip_check(hipStreamWaitEvent(streams.at(0), event, 0));
        // [sphinx-sync-end]

        // Obtain reconstruction time before copying back the result
        auto stop = std::chrono::steady_clock::time_point{};
        hip_check(hipLaunchHostFunc(streams.at(0), [](void* data)
        {
            auto& stop = *(static_cast<std::chrono::steady_clock::time_point*>(data));
            stop = std::chrono::steady_clock::now();
        }, static_cast<void*>(&stop)));

        // Copy volume back to host and save
        auto memcpyParams = hipMemcpy3DParms{};
        std::memset(&memcpyParams, 0, sizeof(hipMemcpy3DParms));
        memcpyParams.dstPos = make_hipPos(0, 0, 0);
        memcpyParams.dstPtr = hostVol;
        memcpyParams.srcPos = make_hipPos(0, 0, 0);
        memcpyParams.srcPtr = vol;
        memcpyParams.extent = volExt;
        memcpyParams.kind = hipMemcpyDeviceToHost;
        hip_check(hipMemcpy3DAsync(&memcpyParams, streams.at(0)));
            
        auto saveVolArgs = new save_volume_args
        {
            "volume.tif",
            hostVolPtr,
            volGeom.dim.x, volGeom.dim.y, volGeom.dim.z,
            volGeom.voxelDim.x, volGeom.voxelDim.y
        };
        hip_check(hipLaunchHostFunc(streams.at(0), save_volume, saveVolArgs));

        std::cout << "All work items enqueued, waiting for completion... " << std::flush;
        hip_check(hipStreamSynchronize(streams.at(0)));
        std::cout << "Done!" << std::endl;

        auto const elapsed = std::chrono::duration<double>{stop - start};
        std::cout << "Reconstruction time: " << elapsed.count() << 's' << std::endl;

        for(auto&& event : completionEvents)
            hip_check(hipEventDestroy(event));

        hip_check(hipFree(vol.ptr));
        hip_check(hipFreeHost(hostVolPtr));

        if(hasTextures)
        {
            for(auto&& tex : textureProjections)
                hip_check(hipDestroyTextureObject(tex));
        }

        for(auto&& plan : backwardPlans)
            hipfft_check(hipfftDestroy(plan));

        for(auto&& plan : forwardPlans)
            hipfft_check(hipfftDestroy(plan));

        for(auto&& p : transformedProjections)
            hip_check(hipFree(p));

        for(auto&& p : expandedProjections)
            hip_check(hipFree(p));

        for(auto&& p : projections)
            hip_check(hipFree(p));

        for(auto&& p : phantomProjections)
            hip_check(hipFree(p.ptr));

        hip_check(hipFree(R));

        for(auto&& stream : streams)
            hip_check(hipStreamDestroy(stream));

        hip_check(hipDeviceSynchronize());

        return EXIT_SUCCESS;
    }
    catch(std::runtime_error const& e)
    {
        std::cerr << "Caught runtime error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

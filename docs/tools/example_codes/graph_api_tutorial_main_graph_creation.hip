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
#include <iterator>
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

        // Fetch device properties
        auto devProps = hipDeviceProp_t{};
        hip_check(hipGetDeviceProperties(&devProps, 0));

        auto const numBranches = devProps.asyncEngineCount;

        std::cout << "Device supports " << numBranches << " asynchronous engines; preprocessing will use "
                  << numBranches << " parallel branches." << std::endl;
        
        // For interoperability with hipFFT we require streams and events
        auto streams = std::vector<hipStream_t>{};
        streams.resize(numBranches);
        for(auto&& stream : streams)
            hip_check(hipStreamCreate(&stream));

        auto r = static_cast<float*>(nullptr);
        auto R = static_cast<hipfftComplex*>(nullptr);

        auto forwardPlans = std::vector<hipfftHandle>{};
        auto forwardSizes = std::vector<std::size_t>{};
        auto backwardPlans = std::vector<hipfftHandle>{};
        auto backwardSizes = std::vector<std::size_t>{};
        forwardPlans.resize(numBranches);
        forwardSizes.resize(numBranches);
        backwardPlans.resize(numBranches);
        backwardSizes.resize(numBranches);

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
        auto const inputProjTotal = inputProjSingle * numBranches;
        auto const projSingle = projGeom.dim.x * projGeom.dim.y * sizeof(float);
        auto const projTotal = projSingle * numBranches;
        auto const expandedSingle = projGeom.dimFFT.x * projGeom.dimFFT.y * sizeof(float);
        auto const expandedTotal = expandedSingle * numBranches;
        auto const transformedSingle = projGeom.dimTrans.x * projGeom.dimTrans.y * sizeof(hipfftComplex);
        auto const transformedTotal = transformedSingle * numBranches;
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

        // Initialize per-branch data
        for(auto branchIdx = 0; branchIdx < numBranches; ++branchIdx)
        {
            std::cout << "Initializing branch #" << branchIdx << "... " << std::flush;
            auto stream = streams.at(branchIdx);

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

            auto& forward = forwardPlans.at(branchIdx);
            auto& forwardSize = forwardSizes.at(branchIdx);
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

            auto& backward = backwardPlans.at(branchIdx);
            auto& backwardSize = backwardSizes.at(branchIdx);
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
        auto graphCreated = false;
        auto graphExec = hipGraphExec_t{};
        auto graphFinalCreated = false;
        auto graphExecFinal = hipGraphExec_t{};
        auto graphStream = hipStream_t{};
        hip_check(hipStreamCreate(&graphStream));

        auto start = std::chrono::steady_clock::now();

        auto projIdx = 0u;
        while(projIdx < projGeom.numProj)
        {
            auto batchSize = std::min(numBranches, static_cast<int>(projGeom.numProj - projIdx));

            // Create graph for current batch
            auto graph = hipGraph_t{};
            hip_check(hipGraphCreate(&graph, 0));

            // Add nodes for each projection in batch
            for(auto branchIdx = 0; branchIdx < batchSize; ++branchIdx, ++projIdx)
            {
                auto stream = streams.at(branchIdx);

                auto threadsPerBlock = dim3{32, 32, 1};
                auto blocksPerGrid = dim3{
                    (projGeom.dim.x / threadsPerBlock.x) + 1, (projGeom.dim.y / threadsPerBlock.y) + 1, 1
                };

                auto inputPitchedPtr = phantomProjections.at(projIdx);
                auto input = static_cast<std::uint16_t*>(inputPitchedPtr.ptr);
                auto inputPitch = inputPitchedPtr.pitch;

                auto proj = projections.at(branchIdx);
                auto projPitch = projectionPitches.at(branchIdx);
                void* normalizationKernelParams[] =
                {
                    static_cast<void*>(&input),
                    static_cast<void*>(&inputPitch),
                    static_cast<void*>(&proj),
                    static_cast<void*>(&projPitch),
                    static_cast<void*>(&projGeom.dim),
                    static_cast<void*>(&projGeom.bps)
                };

                auto normalizationKernelNodeParams = hipKernelNodeParams{};
                normalizationKernelNodeParams.blockDim = threadsPerBlock;
                normalizationKernelNodeParams.extra = nullptr;
                normalizationKernelNodeParams.func = reinterpret_cast<void*>(normalization_kernel);
                normalizationKernelNodeParams.gridDim = blocksPerGrid;
                normalizationKernelNodeParams.kernelParams = normalizationKernelParams;
                normalizationKernelNodeParams.sharedMemBytes = 0;
                auto normalizationKernelNode = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &normalizationKernelNode, graph, nullptr, 0, &normalizationKernelNodeParams
                ));

                void* logTransformationKernelParams[] =
                {
                    static_cast<void*>(&proj),
                    static_cast<void*>(&projPitch),
                    static_cast<void*>(&projGeom.dim)
                };
                auto logTransformationKernelNodeParams = hipKernelNodeParams{};
                logTransformationKernelNodeParams.blockDim = threadsPerBlock;
                logTransformationKernelNodeParams.extra = nullptr;
                logTransformationKernelNodeParams.func = reinterpret_cast<void*>(log_transformation_kernel);
                logTransformationKernelNodeParams.gridDim = blocksPerGrid;
                logTransformationKernelNodeParams.kernelParams = logTransformationKernelParams;
                logTransformationKernelNodeParams.sharedMemBytes = 0;
                auto logTransformationKernelNode = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &logTransformationKernelNode,
                    graph,
                    &normalizationKernelNode,
                    1,
                    &logTransformationKernelNodeParams
                ));

                // [sphinx-weighting-node-start]
                void* weightingKernelParams[] =
                {
                    static_cast<void*>(&proj),
                    static_cast<void*>(&projPitch),
                    static_cast<void*>(&projGeom.dim),
                    static_cast<void*>(&projGeom.d_sd),
                    static_cast<void*>(&projGeom.d_so),
                    static_cast<void*>(&projGeom.minCoord),
                    static_cast<void*>(&projGeom.pixelDim)
                };
                auto weightingKernelNodeParams = hipKernelNodeParams{};
                weightingKernelNodeParams.blockDim = threadsPerBlock;
                weightingKernelNodeParams.extra = nullptr;
                weightingKernelNodeParams.func = reinterpret_cast<void*>(weighting_kernel);
                weightingKernelNodeParams.gridDim = blocksPerGrid;
                weightingKernelNodeParams.kernelParams = weightingKernelParams;
                weightingKernelNodeParams.sharedMemBytes = 0;
                auto weightingKernelNode = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &weightingKernelNode, graph, &logTransformationKernelNode, 1, &weightingKernelNodeParams
                ));
                // [sphinx-weighting-node-end]

                // Expand projection to filter length
                auto expanded = expandedProjections.at(branchIdx);
                auto expandedPitch = expandedPitches.at(branchIdx);
                // [sphinx-memset-node-start]
                auto expandedMemsetNodeParams = hipMemsetParams{};
                expandedMemsetNodeParams.dst = static_cast<void*>(expanded);
                expandedMemsetNodeParams.elementSize = sizeof(float);
                expandedMemsetNodeParams.height = projGeom.dimFFT.y;
                expandedMemsetNodeParams.pitch = expandedPitch;
                expandedMemsetNodeParams.value = 0;
                expandedMemsetNodeParams.width = projGeom.dimFFT.x;
                auto expandedMemsetNode = hipGraphNode_t{};
                hip_check(hipGraphAddMemsetNode(
                    &expandedMemsetNode, graph, &weightingKernelNode, 1, &expandedMemsetNodeParams
                ));
                // [sphinx-memset-node-end]

                auto copyProjToExpandedNodeParams = hipMemcpy3DParms{};
                std::memset(&copyProjToExpandedNodeParams, 0, sizeof(hipMemcpy3DParms));
                copyProjToExpandedNodeParams.srcPos = make_hipPos(0, 0, 0);
                copyProjToExpandedNodeParams.srcPtr = make_hipPitchedPtr(
                    static_cast<void*>(proj), projPitch, projGeom.dim.x, projGeom.dim.y);
                copyProjToExpandedNodeParams.dstPos = make_hipPos(0, 0, 0);
                copyProjToExpandedNodeParams.dstPtr = make_hipPitchedPtr(
                    static_cast<void*>(expanded), expandedPitch, projGeom.dimFFT.x, projGeom.dimFFT.y);
                copyProjToExpandedNodeParams.extent = make_hipExtent(
                    projGeom.dim.x * sizeof(float), projGeom.dim.y, 1);
                copyProjToExpandedNodeParams.kind = hipMemcpyDeviceToDevice;
                auto copyProjToExpandedNode = hipGraphNode_t{};
                hip_check(hipGraphAddMemcpyNode(
                    &copyProjToExpandedNode,
                    graph,
                    &expandedMemsetNode,
                    1,
                    &copyProjToExpandedNodeParams
                ));

                // R2C Fourier-transform projection
                auto transformed = transformedProjections.at(branchIdx);
                auto transformedPitch = transformedPitches.at(branchIdx);
                auto transformedMemsetNodeParams = hipMemsetParams{};
                transformedMemsetNodeParams.dst = static_cast<void*>(transformed);
                transformedMemsetNodeParams.elementSize = sizeof(float); // elementSize maximum is 4 bytes
                transformedMemsetNodeParams.height = projGeom.dimTrans.y;
                transformedMemsetNodeParams.pitch = transformedPitch;
                transformedMemsetNodeParams.value = 0;
                transformedMemsetNodeParams.width = projGeom.dimTrans.x * 2; // hipfftComplex = 2 floats
                auto transformedMemsetNode = hipGraphNode_t{};
                hip_check(hipGraphAddMemsetNode(
                    &transformedMemsetNode, graph, &copyProjToExpandedNode, 1, &transformedMemsetNodeParams
                ));

                // [sphinx-before-forward-start]
                // Before capturing the FFT operations, obtain the set of nodes already present in the graph
                auto nodesBeforeForward = std::vector<hipGraphNode_t>{};
                auto numNodesBeforeForward = std::size_t{};
                hip_check(hipGraphGetNodes(graph, nullptr, &numNodesBeforeForward));
                nodesBeforeForward.resize(numNodesBeforeForward);
                hip_check(hipGraphGetNodes(graph, nodesBeforeForward.data(), &numNodesBeforeForward));
                auto nodesBeforeForwardSorted = std::set<hipGraphNode_t>{
                    std::begin(nodesBeforeForward), std::end(nodesBeforeForward)
                };
                // [sphinx-before-forward-end]

                // [sphinx-hipfft-start]
                hip_check(hipStreamBeginCaptureToGraph(
                    stream, graph, &transformedMemsetNode, nullptr, 1, hipStreamCaptureModeGlobal));

                auto& forward = forwardPlans.at(branchIdx);
                hipfft_check(hipfftExecR2C(forward, expanded, transformed));

                hip_check(hipStreamEndCapture(stream, &graph));
                // [sphinx-hipfft-end]

                // [sphinx-is-leaf-start]
                auto is_leaf = [](hipGraphNode_t node)
                {
                    auto numDependentNodes = std::size_t{};
                    hip_check(hipGraphNodeGetDependentNodes(node, nullptr, &numDependentNodes));
                    return numDependentNodes == 0;
                };
                // [sphinx-is-leaf-end]

                // [sphinx-after-forward-start]
                // Obtain nodes in graph again, the new nodes will be our dependencies for the next step
                auto nodesAfterForward = std::vector<hipGraphNode_t>{};
                auto numNodesAfterForward = std::size_t{};
                hip_check(hipGraphGetNodes(graph, nullptr, &numNodesAfterForward));
                nodesAfterForward.resize(numNodesAfterForward);
                hip_check(hipGraphGetNodes(graph, nodesAfterForward.data(), &numNodesAfterForward));
                auto nodesAfterForwardSorted = std::set<hipGraphNode_t>{
                    std::begin(nodesAfterForward), std::end(nodesAfterForward)
                };
                // [sphinx-after-forward-end]

                // [sphinx-node-difference-start]
                // Compute difference between both sets
                auto forwardFFTNodes = std::vector<hipGraphNode_t>{};
                std::set_difference(std::begin(nodesAfterForwardSorted), std::end(nodesAfterForwardSorted),
                                    std::begin(nodesBeforeForwardSorted), std::end(nodesBeforeForwardSorted),
                                    std::back_inserter(forwardFFTNodes));
                // [sphinx-node-difference-end]

                // [sphinx-find-leaf-start]
                // Find leaf node in difference set
                auto forwardLeafNode = *(std::find_if(std::begin(forwardFFTNodes), std::end(forwardFFTNodes), is_leaf));
                // [sphinx-find-leaf-end]

                // Apply filter
                auto filterBlocksPerGrid = dim3{
                    (projGeom.dimTrans.x / threadsPerBlock.x) + 1,
                    (projGeom.dimTrans.y / threadsPerBlock.y) + 1,
                    1
                };
                void* filterApplicationKernelParams[] =
                {
                    static_cast<void*>(&transformed),
                    static_cast<void*>(&transformedPitch),
                    static_cast<void*>(&R),
                    static_cast<void*>(&projGeom.dimTrans)
                };
                auto filterApplicationKernelNodeParams = hipKernelNodeParams{};
                filterApplicationKernelNodeParams.blockDim = threadsPerBlock;
                filterApplicationKernelNodeParams.extra = nullptr;
                filterApplicationKernelNodeParams.func = reinterpret_cast<void*>(filter_application_kernel);
                filterApplicationKernelNodeParams.gridDim = filterBlocksPerGrid;
                filterApplicationKernelNodeParams.kernelParams = filterApplicationKernelParams;
                filterApplicationKernelNodeParams.sharedMemBytes = 0;
                auto filterApplicationKernelNode = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &filterApplicationKernelNode, graph, &forwardLeafNode, 1, &filterApplicationKernelNodeParams
                ));

                // C2R Fourier-transform projection - same node counting procedure as above
                auto nodesBeforeBackward = std::vector<hipGraphNode_t>{};
                auto numNodesBeforeBackward = std::size_t{};
                hip_check(hipGraphGetNodes(graph, nullptr, &numNodesBeforeBackward));
                nodesBeforeBackward.resize(numNodesBeforeBackward);
                hip_check(hipGraphGetNodes(graph, nodesBeforeBackward.data(), &numNodesBeforeBackward));
                auto nodesBeforeBackwardSorted = std::set<hipGraphNode_t>{
                    std::begin(nodesBeforeBackward), std::end(nodesBeforeBackward)
                };

                hip_check(hipStreamBeginCaptureToGraph(
                    stream, graph, &filterApplicationKernelNode, nullptr, 1, hipStreamCaptureModeGlobal
                ));

                auto& backward = backwardPlans.at(branchIdx);
                hipfft_check(hipfftExecC2R(backward, transformed, expanded));

                hip_check(hipStreamEndCapture(stream, &graph));

                auto nodesAfterBackward = std::vector<hipGraphNode_t>{};
                auto numNodesAfterBackward = std::size_t{};
                hip_check(hipGraphGetNodes(graph, nullptr, &numNodesAfterBackward));
                nodesAfterBackward.resize(numNodesAfterBackward);
                hip_check(hipGraphGetNodes(graph, nodesAfterBackward.data(), &numNodesAfterBackward));
                auto nodesAfterBackwardSorted = std::set<hipGraphNode_t>{
                    std::begin(nodesAfterBackward), std::end(nodesAfterBackward)
                };

                auto backwardFFTNodes = std::vector<hipGraphNode_t>{};
                std::set_difference(std::begin(nodesAfterBackwardSorted), std::end(nodesAfterBackwardSorted),
                                    std::begin(nodesBeforeBackwardSorted), std::end(nodesBeforeBackwardSorted),
                                    std::back_inserter(backwardFFTNodes));

                auto backwardLeafNode = *(
                    std::find_if(std::begin(backwardFFTNodes), std::end(backwardFFTNodes), is_leaf
                ));

                // Shrink projection to original size and normalize
                auto copyExpandedToProjNodeParams = hipMemcpy3DParms{};
                std::memset(&copyExpandedToProjNodeParams, 0, sizeof(hipMemcpy3DParms));
                copyExpandedToProjNodeParams.srcPos = make_hipPos(0, 0, 0);
                copyExpandedToProjNodeParams.srcPtr = make_hipPitchedPtr(
                    static_cast<void*>(expanded), expandedPitch, projGeom.dimFFT.x, projGeom.dimFFT.y
                );
                copyExpandedToProjNodeParams.dstPos = make_hipPos(0, 0, 0);
                copyExpandedToProjNodeParams.dstPtr = make_hipPitchedPtr(
                    static_cast<void*>(proj), projPitch, projGeom.dim.x, projGeom.dim.y
                );
                copyExpandedToProjNodeParams.extent = make_hipExtent(
                    projGeom.dim.x * sizeof(float), projGeom.dim.y, 1);
                copyExpandedToProjNodeParams.kind = hipMemcpyDeviceToDevice;
                auto copyExpandedToProjNode = hipGraphNode_t{};
                hip_check(hipGraphAddMemcpyNode(
                    &copyExpandedToProjNode, graph, &backwardLeafNode, 1, &copyExpandedToProjNodeParams
                ));

                void* filterNormalizationKernelParams[] =
                {
                    static_cast<void*>(&proj),
                    static_cast<void*>(&projPitch),
                    static_cast<void*>(&projGeom.dimFFT.x),
                    static_cast<void*>(&projGeom.dim)
                };
                auto filterNormalizationKernelNodeParams = hipKernelNodeParams{};
                filterNormalizationKernelNodeParams.blockDim = threadsPerBlock;
                filterNormalizationKernelNodeParams.extra = nullptr;
                filterNormalizationKernelNodeParams.func = reinterpret_cast<void*>(filter_normalization_kernel);
                filterNormalizationKernelNodeParams.gridDim = blocksPerGrid;
                filterNormalizationKernelNodeParams.kernelParams = filterNormalizationKernelParams;
                filterNormalizationKernelNodeParams.sharedMemBytes = 0;
                auto filterNormalizationKernelNode = hipGraphNode_t{};
                hip_check(hipGraphAddKernelNode(
                    &filterNormalizationKernelNode,
                    graph,
                    &copyExpandedToProjNode,
                    1,
                    &filterNormalizationKernelNodeParams));

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
                    auto& projTex = textureProjections.at(branchIdx);
                    void* backprojectionKernelParams[] =
                    {
                        &vol.ptr,
                        static_cast<void*>(&vol.pitch),
                        static_cast<void*>(&volGeom.dim),
                        static_cast<void*>(&volGeom.voxelDim),
                        static_cast<void*>(&projTex),
                        static_cast<void*>(&projGeom.minCoord),
                        static_cast<void*>(&sinTheta),
                        static_cast<void*>(&cosTheta),
                        static_cast<void*>(&projGeom.pixelDim),
                        static_cast<void*>(&projGeom.d_sd),
                        static_cast<void*>(&projGeom.d_so)
                    };
                    auto backprojectionKernelNodeParams = hipKernelNodeParams{};
                    backprojectionKernelNodeParams.blockDim = bpBlockSize;
                    backprojectionKernelNodeParams.extra = nullptr;
                    backprojectionKernelNodeParams.func = reinterpret_cast<void*>(backprojection_kernel);
                    backprojectionKernelNodeParams.gridDim = bpBlocks;
                    backprojectionKernelNodeParams.kernelParams = backprojectionKernelParams;
                    backprojectionKernelNodeParams.sharedMemBytes = 0;
                    auto backprojectionKernelNode = hipGraphNode_t{};
                    hip_check(hipGraphAddKernelNode(
                        &backprojectionKernelNode,
                        graph,
                        &filterNormalizationKernelNode,
                        1,
                        &backprojectionKernelNodeParams
                    ));
                }
                else
                {
                    // Fallback for devices without support for texture instructions
                    void* backprojectionKernelParams[] =
                    {
                        &vol.ptr,
                        static_cast<void*>(&vol.pitch),
                        static_cast<void*>(&volGeom.dim),
                        static_cast<void*>(&volGeom.voxelDim),
                        static_cast<void*>(&proj),
                        static_cast<void*>(&projPitch),
                        static_cast<void*>(&projGeom.dim),
                        static_cast<void*>(&projGeom.minCoord),
                        static_cast<void*>(&sinTheta),
                        static_cast<void*>(&cosTheta),
                        static_cast<void*>(&projGeom.pixelDim),
                        static_cast<void*>(&projGeom.d_sd),
                        static_cast<void*>(&projGeom.d_so)
                    };
                    auto backprojectionKernelNodeParams = hipKernelNodeParams{};
                    backprojectionKernelNodeParams.blockDim = bpBlockSize;
                    backprojectionKernelNodeParams.extra = nullptr;
                    backprojectionKernelNodeParams.func = reinterpret_cast<void*>(backprojection_kernel_no_tex);
                    backprojectionKernelNodeParams.gridDim = bpBlocks;
                    backprojectionKernelNodeParams.kernelParams = backprojectionKernelParams;
                    backprojectionKernelNodeParams.sharedMemBytes = 0;
                    auto backprojectionKernelNode = hipGraphNode_t{};
                    hip_check(hipGraphAddKernelNode(
                        &backprojectionKernelNode,
                        graph,
                        &filterNormalizationKernelNode,
                        1,
                        &backprojectionKernelNodeParams
                    ));
                }
            }

            // Instantiate and launch the graph
            if(batchSize == numBranches)
            {
                if(!graphCreated)
                {
                    hip_check(hipGraphDebugDotPrint(graph, "graph_creation.dot", hipGraphDebugDotFlagsVerbose));

                    hip_check(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
                    hip_check(hipGraphDestroy(graph));
                    hip_check(hipGraphLaunch(graphExec, graphStream));
                    graphCreated = true;
                }
                else
                {
                    // Update existing executable graph after each iteration with new input data
                    auto result = hipGraphExecUpdateResult{};
                    auto errorNode = hipGraphNode_t{};
                    hip_check(hipGraphExecUpdate(graphExec, graph, &errorNode, &result));
                    if(result != hipGraphExecUpdateSuccess)
                    {
                        auto msg = std::string{"Failed to update graph: "};
                        switch(result)
                        {
                        case hipGraphExecUpdateError:
                            msg += "Invalid value.";
                            break;
                        case hipGraphExecUpdateErrorFunctionChanged:
                            msg += "Function of kernel node changed.";
                            break;
                        case hipGraphExecUpdateErrorNodeTypeChanged:
                            msg += "Type of node changed.";
                            break;
                        case hipGraphExecUpdateErrorNotSupported:
                            msg += "Something about the node is not supported.";
                            break;
                        case hipGraphExecUpdateErrorParametersChanged:
                            msg += "Unsupported parameter change.";
                            break;
                        case hipGraphExecUpdateErrorTopologyChanged:
                            msg += "Graph topology changed.";
                            break;
                        case hipGraphExecUpdateErrorUnsupportedFunctionChange:
                            msg += "Unsupported change of kernel node function.";
                            break;
                        default:
                            msg += "Unknown error.";
                            break;
                        }
                        throw std::runtime_error{msg};
                    }
                    hip_check(hipGraphDestroy(graph));
                    hip_check(hipGraphLaunch(graphExec, graphStream));
                }
            }
            else
            {
                hip_check(hipGraphDebugDotPrint(graph, "graph_creation_final.dot", hipGraphDebugDotFlagsVerbose));

                // Incomplete batch: topology changed, must instantiate new executable graph
                hip_check(hipGraphInstantiate(&graphExecFinal, graph, nullptr, nullptr, 0));
                hip_check(hipGraphDestroy(graph));
                hip_check(hipGraphLaunch(graphExecFinal, graphStream));

                graphFinalCreated = true;
            }
        }

        // Obtain reconstruction time before copying back the result
        auto stop = std::chrono::steady_clock::time_point{};
        hip_check(hipLaunchHostFunc(graphStream, [](void* data)
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
        hip_check(hipMemcpy3DAsync(&memcpyParams, graphStream));
        
        auto saveVolArgs = new save_volume_args
        {
            "volume.tif",
            hostVolPtr,
            volGeom.dim.x, volGeom.dim.y, volGeom.dim.z,
            volGeom.voxelDim.x, volGeom.voxelDim.y
        };
        hip_check(hipLaunchHostFunc(graphStream, save_volume, saveVolArgs));

        std::cout << "All work items enqueued, waiting for completion... " << std::flush;
        hip_check(hipStreamSynchronize(graphStream));
        std::cout << "Done!" << std::endl;

        auto const elapsed = std::chrono::duration<double>{stop - start};
        std::cout << "Reconstruction time: " << elapsed.count() << 's' << std::endl;

        // Cleanup
        if(graphFinalCreated)
            hip_check(hipGraphExecDestroy(graphExecFinal));
        hip_check(hipGraphExecDestroy(graphExec));
        hip_check(hipStreamDestroy(graphStream));

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

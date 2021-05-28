/* Copyright (c) 2021-present Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "hip_graph_internal.hpp"
#include "platform/command.hpp"
#include "hip_conversions.hpp"
#include "hip_platform.hpp"
#include "hip_event.hpp"

thread_local std::vector<hipStream_t> g_captureStreams;
std::unordered_map<amd::Command*, hipGraphExec_t> hipGraphExec::activeGraphExec_;

hipError_t ihipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipKernelNodeParams* pNodeParams) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pNodeParams == nullptr) {
    return hipErrorInvalidValue;
  }
  hipFunction_t func = nullptr;
  hipError_t status =
      PlatformState::instance().getStatFunc(&func, pNodeParams->func, ihipGetDevice());
  if ((status != hipSuccess) || (func == nullptr)) {
    return hipErrorInvalidDeviceFunction;
  }
  size_t globalWorkSizeX = static_cast<size_t>(pNodeParams->gridDim.x) * pNodeParams->blockDim.x;
  size_t globalWorkSizeY = static_cast<size_t>(pNodeParams->gridDim.y) * pNodeParams->blockDim.y;
  size_t globalWorkSizeZ = static_cast<size_t>(pNodeParams->gridDim.z) * pNodeParams->blockDim.z;
  if (globalWorkSizeX > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeY > std::numeric_limits<uint32_t>::max() ||
      globalWorkSizeZ > std::numeric_limits<uint32_t>::max()) {
    return hipErrorInvalidConfiguration;
  }
  status = ihipLaunchKernel_validate(
      func, static_cast<uint32_t>(globalWorkSizeX), static_cast<uint32_t>(globalWorkSizeY),
      static_cast<uint32_t>(globalWorkSizeZ), pNodeParams->blockDim.x, pNodeParams->blockDim.y,
      pNodeParams->blockDim.z, pNodeParams->sharedMemBytes, pNodeParams->kernelParams,
      pNodeParams->extra, ihipGetDevice(), 0);
  if (status != hipSuccess) {
    return status;
  }
  *pGraphNode = new hipGraphKernelNode(pNodeParams, func);
  if (numDependencies == 0) {
    graph->AddNode(*pGraphNode);
  }
  for (size_t i = 0; i < numDependencies; i++) {
    if (graph->AddEdge(*(pDependencies + i), *pGraphNode) != hipSuccess) {
      return hipErrorInvalidValue;
    }
  }
  return hipSuccess;
}

hipError_t ihipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipMemcpy3DParms* pCopyParams) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pCopyParams == nullptr) {
    return hipErrorInvalidValue;
  }
  ihipMemcpy3D_validate(pCopyParams);
  *pGraphNode = new hipGraphMemcpyNode(pCopyParams);
  if (numDependencies == 0) {
    graph->AddNode(*pGraphNode);
  }
  for (size_t i = 0; i < numDependencies; i++) {
    if (graph->AddEdge(*(pDependencies + i), *pGraphNode) != hipSuccess) {
      return hipErrorInvalidValue;
    }
  }
  return hipSuccess;
}

hipError_t ihipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipMemsetParams* pMemsetParams) {
  if (pGraphNode == nullptr || graph == nullptr ||
      (numDependencies > 0 && pDependencies == nullptr) || pMemsetParams == nullptr) {
    return hipErrorInvalidValue;
  }
  if (pMemsetParams->height == 1) {
    ihipMemset_validate(pMemsetParams->dst, pMemsetParams->value, pMemsetParams->elementSize,
                        pMemsetParams->width * pMemsetParams->elementSize);
  } else {
    auto sizeBytes = pMemsetParams->width * pMemsetParams->height * 1;
    ihipMemset3D_validate(
        {pMemsetParams->dst, pMemsetParams->pitch, pMemsetParams->width, pMemsetParams->height},
        pMemsetParams->value, {pMemsetParams->width, pMemsetParams->height, 1}, sizeBytes);
  }

  *pGraphNode = new hipGraphMemsetNode(pMemsetParams);
  if (numDependencies == 0) {
    graph->AddNode(*pGraphNode);
  }
  for (size_t i = 0; i < numDependencies; i++) {
    if (graph->AddEdge(*(pDependencies + i), *pGraphNode) != hipSuccess) {
      return hipErrorInvalidValue;
    }
  }
  return hipSuccess;
}

hipError_t capturehipLaunchKernel(hipStream_t& stream, const void*& hostFunction, dim3& gridDim,
                                  dim3& blockDim, void**& args, size_t& sharedMemBytes) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node kernel launch on stream : %p", stream);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipKernelNodeParams nodeParams;
  nodeParams.func = const_cast<void*>(hostFunction);
  nodeParams.blockDim = blockDim;
  nodeParams.extra = nullptr;
  nodeParams.gridDim = gridDim;
  nodeParams.kernelParams = args;
  nodeParams.sharedMemBytes = sharedMemBytes;

  hipGraphNode_t pGraphNode;
  hipError_t status =
      ihipGraphAddKernelNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &nodeParams);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpy3DAsync(hipStream_t& stream, const hipMemcpy3DParms*& p) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memcpy3D on stream : %p",
          stream);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipError_t status =
      ihipGraphAddMemcpyNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), p);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyAsync(hipStream_t& stream, void*& dst, const void*& src,
                                 size_t& sizeBytes, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memcpy1D on stream : %p",
          stream);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipGraph_t graph = nullptr;
  std::vector<hipGraphNode_t> pDependencies = s->GetLastCapturedNodes();
  size_t numDependencies = s->GetLastCapturedNodes().size();
  graph = s->GetCaptureGraph();
  ihipMemcpy_validate(dst, src, sizeBytes, kind);
  pGraphNode = new hipGraphMemcpyNode1D(dst, src, sizeBytes, kind);
  if (numDependencies == 0) {
    graph->AddNode(pGraphNode);
  }
  for (size_t i = 0; i < numDependencies; i++) {
    if (graph->AddEdge(pDependencies[i], pGraphNode) != hipSuccess) {
      return hipErrorInvalidValue;
    }
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemcpyFromSymbolAsync(hipStream_t& stream, void*& dst, const void*& symbol,
                                           size_t& sizeBytes, size_t& offset, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node MemcpyFromSymbolNode on stream : %p", stream);
  return hipSuccess;
}

hipError_t capturehipMemcpyToSymbolAsync(hipStream_t& stream, const void*& symbol, const void*& src,
                                         size_t& sizeBytes, size_t& offset, hipMemcpyKind& kind) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node MemcpyToSymbolNode on stream : %p", stream);
  return hipSuccess;
}

hipError_t capturehipMemsetAsync(hipStream_t& stream, void*& dst, int& value, size_t& valueSize,
                                 size_t& sizeBytes) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memset1D on stream : %p",
          stream);

  hipMemsetParams memsetParams = {0};
  memsetParams.dst = dst;
  memsetParams.value = value;
  memsetParams.elementSize = valueSize;
  memsetParams.width = sizeBytes / valueSize;
  memsetParams.height = 1;

  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipError_t status =
      ihipGraphAddMemsetNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &memsetParams);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemset2DAsync(hipStream_t& stream, void*& dst, size_t& pitch, int& value,
                                   size_t& width, size_t& height) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memset2D on stream : %p",
          stream);
  hipMemsetParams memsetParams = {0};

  memsetParams.dst = dst;
  memsetParams.value = value;
  memsetParams.width = width;
  memsetParams.height = height;
  memsetParams.pitch = pitch;
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hipGraphNode_t pGraphNode;
  hipError_t status =
      ihipGraphAddMemsetNode(&pGraphNode, s->GetCaptureGraph(), s->GetLastCapturedNodes().data(),
                             s->GetLastCapturedNodes().size(), &memsetParams);
  if (status != hipSuccess) {
    return status;
  }
  s->SetLastCapturedNode(pGraphNode);
  return hipSuccess;
}

hipError_t capturehipMemset3DAsync(hipStream_t& stream, hipPitchedPtr& pitchedDevPtr, int& value,
                                   hipExtent& extent) {
  ClPrint(amd::LOG_INFO, amd::LOG_API, "[hipGraph] current capture node Memset3D on stream : %p",
          stream);
  return hipSuccess;
}

hipError_t capturehipEventRecord(hipStream_t& stream, hipEvent_t& event) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node EventRecord on stream : %p, Event %p", stream, event);
  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  e->StartCapture(stream);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  std::vector<hipGraphNode_t> lastCapturedNodes = s->GetLastCapturedNodes();
  if (!lastCapturedNodes.empty()) {
    e->SetNodesPrevToRecorded(lastCapturedNodes);
  }
  return hipSuccess;
}

hipError_t capturehipStreamWaitEvent(hipEvent_t& event, hipStream_t& stream, unsigned int& flags) {
  ClPrint(amd::LOG_INFO, amd::LOG_API,
          "[hipGraph] current capture node StreamWaitEvent on stream : %p, Event %p", stream,
          event);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::Event* e = reinterpret_cast<hip::Event*>(event);

  if (event == nullptr || stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (!s->IsOriginStream()) {
    s->SetCaptureGraph(reinterpret_cast<hip::Stream*>(e->GetCaptureStream())->GetCaptureGraph());
    s->SetCaptureMode(reinterpret_cast<hip::Stream*>(e->GetCaptureStream())->GetCaptureMode());
    s->SetParentStream(e->GetCaptureStream());
  }
  s->AddCrossCapturedNode(e->GetNodesPrevToRecorded());
  g_captureStreams.push_back(stream);
  return hipSuccess;
}

hipError_t hipStreamIsCapturing(hipStream_t stream, hipStreamCaptureStatus** pCaptureStatus) {
  HIP_INIT_API(hipStreamIsCapturing, stream, pCaptureStatus);
  if (stream == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  hipStreamCaptureStatus captureStatus = reinterpret_cast<hip::Stream*>(stream)->GetCaptureStatus();
  *pCaptureStatus = &captureStatus;
  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamBeginCapture(hipStream_t stream, hipStreamCaptureMode mode) {
  HIP_INIT_API(hipStreamBeginCapture, stream, mode);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  // capture cannot be initiated on legacy stream
  // It can be initiated if the stream is not already in capture mode
  if (stream == nullptr || s->GetCaptureStatus() == hipStreamCaptureStatusActive) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  s->SetCaptureGraph(new hipGraph());
  s->SetCaptureMode(mode);
  s->SetOriginStream();
  g_captureStreams.push_back(stream);
  HIP_RETURN_DURATION(hipSuccess);
}

hipError_t hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {
  HIP_INIT_API(hipStreamEndCapture, stream, pGraph);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  // Capture must be ended on the same stream in which it was initiated
  if (!s->IsOriginStream()) {
    HIP_RETURN(hipErrorStreamCaptureUnmatched);
  }
  // If mode is not hipStreamCaptureModeRelaxed, hipStreamEndCapture must be called on the stream
  // from the same thread
  if (s->GetCaptureMode() != hipStreamCaptureModeRelaxed &&
      std::find(g_captureStreams.begin(), g_captureStreams.end(), stream) ==
          g_captureStreams.end()) {
    HIP_RETURN(hipErrorStreamCaptureWrongThread);
  }
  // If capture was invalidated, due to a violation of the rules of stream capture
  if (s->GetCaptureStatus() == hipStreamCaptureStatusInvalidated) {
    *pGraph = nullptr;
    HIP_RETURN(hipErrorStreamCaptureInvalidated);
  }
  // check if all parallel streams have joined
  if (s->GetCaptureGraph()->GetLeafNodeCount() != 1) {
    return hipErrorStreamCaptureUnjoined;
  }
  *pGraph = s->GetCaptureGraph();
  // end capture on all streams/events part of graph capture
  HIP_RETURN_DURATION(s->EndCapture());
}

hipError_t hipGraphCreate(hipGraph_t* pGraph, unsigned int flags) {
  HIP_INIT_API(hipGraphCreate, pGraph, flags);
  *pGraph = new hipGraph();
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphDestroy(hipGraph_t graph) {
  HIP_INIT_API(hipGraphDestroy, graph);
  delete graph;
  HIP_RETURN(hipSuccess);
}

hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipKernelNodeParams* pNodeParams) {
  HIP_INIT_API(hipGraphAddKernelNode, pGraphNode, graph, pDependencies, numDependencies,
               pNodeParams);
  HIP_RETURN_DURATION(
      ihipGraphAddKernelNode(pGraphNode, graph, pDependencies, numDependencies, pNodeParams););
}

hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemcpy3DParms* pCopyParams) {
  HIP_INIT_API(hipGraphAddMemcpyNode, pGraphNode, graph, pDependencies, numDependencies,
               pCopyParams);

  HIP_RETURN_DURATION(
      ihipGraphAddMemcpyNode(pGraphNode, graph, pDependencies, numDependencies, pCopyParams););
}

hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies,
                                 const hipMemsetParams* pMemsetParams) {
  HIP_INIT_API(hipGraphAddMemsetNode, pGraphNode, graph, pDependencies, numDependencies,
               pMemsetParams);

  HIP_RETURN_DURATION(
      ihipGraphAddMemsetNode(pGraphNode, graph, pDependencies, numDependencies, pMemsetParams););
}

hipError_t ihipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                                hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  std::vector<std::vector<Node>> parallelLists;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists;
  graph->GetRunList(parallelLists, nodeWaitLists);
  std::vector<Node> levelOrder;
  graph->LevelOrder(levelOrder);
  *pGraphExec = new hipGraphExec(levelOrder, parallelLists, nodeWaitLists);
  if (*pGraphExec != nullptr) {
    return (*pGraphExec)->Init();
  } else {
    return hipErrorOutOfMemory;
  }
}

hipError_t hipGraphInstantiate(hipGraphExec_t* pGraphExec, hipGraph_t graph,
                               hipGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  HIP_INIT_API(hipGraphInstantiate, pGraphExec, graph);
  HIP_RETURN_DURATION(ihipGraphInstantiate(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize));
}

hipError_t hipGraphExecDestroy(hipGraphExec_t pGraphExec) {
  HIP_INIT_API(hipGraphExecDestroy, pGraphExec);
  delete pGraphExec;
  HIP_RETURN(hipSuccess);
}

hipError_t ihipGraphlaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  return graphExec->Run(stream);
}

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
  HIP_INIT_API(hipGraphLaunch, graphExec, stream);
  HIP_RETURN_DURATION(ihipGraphlaunch(graphExec, stream));
}

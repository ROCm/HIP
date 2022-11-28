# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Common Tests - Test independent of all platforms
set(TEST_SRC
  hipGraphAddEmptyNode.cc
  hipGraphAddDependencies.cc
  hipGraphAddEventRecordNode.cc
  hipGraphAddEventWaitNode.cc
  hipGraph.cc
  hipSimpleGraphWithKernel.cc
  hipGraphAddMemcpyNode.cc
  hipGraphClone.cc
  hipGraphInstantiateWithFlags.cc
  hipGraphAddHostNode.cc
  hipGraphAddMemcpyNodeFromSymbol.cc
  hipGraphChildGraphNodeGetGraph.cc
  hipGraphNodeFindInClone.cc
  hipGraphExecHostNodeSetParams.cc
  hipGraphAddMemcpyNodeToSymbol.cc
  hipGraphExecMemsetNodeSetParams.cc
  hipGraphMemcpyNodeSetParamsToSymbol.cc
  hipGraphDestroyNode.cc
  hipGraphGetNodes.cc
  hipGraphGetRootNodes.cc
  hipGraphHostNodeSetParams.cc
  hipGraphAddMemcpyNode1D.cc
  hipGraphAddChildGraphNode.cc
  hipGraphNodeGetType.cc
  hipGraphExecMemcpyNodeSetParams1D.cc
  hipGraphGetEdges.cc
  hipGraphRemoveDependencies.cc
  hipGraphInstantiate.cc
  hipGraphExecUpdate.cc
  hipGraphExecEventRecordNodeSetEvent.cc
  hipGraphEventWaitNodeSetEvent.cc
  hipGraphMemsetNodeGetParams.cc
  hipGraphMemsetNodeSetParams.cc
  hipGraphExecMemcpyNodeSetParamsFromSymbol.cc
  hipGraphEventRecordNodeGetEvent.cc
  hipGraphEventRecordNodeSetEvent.cc
  hipGraphEventWaitNodeGetEvent.cc
  hipGraphExecMemcpyNodeSetParams.cc
  hipStreamBeginCapture.cc
  hipGraphAddMemcpyNode1D.cc
  hipStreamIsCapturing.cc
  hipStreamGetCaptureInfo.cc
  hipStreamEndCapture.cc
  hipGraphMemcpyNodeSetParamsFromSymbol.cc
  hipGraphExecEventWaitNodeSetEvent.cc
  hipGraphAddMemsetNode.cc
  hipGraphAddKernelNode.cc
  hipGraphMemcpyNodeGetParams.cc
  hipGraphMemcpyNodeSetParams.cc
  hipGraphKernelNodeGetParams.cc
  hipGraphKernelNodeSetParams.cc
  hipGraphExecKernelNodeSetParams.cc
  hipGraphLaunch.cc
  hipGraphMemcpyNodeSetParams1D.cc
  hipGraphExecMemcpyNodeSetParamsToSymbol.cc
  hipGraphNodeGetDependentNodes.cc
  hipGraphNodeGetDependencies.cc
  hipGraphHostNodeGetParams.cc
  hipGraphExecChildGraphNodeSetParams.cc
  hipStreamGetCaptureInfo_v2.cc
  hipUserObjectCreate.cc
  hipGraphCreate.cc
  hipGraphDestroy.cc
  hipGraphExecDestroy.cc
)

hip_add_exe_to_target(NAME GraphsTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)

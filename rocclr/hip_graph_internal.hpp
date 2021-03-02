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

#pragma once
#include <algorithm>
#include <queue>
#include <stack>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "hip/hip_runtime.h"
#include "hip_internal.hpp"
#include "hip_graph_helper.hpp"

typedef hipGraphNode* Node;

class hipGraphNode {
 protected:
  uint32_t level_;
  hipGraphNodeType type_;
  std::vector<amd::Command*> commands_;
  bool visited_;

 public:
  hipGraphNode(hipGraphNodeType type) {
    type_ = type;
    level_ = 0;
    visited_ = false;
  }
  virtual ~hipGraphNode() {
    for (auto command : commands_) {
      delete command;
    }
  }
  virtual hipError_t CreateCommand(amd::HostQueue* queue) { return hipSuccess; }
  std::vector<amd::Command*>& GetCommands() { return commands_; }
  hipGraphNodeType GetType() { return type_; }
  uint32_t GetLevel() { return level_; }
  void SetLevel(uint32_t level) { level_ = level; }
};

class hipGraph {
  std::unordered_map<Node, size_t> nodeInDegree_;   // count of in coming edges for every vertex
  std::unordered_map<Node, size_t> nodeOutDegree_;  // count of outgoing edges for every vertex
  std::vector<Node> vertices_;
  std::unordered_map<Node, std::vector<Node>> edges_;

 public:
  hipGraph() {}
  ~hipGraph(){};
  /// add node to the graph
  hipError_t AddNode(const Node& node);
  /// add edge to the graph
  hipError_t AddEdge(const Node& parentNode, const Node& childNode);
  /// Returns root nodes, all vertices with 0 in-degrees
  std::vector<Node> GetRootNodes() const;
  /// Returns leaf nodes, all vertices with 0 out-degrees
  std::vector<Node> GetLeafNodes() const;
  /// Returns number of leaf nodes
  size_t GetLeafNodeCount() const;
  /// Returns total numbers of nodes in the graph
  size_t GetNodeCount() const { return vertices_.size(); }
  /// returns all the nodes in the graph
  std::vector<Node> GetNodes() const { return vertices_; }
  /// returns all the edges in the graph
  std::vector<std::pair<Node, Node>> GetEdges() const;
  void GetRunListUtil(Node v, std::unordered_map<Node, bool>& visited,
                      std::vector<Node>& singleList, std::vector<std::vector<Node>>& parallelList,
                      std::unordered_map<Node, std::vector<Node>>& dependencies);
  void GetRunList(std::vector<std::vector<Node>>& parallelList,
                  std::unordered_map<Node, std::vector<Node>>& dependencies);
  hipError_t LevelOrder(std::vector<Node>& levelOrder);
};

class hipGraphKernelNode : public hipGraphNode {
  hipKernelNodeParams* pKernelParams_;
  hipFunction_t func_;

 public:
  hipGraphKernelNode(const hipKernelNodeParams* pNodeParams, const hipFunction_t func)
      : hipGraphNode(hipGraphNodeTypeKernel) {
    pKernelParams_ = new hipKernelNodeParams(*pNodeParams);
    func_ = func;
  }
  ~hipGraphKernelNode() { delete pKernelParams_; }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    commands_.reserve(1);
    amd::Command* command;
    hipError_t status = ihipLaunchKernelCommand(
        command, func_, pKernelParams_->gridDim.x * pKernelParams_->blockDim.x,
        pKernelParams_->gridDim.y * pKernelParams_->blockDim.y,
        pKernelParams_->gridDim.z * pKernelParams_->blockDim.z, pKernelParams_->blockDim.x,
        pKernelParams_->blockDim.y, pKernelParams_->blockDim.z, pKernelParams_->sharedMemBytes,
        queue, pKernelParams_->kernelParams, pKernelParams_->extra, nullptr, nullptr, 0, 0, 0, 0, 0,
        0, 0);
    commands_.emplace_back(command);
    return status;
  }

  void GetParams(hipKernelNodeParams* params) {
    std::memcpy(params, pKernelParams_, sizeof(hipKernelNodeParams));
  }
  void SetParams(hipKernelNodeParams* params) {
    std::memcpy(pKernelParams_, params, sizeof(hipKernelNodeParams));
  }
};

class hipGraphMemcpyNode : public hipGraphNode {
  hipMemcpy3DParms* pCopyParams_;

 public:
  hipGraphMemcpyNode(const hipMemcpy3DParms* pCopyParams) : hipGraphNode(hipGraphNodeTypeMemcpy) {
    pCopyParams_ = new hipMemcpy3DParms(*pCopyParams);
  }
  ~hipGraphMemcpyNode() { delete pCopyParams_; }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    commands_.reserve(1);
    amd::Command* command;
    hipError_t status = ihipMemcpy3DCommand(command, pCopyParams_, queue);
    commands_.emplace_back(command);
    return status;
  }

  void GetParams(hipMemcpy3DParms* params) {
    std::memcpy(params, pCopyParams_, sizeof(hipMemcpy3DParms));
  }
  void SetParams(hipMemcpy3DParms* params) {
    std::memcpy(pCopyParams_, params, sizeof(hipMemcpy3DParms));
  }
};


class hipGraphMemcpyNode1D : public hipGraphNode {
  void* dst_;
  const void* src_;
  size_t count_;
  hipMemcpyKind kind_;

 public:
  hipGraphMemcpyNode1D(void* dst, const void* src, size_t count, hipMemcpyKind kind)
      : hipGraphNode(hipGraphNodeTypeMemcpy1D), dst_(dst), src_(src), count_(count), kind_(kind) {}
  ~hipGraphMemcpyNode1D() {}

  hipError_t CreateCommand(amd::HostQueue* queue) {
    commands_.reserve(1);
    amd::Command* command = nullptr;
    hipError_t status = ihipMemcpyCommand(command, dst_, src_, count_, kind_, *queue);
    commands_.emplace_back(command);
    return status;
  }

  void SetParams(void* dst, const void* src, size_t count, hipMemcpyKind kind) {
    dst_ = dst;
    src_ = src;
    count_ = count;
    kind_ = kind;
  }
};

template <class T> class hipGraphMemcpyNodeFromSymbol : public hipGraphNode {
  void* dst_;
  const T& symbol_;
  size_t count_;
  size_t offset_;
  hipMemcpyKind kind_;

 public:
  hipGraphMemcpyNodeFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
                               hipMemcpyKind kind)
      : hipGraphNode(hipGraphNodeTypeMemcpyFromSymbol),
        dst_(dst),
        symbol_(symbol),
        count_(count),
        offset_(offset),
        kind_(kind) {}
  ~hipGraphMemcpyNodeFromSymbol() {}

  hipError_t CreateCommand(amd::HostQueue* queue);

  void SetParams(void* dst, const void* symbol, size_t count, size_t offset, hipMemcpyKind kind) {
    dst_ = dst;
    symbol_ = symbol;
    count_ = count;
    offset_ = offset;
    kind_ = kind;
  }
};

template <class T> class hipGraphMemcpyNodeToSymbol : public hipGraphNode {
  const T& symbol_;
  const void* src_;
  size_t count_;
  size_t offset_;
  hipMemcpyKind kind_;

 public:
  hipGraphMemcpyNodeToSymbol(const T& symbol, void* src, size_t count, size_t offset,
                             hipMemcpyKind kind)
      : hipGraphNode(hipGraphNodeTypeMemcpyToSymbol),
        symbol_(symbol),
        src_(src),
        count_(count),
        offset_(offset),
        kind_(kind) {}
  ~hipGraphMemcpyNodeToSymbol() {}

  hipError_t CreateCommand(amd::HostQueue* queue);

  void SetParams(const T& symbol, void* src, size_t count, size_t offset, hipMemcpyKind kind) {
    symbol_ = symbol;
    src_ = src;
    count_ = count;
    offset_ = offset;
    kind_ = kind;
  }
};

class hipGraphMemsetNode : public hipGraphNode {
  hipMemsetParams* pMemsetParams_;

 public:
  hipGraphMemsetNode(const hipMemsetParams* pMemsetParams) : hipGraphNode(hipGraphNodeTypeMemset) {
    pMemsetParams_ = new hipMemsetParams(*pMemsetParams);
  }
  ~hipGraphMemsetNode() { delete pMemsetParams_; }

  hipError_t CreateCommand(amd::HostQueue* queue) {
    if (pMemsetParams_->height == 1) {
      return ihipMemsetCommand(commands_, pMemsetParams_->dst, pMemsetParams_->value,
                               pMemsetParams_->elementSize,
                               pMemsetParams_->width * pMemsetParams_->elementSize, queue);
    } else {
      return ihipMemset3DCommand(commands_,
                                 {pMemsetParams_->dst, pMemsetParams_->pitch, pMemsetParams_->width,
                                  pMemsetParams_->height},
                                 pMemsetParams_->elementSize,
                                 {pMemsetParams_->width, pMemsetParams_->height, 1}, queue);
    }
    return hipSuccess;
  }

  void GetParams(hipMemsetParams* params) {
    std::memcpy(params, pMemsetParams_, sizeof(hipMemsetParams));
  }
  void SetParams(hipMemsetParams* params) {
    std::memcpy(pMemsetParams_, params, sizeof(hipMemsetParams));
  }
};

class hipGraphEventRecordNode : public hipGraphNode {
  hipEvent_t event_;

 public:
  hipGraphEventRecordNode(hipEvent_t event)
      : hipGraphNode(hipGraphNodeTypeEventRecord), event_(event) {}
  ~hipGraphEventRecordNode() {}

  hipError_t CreateCommand(amd::HostQueue* queue);

  void GetParams(hipEvent_t* event) { *event = event_; }
  void SetParams(hipEvent_t event) { event_ = event; }
};

class hipGraphEventWaitNode : public hipGraphNode {
  hipEvent_t event_;

 public:
  hipGraphEventWaitNode(hipEvent_t event)
      : hipGraphNode(hipGraphNodeTypeWaitEvent), event_(event) {}
  ~hipGraphEventWaitNode() {}

  hipError_t CreateCommand(amd::HostQueue* queue);

  void GetParams(hipEvent_t* event) { *event = event_; }
  void SetParams(hipEvent_t event) { event_ = event; }
};

class hipGraphHostNode : public hipGraphNode {
  hipHostNodeParams* pNodeParams_;

 public:
  hipGraphHostNode(const hipHostNodeParams* pNodeParams) : hipGraphNode(hipGraphNodeTypeHost) {
    pNodeParams_ = new hipHostNodeParams(*pNodeParams);
  }
  ~hipGraphHostNode() { delete pNodeParams_; }

  hipError_t CreateCommand(amd::HostQueue* queue);

  void GetParams(hipHostNodeParams* params) {
    std::memcpy(params, pNodeParams_, sizeof(hipHostNodeParams));
  }
  void SetParams(hipHostNodeParams* params) {
    std::memcpy(pNodeParams_, params, sizeof(hipHostNodeParams));
  }
};

class hipGraphExec {
  std::vector<std::vector<Node>> parallelLists_;
  std::vector<Node> levelOrder_;
  std::unordered_map<Node, std::vector<Node>> nodeWaitLists_;
  std::vector<amd::HostQueue*> parallelQueues_;
  static std::unordered_map<amd::Command*, hipGraphExec_t> activeGraphExec_;
  amd::Command::EventWaitList graphLastCmdWaitList_;
  amd::Command* lastEnqueuedGraphCmd_;
  std::atomic<bool> bExecPending_;
  amd::Command* rootCommand_;

 public:
  hipGraphExec(std::vector<Node>& levelOrder, std::vector<std::vector<Node>>& lists,
               std::unordered_map<Node, std::vector<Node>>& nodeWaitLists)
      : parallelLists_(lists),
        levelOrder_(levelOrder),
        nodeWaitLists_(nodeWaitLists),
        lastEnqueuedGraphCmd_(nullptr),
        rootCommand_(nullptr) {
    bExecPending_.store(false);
  }

  ~hipGraphExec() {
    for (auto queue : parallelQueues_) {
      queue->release();
    }
    for (auto node : levelOrder_) {
      delete node;
    }
  }

  hipError_t CreateQueues();
  hipError_t FillCommands();
  hipError_t Init();
  hipError_t UpdateGraphToWaitOnRoot();
  hipError_t Run(hipStream_t stream);
  static void ResetGraph(cl_event event, cl_int command_exec_status, void* user_data);
};

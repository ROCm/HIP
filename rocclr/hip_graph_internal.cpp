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
#include <queue>

#define CASE_STRING(X, C)                                                                          \
  case X:                                                                                          \
    case_string = #C;                                                                              \
    break;
const char* GetGraphNodeTypeString(uint32_t op) {
  const char* case_string;
  switch (static_cast<hipGraphNodeType>(op)) {
    CASE_STRING(hipGraphNodeTypeKernel, KernelNode)
    CASE_STRING(hipGraphNodeTypeMemcpy, Memcpy3DNode)
    CASE_STRING(hipGraphNodeTypeMemset, MemsetNode)
    CASE_STRING(hipGraphNodeTypeHost, HostNode)
    CASE_STRING(hipGraphNodeTypeGraph, GraphNode)
    CASE_STRING(hipGraphNodeTypeEmpty, EmptyNode)
    CASE_STRING(hipGraphNodeTypeWaitEvent, WaitEventNode)
    CASE_STRING(hipGraphNodeTypeEventRecord, EventRecordNode)
    CASE_STRING(hipGraphNodeTypeMemcpy1D, Memcpy1DNode)
    CASE_STRING(hipGraphNodeTypeMemcpyFromSymbol, MemcpyFromSymbolNode)
    CASE_STRING(hipGraphNodeTypeMemcpyToSymbol, MemcpyToSymbolNode)
    default:
      case_string = "Unknown node type";
  };
  return case_string;
};

hipError_t hipGraph::AddNode(const Node& node) {
  vertices_.emplace_back(node);
  nodeOutDegree_[node] = 0;
  nodeInDegree_[node] = 0;
  node->SetLevel(0);
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Add %s(%p)\n",
          GetGraphNodeTypeString(node->GetType()), node);
  return hipSuccess;
}

hipError_t hipGraph::AddEdge(const Node& parentNode, const Node& childNode) {
  // if vertice doesn't exist, add it to the graph
  if (std::find(vertices_.begin(), vertices_.end(), parentNode) == vertices_.end()) {
    AddNode(parentNode);
  }
  if (std::find(vertices_.begin(), vertices_.end(), childNode) == vertices_.end()) {
    AddNode(childNode);
  }
  // Check if edge already exists
  auto connectedEdges = edges_.find(parentNode);
  if (connectedEdges != edges_.end()) {
    if (std::find(connectedEdges->second.begin(), connectedEdges->second.end(), childNode) !=
        connectedEdges->second.end()) {
      return hipSuccess;
    }
    connectedEdges->second.emplace_back(childNode);
  } else {
    edges_[parentNode] = {childNode};
  }
  nodeOutDegree_[parentNode]++;
  nodeInDegree_[childNode]++;
  childNode->SetLevel(std::max(childNode->GetLevel(), parentNode->GetLevel() + 1));
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Add edge btwn %s(%p) - %s(%p)\n",
          GetGraphNodeTypeString(parentNode->GetType()), parentNode,
          GetGraphNodeTypeString(childNode->GetType()), childNode);
  return hipSuccess;
}

// root nodes are all vertices with 0 in-degrees
std::vector<Node> hipGraph::GetRootNodes() const {
  std::vector<Node> roots;
  for (auto entry : vertices_) {
    if (nodeInDegree_.at(entry) == 0) {
      roots.push_back(entry);
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] root node: %s(%p)\n",
              GetGraphNodeTypeString(entry->GetType()), entry);
    }
  }
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "\n");
  return roots;
}

// leaf nodes are all vertices with 0 out-degrees
std::vector<Node> hipGraph::GetLeafNodes() const {
  std::vector<Node> leafNodes;
  for (auto entry : vertices_) {
    if (nodeOutDegree_.at(entry) == 0) {
      leafNodes.push_back(entry);
    }
  }
  return leafNodes;
}

size_t hipGraph::GetLeafNodeCount() const {
  int numLeafNodes = 0;
  for (auto entry : vertices_) {
    if (nodeOutDegree_.at(entry) == 0) {
      numLeafNodes++;
    }
  }
  return numLeafNodes;
}

std::vector<std::pair<Node, Node>> hipGraph::GetEdges() const {
  std::vector<std::pair<Node, Node>> edges;
  for (const auto& i : edges_) {
    for (const auto& j : i.second) {
      edges.push_back(std::make_pair(i.first, j));
    }
  }
  return edges;
}

void hipGraph::GetRunListUtil(Node v, std::unordered_map<Node, bool>& visited,
                              std::vector<Node>& singleList,
                              std::vector<std::vector<Node>>& parallelLists,
                              std::unordered_map<Node, std::vector<Node>>& dependencies) {
  // Mark the current node as visited.
  visited[v] = true;
  singleList.push_back(v);
  // Recurse for all the vertices adjacent to this vertex
  for (auto& adjNode : edges_[v]) {
    if (!visited[adjNode]) {
      // For the parallel list nodes add parent as the dependency
      if (singleList.empty()) {
        ClPrint(amd::LOG_INFO, amd::LOG_CODE,
                "[hipGraph] For %s(%p)- add parent as dependency %s(%p)\n",
                GetGraphNodeTypeString(adjNode->GetType()), adjNode,
                GetGraphNodeTypeString(v->GetType()), v);
        dependencies[adjNode].push_back(v);
      }
      GetRunListUtil(adjNode, visited, singleList, parallelLists, dependencies);
    } else {
      for (auto& list : parallelLists) {
        // Merge singleList when adjNode matches with the first element of the list in existing
        // lists
        if (adjNode == list[0]) {
          for (auto k = singleList.rbegin(); k != singleList.rend(); ++k) {
            list.insert(list.begin(), *k);
          }
          singleList.erase(singleList.begin(), singleList.end());
        }
      }
      // If the list cannot be merged with the existing list add as dependancy
      if (!singleList.empty()) {
        ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] For %s(%p)- add dependency %s(%p)\n",
                GetGraphNodeTypeString(adjNode->GetType()), adjNode,
                GetGraphNodeTypeString(v->GetType()), v);
        dependencies[adjNode].push_back(v);
      }
    }
  }
  if (!singleList.empty()) {
    parallelLists.push_back(singleList);
    singleList.erase(singleList.begin(), singleList.end());
  }
}
// The function to do Topological Sort.
// It uses recursive GetRunListUtil()
void hipGraph::GetRunList(std::vector<std::vector<Node>>& parallelLists,
                          std::unordered_map<Node, std::vector<Node>>& dependencies) {
  std::vector<Node> singleList;

  // Mark all the vertices as not visited
  std::unordered_map<Node, bool> visited;
  for (auto node : vertices_) visited[node] = false;

  // Call the recursive helper function for all vertices one by one
  for (auto node : vertices_) {
    if (visited[node] == false) {
      GetRunListUtil(node, visited, singleList, parallelLists, dependencies);
    }
  }
  for (size_t i = 0; i < parallelLists.size(); i++) {
    for (size_t j = 0; j < parallelLists[i].size(); j++) {
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] list %d - %s(%p)\n", i + 1,
              GetGraphNodeTypeString(parallelLists[i][j]->GetType()), parallelLists[i][j]);
    }
  }
}

hipError_t hipGraph::LevelOrder(std::vector<Node>& levelOrder) {
  std::vector<Node> roots = GetRootNodes();
  std::unordered_map<Node, bool> visited;
  std::queue<Node> q;
  for (auto it = roots.begin(); it != roots.end(); it++) {
    q.push(*it);
    ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] %s(%p) level:%d \n",
            GetGraphNodeTypeString((*it)->GetType()), *it, (*it)->GetLevel());
  }
  while (!q.empty()) {
    Node& node = q.front();
    q.pop();
    levelOrder.push_back(node);
    for (const auto& i : edges_[node]) {
      if (visited.find(i) == visited.end() && i->GetLevel() == (node->GetLevel() + 1)) {
        q.push(i);
        ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] %s(%p) level:%d \n",
                GetGraphNodeTypeString(i->GetType()), i, i->GetLevel());
        visited[i] = true;
      }
    }
  }
  return hipSuccess;
}

hipError_t hipGraphExec::CreateQueues() {
  parallelQueues_.reserve(parallelLists_.size());
  for (size_t i = 0; i < parallelLists_.size(); i++) {
    amd::HostQueue* queue;
    cl_command_queue_properties properties =
        (callbacks_table.is_enabled() || HIP_FORCE_QUEUE_PROFILING) ? CL_QUEUE_PROFILING_ENABLE : 0;
    queue = new amd::HostQueue(*hip::getCurrentDevice()->asContext(),
                               *hip::getCurrentDevice()->devices()[0], properties);

    bool result = (queue != nullptr) ? queue->create() : false;
    // Create a host queue
    if (result) {
      parallelQueues_.push_back(queue);
    } else {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to create host queue\n");
      return hipErrorOutOfMemory;
    }
  }
  return hipSuccess;
}

hipError_t hipGraphExec::FillCommands() {
  // Create commands
  int i = 0;
  hipError_t status;
  for (const auto& list : parallelLists_) {
    for (auto& node : list) {
      status = node->CreateCommand(parallelQueues_[i]);
      if (status != hipSuccess) return status;
    }
    i++;
  }
  // Add waitlists for all the commands
  for (auto& node : levelOrder_) {
    auto nodeWaitList = nodeWaitLists_.find(node);
    if (nodeWaitList != nodeWaitLists_.end()) {
      amd::Command::EventWaitList waitList;
      for (auto depNode : nodeWaitList->second) {
        for (auto command : depNode->GetCommands()) {
          waitList.push_back(command);
        }
      }
      for (auto command : nodeWaitList->first->GetCommands()) {
        command->updateEventWaitList(waitList);
      }
    }
  }
  return status;
}


hipError_t hipGraphExec::Init() {
  hipError_t status;
  status = CreateQueues();
  if (status != hipSuccess) {
    return status;
  }
  status = FillCommands();
  if (status != hipSuccess) {
    return status;
  }
  rootCommand_ = nullptr;
  /// stream should execute next command after graph finishes
  /// Add marker to the stream that waits for all the last commands in parallel queues of graph
  for (auto& singleList : parallelLists_) {
    graphLastCmdWaitList_.push_back(singleList.back()->GetCommands().back());
  }
  return status;
}

void hipGraphExec::ResetGraph(cl_event event, cl_int command_exec_status, void* user_data) {
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Inside resetGraph!\n");
  hipGraphExec_t graphExec =
      hipGraphExec::activeGraphExec_[reinterpret_cast<amd::Command*>(user_data)];
  if (graphExec != nullptr) {
    for (auto& node : graphExec->levelOrder_) {
      for (auto& command : node->GetCommands()) {
        command->resetStatus(CL_INT_MAX);
      }
    }
    graphExec->rootCommand_->resetStatus(CL_INT_MAX);
    graphExec->bExecPending_.store(false);
  } else {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] graphExec is nullptr during resetGraph!\n");
  }
}

hipError_t hipGraphExec::UpdateGraphToWaitOnRoot() {
  for (auto& singleList : parallelLists_) {
    amd::Command::EventWaitList waitList;
    waitList.push_back(rootCommand_);
    if (!singleList.empty()) {
      auto commands = singleList[0]->GetCommands();
      if (!commands.empty()) {
        commands[0]->updateEventWaitList(waitList);
      }
    }
  }
  return hipSuccess;
}

hipError_t hipGraphExec::Run(hipStream_t stream) {
  if (bExecPending_.load() == true) {
    ClPrint(
        amd::LOG_INFO, amd::LOG_CODE,
        "[hipGraph] Same graph launched while previous one is active, wait for it to finish!\n");
    lastEnqueuedGraphCmd_->awaitCompletion();
  }
  amd::HostQueue* queue = hip::getQueue(stream);
  if (queue == nullptr) {
    return hipErrorInvalidResourceHandle;
  }
  if (rootCommand_ == nullptr || rootCommand_->queue() != queue) {
    if (rootCommand_ != nullptr) {
      rootCommand_->release();
    }
    rootCommand_ = new amd::Marker(*queue, false, {});
    UpdateGraphToWaitOnRoot();
  }
  rootCommand_->enqueue();
  for (auto& node : levelOrder_) {
    for (auto& command : node->GetCommands()) {
      command->enqueue();
    }
  }

  amd::Command* command = new amd::Marker(*queue, false, graphLastCmdWaitList_);
  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }
  amd::Event& event = command->event();
  if (!event.setCallback(CL_COMPLETE, hipGraphExec::ResetGraph, command)) {
    return hipErrorInvalidHandle;
  }
  hipGraphExec::activeGraphExec_[command] = this;
  lastEnqueuedGraphCmd_ = command;
  bExecPending_.store(true);
  command->enqueue();
  command->release();
  return hipSuccess;
}
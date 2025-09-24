.. meta::
    :description: This chapter describes how to use HIP graphs and highlights their use cases.
    :keywords: ROCm, HIP, graph, stream

.. _how_to_HIP_graph:

********************************************************************************
HIP graphs
********************************************************************************

HIP graphs are an alternative way of executing tasks on a GPU that can provide
performance benefits over launching kernels using the standard
method via streams. A HIP graph is made up of nodes and edges. The nodes of a
HIP graph represent the operations performed, while the edges mark dependencies
between those operations.

The nodes can be one of the following:

- empty nodes
- nested graphs
- kernel launches
- host-side function calls
- HIP memory functions (copy, memset, ...)
- HIP events
- signalling or waiting on external semaphores

.. note::
    The available node types are specified by :cpp:enum:`hipGraphNodeType`.

The following figure visualizes the concept of graphs, compared to using streams.

.. figure:: ../../data/how-to/hip_runtime_api/hipgraph/hip_graph.svg
    :alt: Diagram depicting the difference between using streams to execute
          kernels with dependencies, resolved by explicitly synchronizing,
          or using graphs, where the edges denote the dependencies.

The standard method of launching kernels incurs a small overhead for each
iteration of the operation involved. That overhead is negligible, when the
kernel is launched directly with the HIP C/C++ API, but depending on the
framework used, there can be several levels of redirection, until the actual
kernel is launched by the HIP runtime, leading to significant overhead.
Especially for some AI frameworks, a GPU kernel might run faster than the time
it takes for the framework to set up and launch the kernel, and so the overhead
of repeatedly launching kernels can have a significant impact on performance.

HIP graphs are designed to address this issue, by predefining the HIP API calls
and their dependencies with a graph, and performing most of the initialization
beforehand. Launching a graph only requires a single call, after which the
HIP runtime takes care of executing the operations within the graph.
Graphs can provide additional performance benefits, by enabling optimizations
that are only possible when knowing the dependencies between the operations.

.. figure:: ../../data/how-to/hip_runtime_api/hipgraph/hip_graph_speedup.svg
    :alt: Diagram depicting the speed up achievable with HIP graphs compared to
          HIP streams when launching many short-running kernels.

    Qualitative presentation of the execution time of many short-running kernels
    when launched using HIP stream versus HIP graph. This does not include the
    time needed to set up the graph.

Using HIP graphs
================================================================================

There are two different ways of creating graphs: Capturing kernel launches from
a stream, or explicitly creating graphs. The difference between the two
approaches is explained later in this chapter.

The general flow for using HIP graphs includes the following steps.

#. Create a :cpp:type:`hipGraph_t` graph template using one of the two approaches described in this chapter
#. Create a :cpp:type:`hipGraphExec_t` executable instance of the graph template using :cpp:func:`hipGraphInstantiate`
#. Use :cpp:func:`hipGraphLaunch` to launch the executable graph to a stream
#. After execution completes free and destroy graph resources

The first two steps are the initial setup and only need to be executed once. First
step is the definition of the operations (nodes) and the dependencies (edges)
between them. The second step is the instantiation of the graph. This takes care
of validating and initializing the graph, to reduce the overhead when executing
the graph. The third step is the execution of the graph, which takes care of
launching all the kernels and executing the operations while respecting their
dependencies and necessary synchronizations as specified.

Because HIP graphs require some setup and initialization overhead before their
first execution, graphs only provide a benefit for workloads that require
many iterations to complete.

In both methods the :cpp:type:`hipGraph_t` template for a graph is used to define the graph.
In order to actually launch a graph, the template needs to be instantiated using
:cpp:func:`hipGraphInstantiate`, which results in an executable graph of type :cpp:type:`hipGraphExec_t`.
This executable graph can then be launched with :cpp:func:`hipGraphLaunch`, replaying the
operations within the graph. Note, that launching graphs is fundamentally no
different to executing other HIP functions on a stream, except for the fact,
that scheduling the operations within the graph encompasses less overhead and
can enable some optimizations, but they still need to be associated with a stream for execution.

Memory management
--------------------------------------------------------------------------------

Memory that is used by operations in graphs can either be pre-allocated or
managed within the graph. Graphs can contain nodes that take care of allocating
memory on the device or copying memory between the host and the device.
Whether you want to pre-allocate the memory or manage it within the graph
depends on the use-case. If the graph is executed in a tight loop the
performance is usually better when the memory is preallocated, so that it
does not need to be reallocated in every iteration.

The same rules as for normal memory allocations apply for memory allocated and
freed by nodes, meaning that the nodes that access memory allocated in a graph
must be ordered after allocation and before freeing.

Memory management within the graph enables the runtime to take care of memory reuse and optimizations.
The lifetime of memory managed in a graph begins when the execution reaches the
node allocating the memory, and ends when either reaching the corresponding
free node within the graph, or after graph execution when a corresponding
:cpp:func:`hipFreeAsync` or :cpp:func:`hipFree` call is reached.
The memory can also be freed with a free node in a different graph that is
associated with the same memory address.

Unlike device memory that is not associated with a graph, this does not necessarily
mean that the freed memory is returned back to the operating system immediately.
Graphs can retain a memory pool for quickly reusing memory within the graph.
This can be especially useful when memory is freed and reallocated later on
within a graph, as that memory doesn't have to be requested from the operating system.
It also potentially reduces the total memory footprint of the graph, by reusing the same memory.

The amount of memory allocated for graph memory pools on a specific device can
be queried using :cpp:func:`hipDeviceGetGraphMemAttribute`.
In order to return the freed memory :cpp:func:`hipDeviceGraphMemTrim` can be used.
This will return any memory that is not in active use by graphs.

These memory allocations can also be set up to allow access from multiple GPUs,
just like normal allocations. HIP then takes care of allocating and mapping the
memory to the GPUs. When capturing a graph from a stream, the node sets the
accessibility according to :cpp:func:`hipMemPoolSetAccess` at the time of capturing.


Capture graphs from a stream
================================================================================

The easy way to integrate HIP graphs into already existing code is to use
:cpp:func:`hipStreamBeginCapture` and :cpp:func:`hipStreamEndCapture` to obtain a :cpp:type:`hipGraph_t`
graph template that includes the captured operations.

When starting to capture operations for a graph using :cpp:func:`hipStreamBeginCapture`,
the operations assigned to the stream are captured into a graph instead of being
executed. The associated graph is returned when calling :cpp:func:`hipStreamEndCapture`, which
also stops capturing operations.
In order to capture to an already existing graph use :cpp:func:`hipStreamBeginCaptureToGraph`.

The functions assigned to the capturing stream are not executed, but instead are
captured and defined as nodes in the graph, to be run when the instantiated
graph is launched.

Functions must be associated with a stream in order to be captured.
This means that non-HIP API functions are not captured by default, but are
executed as standard functions when encountered and not added to the graph.
In order to assign host functions to a stream use
:cpp:func:`hipLaunchHostFunc`, as shown in the following code example.
They will then be captured and defined as a host node in the resulting graph,
and won't be executed when encountered.

Synchronous HIP API calls that are implicitly assigned to the default stream are
not permitted while capturing a stream  and will return an error. This is
because they implicitly synchronize and cause a dependency that can not be
captured within the stream. This includes functions like :cpp:func:`hipMalloc`,
:cpp:func:`hipMemcpy` and :cpp:func:`hipFree`. In order to capture these to the stream, replace
them with the corresponding asynchronous calls like :cpp:func:`hipMallocAsync`, :cpp:func:`hipMemcpyAsync` or :cpp:func:`hipFreeAsync`.

The general flow for using stream capture to create a graph template is:

#. Create a stream from which to capture the operations

#. Call :cpp:func:`hipStreamBeginCapture` before the first operation to be captured

#. Call :cpp:func:`hipStreamEndCapture` after the last operation to be captured

   #. Define a :cpp:type:`hipGraph_t` graph template to which :cpp:func:`hipStreamEndCapture`
      passes the captured graph

The following code is an example of how to use the HIP graph API to capture a
graph from a stream.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <vector>
    #include <iostream>

    #define HIP_CHECK(expression)                \
    {                                            \
        const hipError_t status = expression;    \
        if(status != hipSuccess){                \
                std::cerr << "HIP error "        \
                    << status << ": "            \
                    << hipGetErrorString(status) \
                    << " at " << __FILE__ << ":" \
                    << __LINE__ << std::endl;    \
        }                                        \
    }


    __global__ void kernelA(double* arrayA, size_t size){
        const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
        if(x < size){arrayA[x] *= 2.0;}
    };
    __global__ void kernelB(int* arrayB, size_t size){
        const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
        if(x < size){arrayB[x] = 3;}
    };
    __global__ void kernelC(double* arrayA, const int* arrayB, size_t size){
        const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
        if(x < size){arrayA[x] += arrayB[x];}
    };

    struct set_vector_args{
        std::vector<double>& h_array;
        double value;
    };

    void set_vector(void* args){
        set_vector_args h_args{*(reinterpret_cast<set_vector_args*>(args))};

        std::vector<double>& vec{h_args.h_array};
        vec.assign(vec.size(), h_args.value);
    }

    int main(){
        constexpr int numOfBlocks = 1024;
        constexpr int threadsPerBlock = 1024;
        constexpr size_t arraySize = 1U << 20;

        // This example assumes that kernelA operates on data that needs to be initialized on
        // and copied from the host, while kernelB initializes the array that is passed to it.
        // Both arrays are then used as input to kernelC, where arrayA is also used as
       //  output, that is copied back to the host, while arrayB is only read from and not modified.

        double* d_arrayA;
        int* d_arrayB;
        std::vector<double> h_array(arraySize);
        constexpr double initValue = 2.0;

        hipStream_t captureStream;
        HIP_CHECK(hipStreamCreate(&captureStream));

        // Start capturing the operations assigned to the stream
        HIP_CHECK(hipStreamBeginCapture(captureStream, hipStreamCaptureModeGlobal));

        // hipMallocAsync and hipMemcpyAsync are needed, to be able to assign it to a stream
        HIP_CHECK(hipMallocAsync(&d_arrayA, arraySize*sizeof(double), captureStream));
        HIP_CHECK(hipMallocAsync(&d_arrayB, arraySize*sizeof(int), captureStream));

        // Assign host function to the stream
        // Needs a custom struct to pass the arguments
        set_vector_args args{h_array, initValue};
        HIP_CHECK(hipLaunchHostFunc(captureStream, set_vector, &args));

        HIP_CHECK(hipMemcpyAsync(d_arrayA, h_array.data(), arraySize*sizeof(double), hipMemcpyHostToDevice, captureStream));

        kernelA<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, arraySize);
        kernelB<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayB, arraySize);
        kernelC<<<numOfBlocks, threadsPerBlock, 0, captureStream>>>(d_arrayA, d_arrayB, arraySize);

        HIP_CHECK(hipMemcpyAsync(h_array.data(), d_arrayA, arraySize*sizeof(*d_arrayA), hipMemcpyDeviceToHost, captureStream));

        HIP_CHECK(hipFreeAsync(d_arrayA, captureStream));
        HIP_CHECK(hipFreeAsync(d_arrayB, captureStream));

        // Stop capturing
        hipGraph_t graph;
        HIP_CHECK(hipStreamEndCapture(captureStream, &graph));

        // Create an executable graph from the captured graph
        hipGraphExec_t graphExec;
        HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // The graph template can be deleted after the instantiation if it's not needed for later use
        HIP_CHECK(hipGraphDestroy(graph));

        // Actually launch the graph. The stream does not have
        // to be the same as the one used for capturing.
        HIP_CHECK(hipGraphLaunch(graphExec, captureStream));

        // Verify results
        constexpr double expected = initValue * 2.0 + 3;
        bool passed = true;
        for(size_t i = 0; i < arraySize; ++i){
                if(h_array[i] != expected){
                        passed = false;
                        std::cerr << "Validation failed! Expected " << expected << " got " << h_array[0] << std::endl;
                        break;
                }
        }
        if(passed){
                std::cerr << "Validation passed." << std::endl;
        }

        // Free graph and stream resources after usage
        HIP_CHECK(hipGraphExecDestroy(graphExec));
        HIP_CHECK(hipStreamDestroy(captureStream));
    }

Explicit graph creation
================================================================================

Graphs can also be created directly using the HIP graph API, giving more
fine-grained control over the graph. In this case, the graph nodes are created
explicitly, together with their parameters and dependencies, which specify the
edges of the graph, thereby forming the graph structure.

The nodes are represented by the generic :cpp:type:`hipGraphNode_t` type. The actual
node type is implicitly defined by the specific function used to add the node to
the graph, for example :cpp:func:`hipGraphAddKernelNode` See the
:ref:`HIP graph API documentation<graph_management_reference>` for the
available functions, they are of type ``hipGraphAdd{Type}Node``. Each type of
node also has a predefined set of parameters depending on the operation, for
example :cpp:class:`hipKernelNodeParams` for a kernel launch. See the
:doc:`documentation for the general hipGraphNodeParams type<../../doxygen/html/structhip_graph_node_params>`
for a list of available parameter types and their members.

The general flow for explicitly creating a graph is usually:

#. Create a graph :cpp:type:`hipGraph_t`

#. Create the nodes and their parameters and add them to the graph

   #. Define a :cpp:type:`hipGraphNode_t`

   #. Define the parameter struct for the desired operation, by explicitly setting the appropriate struct's members.

   #. Use the appropriate ``hipGraphAdd{Type}Node`` function to add the node to the graph.

      #. The dependencies can be defined when adding the node to the graph, or afterwards by using :cpp:func:`hipGraphAddDependencies`

The following code example demonstrates how to explicitly create nodes in order to create a graph.

.. code-block:: cpp

    #include <hip/hip_runtime.h>
    #include <vector>
    #include <iostream>

    #define HIP_CHECK(expression)                \
    {                                            \
        const hipError_t status = expression;    \
        if(status != hipSuccess){                \
                std::cerr << "HIP error "        \
                    << status << ": "            \
                    << hipGetErrorString(status) \
                    << " at " << __FILE__ << ":" \
                    << __LINE__ << std::endl;    \
        }                                        \
    }

    __global__ void kernelA(double* arrayA, size_t size){
        const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
        if(x < size){arrayA[x] *= 2.0;}
    };
    __global__ void kernelB(int* arrayB, size_t size){
        const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
        if(x < size){arrayB[x] = 3;}
    };
    __global__ void kernelC(double* arrayA, const int* arrayB, size_t size){
        const size_t x = threadIdx.x + blockDim.x * blockIdx.x;
        if(x < size){arrayA[x] += arrayB[x];}
    };

    struct set_vector_args{
        std::vector<double>& h_array;
        double value;
    };

    void set_vector(void* args){
        set_vector_args h_args{*(reinterpret_cast<set_vector_args*>(args))};

        std::vector<double>& vec{h_args.h_array};
        vec.assign(vec.size(), h_args.value);
    }

    int main(){
        constexpr int numOfBlocks = 1024;
        constexpr int threadsPerBlock = 1024;
        size_t arraySize = 1U << 20;

        // The pointers to the device memory don't need to be declared here,
        // they are contained within the hipMemAllocNodeParams as the dptr member
        std::vector<double> h_array(arraySize);
        constexpr double initValue = 2.0;

        // Create graph an empty graph
        hipGraph_t graph;
        HIP_CHECK(hipGraphCreate(&graph, 0));

        // Parameters to allocate arrays
        hipMemAllocNodeParams allocArrayAParams{};
        allocArrayAParams.poolProps.allocType = hipMemAllocationTypePinned;
        allocArrayAParams.poolProps.location.type = hipMemLocationTypeDevice;
        allocArrayAParams.poolProps.location.id = 0; // GPU on which memory resides
        allocArrayAParams.bytesize = arraySize * sizeof(double);

        hipMemAllocNodeParams allocArrayBParams{};
        allocArrayBParams.poolProps.allocType = hipMemAllocationTypePinned;
        allocArrayBParams.poolProps.location.type = hipMemLocationTypeDevice;
        allocArrayBParams.poolProps.location.id = 0; // GPU on which memory resides
        allocArrayBParams.bytesize = arraySize * sizeof(int);

        // Add the allocation nodes to the graph. They don't have any dependencies
        hipGraphNode_t allocNodeA, allocNodeB;
        HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocArrayAParams));
        HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeB, graph, nullptr, 0, &allocArrayBParams));

        // Parameters for the host function
        // Needs custom struct to pass the arguments
        set_vector_args args{h_array, initValue};
        hipHostNodeParams hostParams{};
        hostParams.fn = set_vector;
        hostParams.userData = static_cast<void*>(&args);

        // Add the host node that initializes the host array. It also doesn't have any dependencies
        hipGraphNode_t hostNode;
        HIP_CHECK(hipGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams));

        // Add memory copy node, that copies the initialized host array to the device.
        // It has to wait for the host array to be initialized and the device memory to be allocated
        hipGraphNode_t cpyNodeDependencies[] = {allocNodeA, hostNode};
        hipGraphNode_t cpyToDevNode;
        HIP_CHECK(hipGraphAddMemcpyNode1D(&cpyToDevNode, graph, cpyNodeDependencies, 1, allocArrayAParams.dptr, h_array.data(), arraySize * sizeof(double), hipMemcpyHostToDevice));

        // Parameters for kernelA
        hipKernelNodeParams kernelAParams;
        void* kernelAArgs[] = {&allocArrayAParams.dptr, static_cast<void*>(&arraySize)};
        kernelAParams.func = reinterpret_cast<void*>(kernelA);
        kernelAParams.gridDim = numOfBlocks;
        kernelAParams.blockDim = threadsPerBlock;
        kernelAParams.sharedMemBytes = 0;
        kernelAParams.kernelParams = kernelAArgs;
        kernelAParams.extra = nullptr;

        // Add the node for kernelA. It has to wait for the memory copy to finish, as it depends on the values from the host array.
        hipGraphNode_t kernelANode;
        HIP_CHECK(hipGraphAddKernelNode(&kernelANode, graph, &cpyToDevNode, 1, &kernelAParams));

        // Parameters for kernelB
        hipKernelNodeParams kernelBParams;
        void* kernelBArgs[] = {&allocArrayBParams.dptr, static_cast<void*>(&arraySize)};
        kernelBParams.func = reinterpret_cast<void*>(kernelB);
        kernelBParams.gridDim = numOfBlocks;
        kernelBParams.blockDim = threadsPerBlock;
        kernelBParams.sharedMemBytes = 0;
        kernelBParams.kernelParams = kernelBArgs;
        kernelBParams.extra = nullptr;

        //  Add the node for kernelB. It only has to wait for the memory to be allocated, as it initializes the array.
        hipGraphNode_t kernelBNode;
        HIP_CHECK(hipGraphAddKernelNode(&kernelBNode, graph, &allocNodeB, 1, &kernelBParams));

        // Parameters for kernelC
        hipKernelNodeParams kernelCParams;
        void* kernelCArgs[] = {&allocArrayAParams.dptr, &allocArrayBParams.dptr, static_cast<void*>(&arraySize)};
        kernelCParams.func = reinterpret_cast<void*>(kernelC);
        kernelCParams.gridDim = numOfBlocks;
        kernelCParams.blockDim = threadsPerBlock;
        kernelCParams.sharedMemBytes = 0;
        kernelCParams.kernelParams = kernelCArgs;
        kernelCParams.extra = nullptr;

        // Add the node for kernelC. It has to wait on both kernelA and kernelB to finish, as it depends on their results.
        hipGraphNode_t kernelCNode;
        hipGraphNode_t kernelCDependencies[] = {kernelANode, kernelBNode};
        HIP_CHECK(hipGraphAddKernelNode(&kernelCNode, graph, kernelCDependencies, 1, &kernelCParams));

        // Copy the results back to the host. Has to wait for kernelC to finish.
        hipGraphNode_t cpyToHostNode;
        HIP_CHECK(hipGraphAddMemcpyNode1D(&cpyToHostNode, graph, &kernelCNode, 1, h_array.data(), allocArrayAParams.dptr, arraySize * sizeof(double), hipMemcpyDeviceToHost));

        // Free array of allocNodeA. It needs to wait for the copy to finish, as kernelC stores its results in it.
        hipGraphNode_t freeNodeA;
        HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph, &cpyToHostNode, 1, allocArrayAParams.dptr));
        // Free array of allocNodeB. It only needs to wait for kernelC to finish, as it is not written back to the host.
        hipGraphNode_t freeNodeB;
        HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeB, graph, &kernelCNode, 1, allocArrayBParams.dptr));

        // Instantiate the graph in order to execute it
        hipGraphExec_t graphExec;
        HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

        // The graph can be freed after the instantiation if it's not needed for other purposes
        HIP_CHECK(hipGraphDestroy(graph));

        // Actually launch the graph
        hipStream_t graphStream;
        HIP_CHECK(hipStreamCreate(&graphStream));
        HIP_CHECK(hipGraphLaunch(graphExec, graphStream));

        // Verify results
        constexpr double expected = initValue * 2.0 + 3;
        bool passed = true;
        for(size_t i = 0; i < arraySize; ++i){
                if(h_array[i] != expected){
                        passed = false;
                        std::cerr << "Validation failed! Expected " << expected << " got " << h_array[0] << std::endl;
                        break;
                }
        }
        if(passed){
                std::cerr << "Validation passed." << std::endl;
        }

        HIP_CHECK(hipGraphExecDestroy(graphExec));
        HIP_CHECK(hipStreamDestroy(graphStream));
    }

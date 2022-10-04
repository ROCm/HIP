/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#define GRIDSIZE 512
#define BLOCKSIZE 256
#define NUM_OF_STREAM 3
#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 1000

static __global__ void reduce(float* d_in, double* d_out) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_out[blockIdx.x] = d_in[myId];
  }
}
static __global__ void reduceFinal(double* d_in, double* d_out) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      d_in[myId] += d_in[myId + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *d_out = d_in[myId];
  }
}

static void init_input(float* a, size_t size) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < size; i++) {
    a[i] = (HipTest::RAND_R(&seed) & 0xFF) / static_cast<float>(RAND_MAX);
  }
}

static bool gPassed = true;
static void *gusrptr;
static void *ptr0xff = reinterpret_cast<void *>(0xffffffff);
static size_t NSize = GRIDSIZE * BLOCKSIZE;
static size_t Nbytes = NSize * sizeof(float);

typedef struct userDataStruct {
  float *A_h;
  float *C_h;
  float *A_d;
  float *C_d;
  bool isPassed;
  bool isOpCompleted;
} usrDataS;

// Common callback function.
static void Fn_validateSq(void* userData) {
  REQUIRE(userData != nullptr);
  usrDataS *ptrUsrData = reinterpret_cast<usrDataS *>(userData);
  for (size_t i = 0; i < NSize; i++) {
    if (ptrUsrData->C_h[i] !=
       (ptrUsrData->A_h[i] * ptrUsrData->A_h[i])) {
      ptrUsrData->isPassed = false;
      return;
    }
  }
  ptrUsrData->isPassed = true;
}

// Test scenario 1
// simple scenario that validates passing userData to host function.
static void Fn_ChkUserdataPtr(void* userData) {
  gPassed = true;
  if (gusrptr != userData) {
    gPassed = false;
  }
}

TEST_CASE("Unit_hipLaunchHostFunc_basic") {
  hipStream_t mystream;
  HIP_CHECK(hipStreamCreate(&mystream));
  gusrptr = ptr0xff;
  gPassed = true;
  HIP_CHECK(hipLaunchHostFunc(mystream, Fn_ChkUserdataPtr, gusrptr));
  HIP_CHECK(hipStreamSynchronize(mystream));
  HIP_CHECK(hipStreamDestroy(mystream));
  REQUIRE(gPassed);
}

// Negative test scenario for hipLaunchHostFunc
TEST_CASE("Unit_hipLaunchHostFunc_Negative") {
  hipStream_t mystream;
  HIP_CHECK(hipStreamCreate(&mystream));

  SECTION("Pass nullptr as function") {
    REQUIRE(hipLaunchHostFunc(mystream, nullptr, 0) == hipErrorInvalidValue);
  }
  HIP_CHECK(hipStreamDestroy(mystream));
}

// Local Function
static void launchOperationOnStrm(usrDataS *usrDataptr, hipStream_t stream) {
  usrDataptr->isPassed = false;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&(usrDataptr->A_d)),
                            Nbytes, stream));
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&(usrDataptr->C_d)),
                            Nbytes, stream));
  HIP_CHECK(hipMemcpyAsync(usrDataptr->A_d, usrDataptr->A_h, Nbytes,
                            hipMemcpyHostToDevice, stream));
  hipLaunchKernelGGL((HipTest::vector_square), dim3(GRIDSIZE),
                  dim3(BLOCKSIZE), 0, stream, usrDataptr->A_d,
                  usrDataptr->C_d, NSize);
  HIP_CHECK(hipMemcpyAsync(usrDataptr->C_h, usrDataptr->C_d, Nbytes,
                            hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipLaunchHostFunc(stream, Fn_validateSq,
                            reinterpret_cast<void*>(usrDataptr)));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(usrDataptr->A_d),
                           stream));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(usrDataptr->C_d),
                           stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(usrDataptr->isPassed);
}

// Test scenario 2
// scenario that validates the host launch function on 3 different streams,
// created stream, default/null stream and hipStreamPerThread.
TEST_CASE("Unit_hipLaunchHostFunc_streams") {
  hipStream_t stream[NUM_OF_STREAM];
  HIP_CHECK(hipStreamCreate(&stream[0]));
  stream[1] = 0;  // Null stream
  stream[2] = hipStreamPerThread;
  usrDataS *usrDataptr = reinterpret_cast<usrDataS *>(
                        malloc(sizeof(usrDataS)));
  REQUIRE(usrDataptr != nullptr);
  usrDataptr->A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->A_h != nullptr);
  usrDataptr->C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->C_h != nullptr);
  for (size_t i = 0; i < NSize; i++) {
    usrDataptr->A_h[i] = 21.0f;
  }
  for (int idx = 0; idx < NUM_OF_STREAM; idx++) {
    launchOperationOnStrm(usrDataptr, stream[idx]);
  }
  HIP_CHECK(hipStreamDestroy(stream[0]));
  free(usrDataptr->A_h);
  free(usrDataptr->C_h);
  free(usrDataptr);
}

// Test scenario 3
// test case to validate hipLaunchHostFunc for multi stream scenario.
// create 2 different streams and call hipLaunchHostFunc, stream synchronize.
static void Fn_validateMul_stream(void* userData) {
  REQUIRE(userData != nullptr);
  usrDataS *ptrUsrData = reinterpret_cast<usrDataS *>(userData);
  for (size_t i = 0; i < NSize; i++) {
    if (ptrUsrData->C_h[i] !=
    (ptrUsrData->A_h[i] * ptrUsrData->A_h[i])) {
      ptrUsrData->isPassed = false;
      return;
    }
  }
  ptrUsrData->isPassed = true;
}

TEST_CASE("Unit_hipLaunchHostFunc_multistreams") {
  hipStream_t mystream1, mystream2;
  HIP_CHECK(hipStreamCreateWithFlags(&mystream1, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&mystream2, hipStreamNonBlocking));
  usrDataS *usrDataptr1 = reinterpret_cast<usrDataS *>(
                        malloc(sizeof(usrDataS)));
  REQUIRE(usrDataptr1 != nullptr);
  usrDataS *usrDataptr2 = reinterpret_cast<usrDataS *>(
                        malloc(sizeof(usrDataS)));
  REQUIRE(usrDataptr2 != nullptr);
  usrDataptr1->A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr1->A_h != nullptr);
  usrDataptr1->C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr1->C_h != nullptr);
  // input data
  for (size_t i = 0; i < NSize; i++) {
    usrDataptr1->A_h[i] = 11.0f;
  }
  usrDataptr1->isPassed = false;
  usrDataptr2->isPassed = false;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&(usrDataptr1->A_d)),
                            Nbytes, mystream1));
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&(usrDataptr1->C_d)),
                            Nbytes, mystream1));
  HIP_CHECK(hipMemcpyAsync(usrDataptr1->A_d, usrDataptr1->A_h, Nbytes,
                            hipMemcpyHostToDevice, mystream1));
  const unsigned blocks = GRIDSIZE;
  const unsigned threadsPerBlock = BLOCKSIZE;
  hipLaunchKernelGGL((HipTest::vector_square), dim3(blocks),
                dim3(threadsPerBlock), 0, mystream1, usrDataptr1->A_d,
                usrDataptr1->C_d, NSize);
  HIP_CHECK(hipMemcpyAsync(usrDataptr1->C_h, usrDataptr1->C_d, Nbytes,
                            hipMemcpyDeviceToHost, mystream1));
  HIP_CHECK(hipLaunchHostFunc(mystream1, Fn_validateMul_stream,
            reinterpret_cast<void*>(usrDataptr1)));
  // launch kernel function for mystream2
  usrDataptr2->A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr2->A_h != nullptr);
  usrDataptr2->C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr2->C_h != nullptr);
  // input data
  for (size_t i = 0; i < NSize; i++) {
    usrDataptr2->A_h[i] = 9.0f;
  }
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&(usrDataptr2->A_d)),
                            Nbytes, mystream2));
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&(usrDataptr2->C_d)),
                            Nbytes, mystream2));
  HIP_CHECK(hipMemcpyAsync(usrDataptr2->A_d, usrDataptr2->A_h, Nbytes,
                            hipMemcpyHostToDevice, mystream2));
  hipLaunchKernelGGL((HipTest::vector_square), dim3(blocks),
                dim3(threadsPerBlock), 0, mystream2, usrDataptr2->A_d,
                usrDataptr2->C_d, NSize);
  HIP_CHECK(hipMemcpyAsync(usrDataptr2->C_h, usrDataptr2->C_d, Nbytes,
                        hipMemcpyDeviceToHost, mystream2));
  HIP_CHECK(hipLaunchHostFunc(mystream2, Fn_validateMul_stream,
            reinterpret_cast<void*>(usrDataptr2)));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(usrDataptr1->A_d),
                        mystream1));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(usrDataptr1->C_d),
                        mystream1));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(usrDataptr2->A_d),
                        mystream2));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(usrDataptr2->C_d),
                        mystream2));
  HIP_CHECK(hipStreamSynchronize(mystream1));
  HIP_CHECK(hipStreamSynchronize(mystream2));
  HIP_CHECK(hipStreamDestroy(mystream1));
  HIP_CHECK(hipStreamDestroy(mystream2));
  REQUIRE(usrDataptr1->isPassed);
  REQUIRE(usrDataptr2->isPassed);
  free(usrDataptr1->A_h);
  free(usrDataptr1->C_h);
  free(usrDataptr2->A_h);
  free(usrDataptr2->C_h);
  free(usrDataptr2);
  free(usrDataptr1);
}

// Test scenario 4
// test case to validate hipLaunchHostFunc for the kernel,
// validate hipLaunchHostFunc after kernel launch.
static void Fn_Completion_state(void* userData) {
  REQUIRE(userData != nullptr);
  usrDataS *ptrUsrData = reinterpret_cast<usrDataS *>(userData);
  ptrUsrData->isOpCompleted = true;
}

TEST_CASE("Unit_hipLaunchHostFunc_KernelHost") {
  hipStream_t stream1, stream2, stream3;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  usrDataS *usrDataptr = reinterpret_cast<usrDataS *>(
                        malloc(sizeof(usrDataS)));
  REQUIRE(usrDataptr != nullptr);
  usrDataptr->A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->A_h != nullptr);
  usrDataptr->C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->C_h != nullptr);
  // input data
  for (size_t i = 0; i < NSize; i++) {
    usrDataptr->A_h[i] = 7.0f;
  }
  usrDataptr->isOpCompleted = false;
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&(usrDataptr->A_d)),
                        Nbytes, stream1));
  HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&(usrDataptr->C_d)),
                        Nbytes, stream1));
  HIP_CHECK(hipMemcpyAsync(usrDataptr->A_d, usrDataptr->A_h, Nbytes,
                           hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipLaunchHostFunc(stream1, Fn_Completion_state,
                           reinterpret_cast<void*>(usrDataptr)));
  while (!usrDataptr->isOpCompleted) {
    std::this_thread::sleep_for(std::chrono::microseconds(100000));
  }  // Sleep for 100 ms*/
  usrDataptr->isOpCompleted = false;
  const unsigned blocks = GRIDSIZE;
  const unsigned threadsPerBlock = BLOCKSIZE;
  hipLaunchKernelGGL((HipTest::vector_square), dim3(blocks),
                dim3(threadsPerBlock), 0, stream2, usrDataptr->A_d,
                usrDataptr->C_d, NSize);
  HIP_CHECK(hipLaunchHostFunc(stream2, Fn_Completion_state,
                           reinterpret_cast<void*>(usrDataptr)));
  while (!usrDataptr->isOpCompleted) {
    std::this_thread::sleep_for(std::chrono::microseconds(100000));
  }  // Sleep for 100 ms*/
  usrDataptr->isOpCompleted = false;
  HIP_CHECK(hipMemcpyAsync(usrDataptr->C_h, usrDataptr->C_d, Nbytes,
            hipMemcpyDeviceToHost, stream3));
  HIP_CHECK(hipLaunchHostFunc(stream2, Fn_Completion_state,
                        reinterpret_cast<void*>(usrDataptr)));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(usrDataptr->A_d),
                        stream3));
  HIP_CHECK(hipFreeAsync(reinterpret_cast<void*>(usrDataptr->C_d),
                        stream3));
  while (!usrDataptr->isOpCompleted) {
    std::this_thread::sleep_for(std::chrono::microseconds(100000));
  }  // Sleep for 100 ms*/
  for (size_t i = 0; i < NSize; i++) {
    if (usrDataptr->C_h[i] !=
    (usrDataptr->A_h[i] * usrDataptr->A_h[i])) {
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream3));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
  free(usrDataptr->A_h);
  free(usrDataptr->C_h);
  free(usrDataptr);
}

// Test scenario 5
// scenario that validates the host launch function on multi device
// environment.
TEST_CASE("Unit_hipLaunchHostFunc_multidevice") {
  int num_devices;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  if (num_devices < 2) {
    SUCCEED("Skipping the testcases as numDevices < 2");
    return;
  }
  usrDataS *usrDataptr = reinterpret_cast<usrDataS *>(
                        malloc(sizeof(usrDataS)));
  REQUIRE(usrDataptr != nullptr);
  usrDataptr->A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->A_h != nullptr);
  usrDataptr->C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->C_h != nullptr);
  for (size_t i = 0; i < NSize; i++) {
    usrDataptr->A_h[i] = 21.0f;
  }
  for (int dev = 0; dev < num_devices; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    launchOperationOnStrm(usrDataptr, stream);
    HIP_CHECK(hipStreamDestroy(stream));
  }
  free(usrDataptr->A_h);
  free(usrDataptr->C_h);
  free(usrDataptr);
}

// Test scenario 6
// scenario that validates the host launch function on created
// stream with same priority.
TEST_CASE("Unit_hipLaunchHostFunc_Samepriority") {
  int priority = 0;
  unsigned int flags = 0;
  usrDataS *usrDataptr = reinterpret_cast<usrDataS *>(
                        malloc(sizeof(usrDataS)));
  REQUIRE(usrDataptr != nullptr);
  usrDataptr->A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->A_h != nullptr);
  usrDataptr->C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->C_h != nullptr);
  for (size_t i = 0; i < NSize; i++) {
    usrDataptr->A_h[i] = 21.0f;
  }
  for (int idx = 0; idx < NUM_OF_STREAM; idx++) {
    hipStream_t stream[NUM_OF_STREAM];
    HIP_CHECK(hipStreamCreateWithPriority(&stream[idx], flags, priority));
    launchOperationOnStrm(usrDataptr, stream[idx]);
    HIP_CHECK(hipStreamDestroy(stream[idx]));
  }
  free(usrDataptr->A_h);
  free(usrDataptr->C_h);
  free(usrDataptr);
}

// Test scenario 7
// scenario that validates the host launch function on
// created stream with different priority.
TEST_CASE("Unit_hipLaunchHostFunc_Diffpriority") {
  int priority;
  int priority_low{};
  int priority_high{};
  unsigned int flags = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
  int numOfPriorities = priority_low - priority_high;
  const float arr_size = numOfPriorities + 1;
  hipStream_t *stream = reinterpret_cast<hipStream_t*>(
                       malloc(arr_size*sizeof(hipStream_t)));
  stream[0] = 0;
  int count = 1;
  // Create a stream for each of the priority levels
  for (priority = priority_high; priority < priority_low; priority++) {
    HIP_CHECK(hipStreamCreateWithPriority(&stream[count++], flags, priority));
  }
  usrDataS *usrDataptr = reinterpret_cast<usrDataS *>(
                         malloc(sizeof(usrDataS)));
  REQUIRE(usrDataptr != nullptr);
  usrDataptr->A_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->A_h != nullptr);
  usrDataptr->C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(usrDataptr->C_h != nullptr);
  for (size_t i = 0; i < NSize; i++) {
    usrDataptr->A_h[i] = 11.0f;
  }
  for (int idx = 0; idx < arr_size; idx++) {
    launchOperationOnStrm(usrDataptr, stream[idx]);
  }
  count = 1;
  for (priority = priority_high; priority < priority_low; priority++) {
    HIP_CHECK(hipStreamDestroy(stream[count++]));
  }
  free(usrDataptr->A_h);
  free(usrDataptr->C_h);
  free(usrDataptr);
}

// Test scenario 8
// create a graph by using hipGraphsUsingStreamCapture and call host function.

typedef struct callBackData {
  const char* fn_name;
  double* data;
} callBackData_t;
double result_gpu = 0.0;
void myHostNodeCallback(void* data) {
  static int iter = 0;
  iter++;
  // Check status of GPU after stream operations are done
  callBackData_t* tmp = reinterpret_cast<callBackData_t*>(data);
  // checkCudaErrors(tmp->status);
  double* result = reinterpret_cast<double*>(tmp->data);
  const char* function = reinterpret_cast<const char*>(tmp->fn_name);
  if (iter == GRAPH_LAUNCH_ITERATIONS)
    printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
  result_gpu = *result;
  *result = 0.0;  // reset the result
}

TEST_CASE("Unit_hipLaunchHostFunc_Graph") {
  size_t size = 1 << 12;
  size_t maxBlocks = 512;
  float *inputVec_d = NULL, *inputVec_h = NULL;
  double *outputVec_d = NULL, *result_d;
  inputVec_h = reinterpret_cast<float*>(malloc(sizeof(float) * size));
  HIP_CHECK(hipMalloc(&inputVec_d, sizeof(float) * size));
  HIP_CHECK(hipMalloc(&outputVec_d, sizeof(double) * maxBlocks));
  HIP_CHECK(hipMalloc(&result_d, sizeof(double)));
  init_input(inputVec_h, size);
  hipStream_t stream1, stream2, stream3, streamForGraph;
  hipEvent_t forkStreamEvent, memsetEvent1, memsetEvent2;
  hipGraph_t graph;
  double result_h = 0.0;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipEventCreate(&forkStreamEvent));
  HIP_CHECK(hipEventCreate(&memsetEvent1));
  HIP_CHECK(hipEventCreate(&memsetEvent2));
  auto start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(forkStreamEvent, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, forkStreamEvent, 0));
  HIP_CHECK(hipStreamWaitEvent(stream3, forkStreamEvent, 0));
  HIP_CHECK(
         hipMemcpyAsync(inputVec_d, inputVec_h, sizeof(float) * size,
                        hipMemcpyDefault, stream1));
  HIP_CHECK(hipMemsetAsync(outputVec_d, 0, sizeof(double) * maxBlocks,
                               stream2));
  HIP_CHECK(hipEventRecord(memsetEvent1, stream2));
  HIP_CHECK(hipMemsetAsync(result_d, 0, sizeof(double), stream3));
  HIP_CHECK(hipEventRecord(memsetEvent2, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent1, 0));
  hipLaunchKernelGGL(reduce, dim3(size / THREADS_PER_BLOCK, 1, 1),
                     dim3(THREADS_PER_BLOCK, 1, 1), 0, stream1,
                     inputVec_d, outputVec_d);
  HIP_CHECK(hipStreamWaitEvent(stream1, memsetEvent2, 0));
  hipLaunchKernelGGL(reduceFinal, dim3(1, 1, 1), dim3(THREADS_PER_BLOCK, 1, 1),
                      0, stream1, outputVec_d, result_d);
  HIP_CHECK(hipMemcpyAsync(&result_h, result_d, sizeof(double),
                            hipMemcpyDefault, stream1));

  callBackData_t hostFnData;
  hostFnData.data = &result_h;
  hostFnData.fn_name = "hipGraphsUsingStreamCapture";
  hipHostFn_t fn = myHostNodeCallback;
  HIP_CHECK(hipLaunchHostFunc(stream1, fn, &hostFnData));

  HIP_CHECK(hipStreamEndCapture(stream1, &graph));
  hipGraphNode_t* nodes = NULL;
  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nodes, &numNodes));
  printf("\nNum of nodes in the graph created using stream"
                   "capture API = %zu\n", numNodes);
  HIP_CHECK(hipGraphGetRootNodes(graph, nodes, &numNodes));
  printf("root nodes in the graph created using stream capture API = %zu\n",
                             numNodes);
  hipGraphExec_t graphExec;

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  auto start1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  }
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  auto stop = std::chrono::high_resolution_clock::now();
  auto WithInit = std::chrono::duration<double, std::milli>(stop - start);
  auto WithoutInit = std::chrono::duration<double, std::milli>(stop - start1);
  std::cout << "Time taken for hipGraphsUsingStreamCapture with Init: "
  << std::chrono::duration_cast<std::chrono::milliseconds>(WithInit).count()
            << " milliseconds without Init:"
  << std::chrono::duration_cast<std::chrono::milliseconds>(WithoutInit).count()
            << " milliseconds " << std::endl;

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  double result_h_cpu = 0.0;
  for (size_t i = 0; i < size; i++) {
    result_h_cpu += inputVec_h[i];
  }
  REQUIRE(result_h_cpu == result_gpu);
  HIP_CHECK(hipFree(inputVec_d));
  HIP_CHECK(hipFree(outputVec_d));
  HIP_CHECK(hipFree(result_d));
}

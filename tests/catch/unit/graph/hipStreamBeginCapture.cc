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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios : Functional
 1) Initiate stream capture with different modes on custom stream.
 Capture stream sequence and replay the sequence in multiple iterations.
 2) End capture and validate that API returns captured graph for
 all possible modes on custom stream.
 3) Initiate stream capture with different modes on hipStreamPerThread.
 Capture stream sequence and replay the sequence in multiple iterations.
 4) End capture and validate that API returns captured graph for
 all possible modes on hipStreamPerThread.
 5) Waiting on an event recorded on a captured stream. Initiate capture
 on stream1, record an event on stream1, wait for the event on stream2,
 end the stream1 capture and Initiate stream capture on stream2
    5.1) Both streams are created with default flags.
    5.2) Both streams are created with flag = hipStreamCaptureModeGlobal.
    5.3) Both streams are created with different flags.
    5.4) Both streams are created with different priorities.
    5.5) Validate the number of nodes in both the captured graphs.
 6) Colligated Streams capture. Capture operation sequences queued in
 2 streams by overlapping the 2 captures.
    6.1) Both streams are created with default flags.
    6.2) Both streams are created with flag = hipStreamCaptureModeGlobal.
    6.3) Both streams are created with different flags.
    6.4) Both streams are created with different priorities.
 7) Extend the scenario 5.1 for 3 streamsss.
 8) Create 2 streams. Start capturing both stream1 and stream2 at the same
 time. On stream1 queue memcpy, kernel and memcpy operations and on stream2
 queue memcpy, kernel and memcpy operations. Execute both the captured
 graphs and validate the results.
 9) Capture 2 streams in parallel using threads. Execute the graphs in
 sequence in main thread and validate the results.
    9.1) mode = hipStreamCaptureModeGlobal
    9.2) mode = hipStreamCaptureModeThreadLocal
    9.3) mode = hipStreamCaptureModeRelaxed
 10) Queue operations (increment kernels) in 3 streams. Start capturing
  the streams after some operations have been queued. This scenario validates
  that only operations queued after hipStreamBeginCapture are captured in
  the graph.
 11) Detecting invalid capture. Create 2 streams s1 and s2. Start capturing
 s1. Create event dependency between s1 and s2 using event record and event
 wait. Try capturing s2. hipStreamBeginCapture must return error.
 12) Stream reuse. Capture multiple graphs from the same stream. Validate
 graphs are captured correctly.
 13) Test different synchronization during stream capture.
    13.1) Test hipStreamSynchronize. Must return
        hipErrorStreamCaptureUnsupported.
    13.2) Test hipDeviceSynchronize. Must return
        hipErrorStreamCaptureUnsupported.
    13.3) Test hipDeviceSynchronize. Must return
        hipEventSynchronize.
    13.4) Test hipStreamWaitEvent. Must return
        hipErrorStreamCaptureIsolation.
 14) End Stream Capture when the stream capture is still in progress.
    14.1) Abruptly end stream capture when stream capture is in progress in
        forked stream. hipStreamEndCapture must return
        hipErrorStreamCaptureUnjoined.
    14.2) Abruptly end stream capture when operations in forked stream
        are still waiting to be captured. hipStreamEndCapture must return
        hipErrorStreamCaptureUnjoined.
 15) Testing independent stream capture using multiple GPUs. Capture
 a stream in each device context and execute the captured graph in the
 context GPU.
 16) Test Nested Stream Capture Functionality: Create 3 streams s1, s2 & s3.
 Capture s1, record event e1 on s1, wait for event e1 on s2 and queue
 operations in s1. Record event e2 on s2 and wait for it on s3. Queue
 operations on both s2 and s3. Record event e4 on s3 and wait for it in s1.
 Record event e3 on s2 and wait for it in s1. End stream capture on s1.
 Execute the graph and verify the result.
 17) Forked Stream Reuse: In scenario 16, after end capture on s1, queue
 operations on both s2 and s3, and capture their graphs. Execute both the
 graphs and validate the functionality.
 18) Capture a complex graph containing multiple independent memcpy, kernel
 and host nodes. Launch the graph on random input data and validate the
 output.
 19) Capture empty streams (parent + forked streams) and validate the
 functionality.
*/

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#define INCREMENT_KERNEL_FINALEXP_VAL 7
constexpr size_t N = 1000000;
constexpr int LAUNCH_ITERS = 50;
static int gCbackIter = 0;
#define GRIDSIZE 256
#define BLOCKSIZE 256
#define CONST_KER1_VAL 3
#define CONST_KER2_VAL 2
#define CONST_KER3_VAL 5

static __global__ void dummyKernel() {
  return;
}

static __global__ void incrementKernel(int *data) {
  atomicAdd(data, 1);
  return;
}

static __global__ void myadd(int* A_d, int* B_d) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  A_d[myId] = A_d[myId] + B_d[myId];
}

static __global__ void mymul(int* devMem, int value) {
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  devMem[myId] = devMem[myId] * value;
}

static void hostNodeCallback(void* data) {
  REQUIRE(data == nullptr);
  gCbackIter++;
}

bool CaptureStreamAndLaunchGraph(float *A_d, float *C_d, float *A_h,
                 float *C_h, hipStreamCaptureMode mode, hipStream_t stream) {
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  constexpr unsigned blocks = 512;
  constexpr unsigned threadsPerBlock = 256;
  size_t Nbytes = N * sizeof(float);

  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  HIP_CHECK(hipMemcpyAsync(A_d, A_h, Nbytes, hipMemcpyHostToDevice, stream));

  HIP_CHECK(hipMemsetAsync(C_d, 0, Nbytes, stream));
  hipLaunchKernelGGL(HipTest::vector_square, dim3(blocks),
                              dim3(threadsPerBlock), 0, stream, A_d, C_d, N);
  HIP_CHECK(hipMemcpyAsync(C_h, C_d, Nbytes, hipMemcpyDeviceToHost, stream));

  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  // Validate end capture is successful
  REQUIRE(graph != nullptr);

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  REQUIRE(graphExec != nullptr);

  // Replay the recorded sequence multiple times
  for (int i = 0; i < LAUNCH_ITERS; i++) {
    HIP_CHECK(hipGraphLaunch(graphExec, stream));
  }

  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      UNSCOPED_INFO("A and C not matching at " << i);
      return false;
    }
  }
  return true;
}

/**
 * Basic Functional Test for API capturing custom stream and replaying sequence.
 * Test exercises the API on available/possible modes.
 * Stream capture with different modes behave the same when supported/
 * safe apis are used in sequence.
 */
TEST_CASE("Unit_hipStreamBeginCapture_BasicFunctional") {
  float *A_d, *C_d;
  float *A_h, *C_h;
  size_t Nbytes = N * sizeof(float);
  hipStream_t stream;
  bool ret;

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
      A_h[i] = 1.618f + i;
  }

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  SECTION("Capture stream and launch graph when mode is global") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                          hipStreamCaptureModeGlobal, stream);
    REQUIRE(ret == true);
  }

  SECTION("Capture stream and launch graph when mode is local") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                     hipStreamCaptureModeThreadLocal, stream);
    REQUIRE(ret == true);
  }

  SECTION("Capture stream and launch graph when mode is relaxed") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                         hipStreamCaptureModeRelaxed, stream);
    REQUIRE(ret == true);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}

/**
 * Perform capture on hipStreamPerThread, launch the graph and verify results.
 */
TEST_CASE("Unit_hipStreamBeginCapture_hipStreamPerThread") {
  float *A_d, *C_d;
  float *A_h, *C_h;
  size_t Nbytes = N * sizeof(float);
  hipStream_t stream{hipStreamPerThread};
  bool ret;

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  REQUIRE(A_h != nullptr);
  REQUIRE(C_h != nullptr);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
      A_h[i] = 1.618f + i;
  }

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  REQUIRE(A_d != nullptr);
  REQUIRE(C_d != nullptr);

  SECTION("Capture hipStreamPerThread and launch graph when mode is global") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                           hipStreamCaptureModeGlobal, stream);
    REQUIRE(ret == true);
  }

  SECTION("Capture hipStreamPerThread and launch graph when mode is local") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                      hipStreamCaptureModeThreadLocal, stream);
    REQUIRE(ret == true);
  }

  SECTION("Capture hipStreamPerThread and launch graph when mode is relaxed") {
    ret = CaptureStreamAndLaunchGraph(A_d, C_d, A_h, C_h,
                                          hipStreamCaptureModeRelaxed, stream);
    REQUIRE(ret == true);
  }

  free(A_h);
  free(C_h);
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
}


/* Test verifies hipStreamBeginCapture API Negative scenarios.
 */

TEST_CASE("Unit_hipStreamBeginCapture_Negative") {
  hipError_t ret;
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));

  SECTION("Stream capture on legacy/null stream returns error code.") {
    ret = hipStreamBeginCapture(nullptr, hipStreamCaptureModeGlobal);
    REQUIRE(hipErrorStreamCaptureUnsupported == ret);
  }
  SECTION("Capturing hipStream status with same stream again") {
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    ret = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    REQUIRE(hipErrorIllegalState == ret);
  }
  SECTION("Creating hipStream with invalid mode") {
    ret = hipStreamBeginCapture(stream, hipStreamCaptureMode(-1));
    REQUIRE(hipErrorInvalidValue == ret);
  }
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipStreamBeginCapture_Basic") {
  hipStream_t s1, s2, s3;

  HIP_CHECK(hipStreamCreate(&s1));
  HIP_CHECK(hipStreamBeginCapture(s1, hipStreamCaptureModeGlobal));

  HIP_CHECK(hipStreamCreate(&s2));
  HIP_CHECK(hipStreamBeginCapture(s2, hipStreamCaptureModeThreadLocal));

  HIP_CHECK(hipStreamCreate(&s3));
  HIP_CHECK(hipStreamBeginCapture(s3, hipStreamCaptureModeRelaxed));

  HIP_CHECK(hipStreamDestroy(s1));
  HIP_CHECK(hipStreamDestroy(s2));
  HIP_CHECK(hipStreamDestroy(s3));
}
/* Local Function
 */
static void interStrmEventSyncCapture(const hipStream_t &stream1,
                                      const hipStream_t &stream2) {
  hipGraph_t graph1, graph2;
  hipEvent_t event;
  hipGraphExec_t graphExec1{nullptr}, graphExec2{nullptr};
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(event, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, event, 0));
  dummyKernel<<<1, 1, 0, stream1>>>();
  HIP_CHECK(hipStreamEndCapture(stream1, &graph1));
  HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeGlobal));
  dummyKernel<<<1, 1, 0, stream2>>>();
  dummyKernel<<<1, 1, 0, stream2>>>();
  HIP_CHECK(hipStreamEndCapture(stream2, &graph2));
  // Create Executable Graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  REQUIRE(graphExec1 != nullptr);
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  REQUIRE(graphExec2 != nullptr);
  size_t numNodes1 = 0, numNodes2 = 0;
  HIP_CHECK(hipGraphGetNodes(graph1, nullptr, &numNodes1));
  HIP_CHECK(hipGraphGetNodes(graph2, nullptr, &numNodes2));
  REQUIRE(numNodes1 == 1);
  REQUIRE(numNodes2 == 2);
  // Execute the Graphs
  HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
  HIP_CHECK(hipGraphLaunch(graphExec2, stream2));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamSynchronize(stream2));
  // Free
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipEventDestroy(event));
}
/* Local Function
 */
static void colligatedStrmCapture(const hipStream_t &stream1,
                                const hipStream_t &stream2) {
  hipGraph_t graph1, graph2;
  hipEvent_t event;
  hipGraphExec_t graphExec1{nullptr}, graphExec2{nullptr};
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(event, stream1));
  HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamWaitEvent(stream1, event, 0));
  dummyKernel<<<1, 1, 0, stream1>>>();
  HIP_CHECK(hipStreamEndCapture(stream1, &graph1));
  dummyKernel<<<1, 1, 0, stream2>>>();
  HIP_CHECK(hipStreamEndCapture(stream2, &graph2));
  // Validate end capture is successful
  REQUIRE(graph2 != nullptr);
  REQUIRE(graph1 != nullptr);
  // Create Executable Graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  REQUIRE(graphExec1 != nullptr);
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  REQUIRE(graphExec2 != nullptr);
  // Execute the Graphs
  HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
  HIP_CHECK(hipGraphLaunch(graphExec2, stream2));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipStreamSynchronize(stream2));
  // Free
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipEventDestroy(event));
}
/* Fill input Data
 */
static void init_input(int* a, size_t size) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < size; i++) {
    a[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
}
/* Validate Output
 */
static void validate_output(int* a, int *b, size_t size) {
  for (size_t i = 0; i < size; i++) {
    REQUIRE(a[i] == (b[i]*b[i]));
  }
}
/* Local Function
 */
static void colligatedStrmCaptureFunc(const hipStream_t &stream1,
                                const hipStream_t &stream2) {
  constexpr size_t size = 1024;
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU,
                            threadsPerBlock, size);
  hipGraph_t graph1, graph2;
  int *inputVec_d1{nullptr}, *inputVec_h1{nullptr}, *outputVec_h1{nullptr},
      *outputVec_d1{nullptr};
  int *inputVec_d2{nullptr}, *inputVec_h2{nullptr}, *outputVec_h2{nullptr},
      *outputVec_d2{nullptr};
  hipGraphExec_t graphExec1{nullptr}, graphExec2{nullptr};
  // host and device allocation
  HipTest::initArrays<int>(&inputVec_d1, &outputVec_d1, nullptr,
               &inputVec_h1, &outputVec_h1, nullptr, size, false);
  HipTest::initArrays<int>(&inputVec_d2, &outputVec_d2, nullptr,
               &inputVec_h2, &outputVec_h2, nullptr, size, false);
  // Capture 2 streams
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemcpyAsync(inputVec_d1, inputVec_h1, sizeof(int) * size,
                                            hipMemcpyDefault, stream1));
  HIP_CHECK(hipMemcpyAsync(inputVec_d2, inputVec_h2, sizeof(int) * size,
                                            hipMemcpyDefault, stream2));
  HipTest::vector_square<int><<<blocks, threadsPerBlock, 0, stream1>>>(
                                        inputVec_d1, outputVec_d1, size);
  HipTest::vector_square<int><<<blocks, threadsPerBlock, 0, stream2>>>(
                                        inputVec_d2, outputVec_d2, size);
  HIP_CHECK(hipMemcpyAsync(outputVec_h1, outputVec_d1, sizeof(int) * size,
                                            hipMemcpyDefault, stream1));
  HIP_CHECK(hipMemcpyAsync(outputVec_h2, outputVec_d2, sizeof(int) * size,
                                            hipMemcpyDefault, stream2));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph1));
  HIP_CHECK(hipStreamEndCapture(stream2, &graph2));
  // Validate end capture is successful
  REQUIRE(graph2 != nullptr);
  REQUIRE(graph1 != nullptr);
  // Create Executable Graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  REQUIRE(graphExec1 != nullptr);
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  REQUIRE(graphExec2 != nullptr);
  // Execute the Graphs
  for (int iter = 0; iter < LAUNCH_ITERS; iter++) {
    init_input(inputVec_h1, size);
    init_input(inputVec_h2, size);
    HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
    HIP_CHECK(hipGraphLaunch(graphExec2, stream2));
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamSynchronize(stream2));
    validate_output(outputVec_h1, inputVec_h1, size);
    validate_output(outputVec_h2, inputVec_h2, size);
  }
  // Free
  HipTest::freeArrays<int>(inputVec_d1, outputVec_d1, nullptr,
                   inputVec_h1, outputVec_h1, nullptr, false);
  HipTest::freeArrays<int>(inputVec_d2, outputVec_d2, nullptr,
                   inputVec_h2, outputVec_h2, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
}
/* Stream Capture thread function
 */
static void threadStrmCaptureFunc(hipStream_t stream, int *inputVec_d,
int *outputVec_d, int *inputVec_h, int *outputVec_h, hipGraph_t *graph,
size_t size, hipStreamCaptureMode mode) {
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU,
                            threadsPerBlock, size);
  // Capture stream
  HIP_CHECK(hipStreamBeginCapture(stream, mode));
  HIP_CHECK(hipMemcpyAsync(inputVec_d, inputVec_h, sizeof(int) * size,
                                            hipMemcpyDefault, stream));
  HipTest::vector_square<int><<<blocks, threadsPerBlock, 0, stream>>>(
                                        inputVec_d, outputVec_d, size);
  HIP_CHECK(hipMemcpyAsync(outputVec_h, outputVec_d, sizeof(int) * size,
                                            hipMemcpyDefault, stream));
  HIP_CHECK(hipStreamEndCapture(stream, graph));
}
/* Local Function for multithreaded tests
 */
static void multithreadedTest(hipStreamCaptureMode mode) {
  hipStream_t stream1, stream2;
  constexpr size_t size = 1024;
  hipGraph_t graph1, graph2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  int *inputVec_d1{nullptr}, *inputVec_h1{nullptr}, *outputVec_h1{nullptr},
      *outputVec_d1{nullptr};
  int *inputVec_d2{nullptr}, *inputVec_h2{nullptr}, *outputVec_h2{nullptr},
      *outputVec_d2{nullptr};
  hipGraphExec_t graphExec1{nullptr}, graphExec2{nullptr};
  // host and device allocation
  HipTest::initArrays<int>(&inputVec_d1, &outputVec_d1, nullptr,
               &inputVec_h1, &outputVec_h1, nullptr, size, false);
  HipTest::initArrays<int>(&inputVec_d2, &outputVec_d2, nullptr,
               &inputVec_h2, &outputVec_h2, nullptr, size, false);
  // Launch 2 threads to capture the 2 streams into graphs
  std::thread t1(threadStrmCaptureFunc, stream1, inputVec_d1,
  outputVec_d1, inputVec_h1, outputVec_h1, &graph1, size, mode);
  std::thread t2(threadStrmCaptureFunc, stream2, inputVec_d2,
  outputVec_d2, inputVec_h2, outputVec_h2, &graph2, size, mode);
  t1.join();
  t2.join();
  // Create Executable Graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  // Execute the Graphs
  for (int iter = 0; iter < LAUNCH_ITERS; iter++) {
    init_input(inputVec_h1, size);
    init_input(inputVec_h2, size);
    HIP_CHECK(hipGraphLaunch(graphExec1, stream1));
    HIP_CHECK(hipGraphLaunch(graphExec2, stream2));
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipStreamSynchronize(stream2));
    validate_output(outputVec_h1, inputVec_h1, size);
    validate_output(outputVec_h2, inputVec_h2, size);
  }
  // Free
  HipTest::freeArrays<int>(inputVec_d1, outputVec_d1, nullptr,
                   inputVec_h1, outputVec_h1, nullptr, false);
  HipTest::freeArrays<int>(inputVec_d2, outputVec_d2, nullptr,
                   inputVec_h2, outputVec_h2, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
}
/* Test scenario 5.1
 */
TEST_CASE("Unit_hipStreamBeginCapture_InterStrmEventSync_defaultflag") {
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  interStrmEventSyncCapture(stream1, stream2);
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 5.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_InterStrmEventSync_blockingflag") {
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));
  interStrmEventSyncCapture(stream1, stream2);
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 5.3
 */
TEST_CASE("Unit_hipStreamBeginCapture_InterStrmEventSync_diffflags") {
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamDefault));
  interStrmEventSyncCapture(stream1, stream2);
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 5.4
 */
TEST_CASE("Unit_hipStreamBeginCapture_InterStrmEventSync_diffprio") {
  hipStream_t stream1, stream2;
  int minPriority = 0, maxPriority = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
  HIP_CHECK(hipStreamCreateWithPriority(&stream1, hipStreamDefault,
                                    minPriority));
  HIP_CHECK(hipStreamCreateWithPriority(&stream2, hipStreamDefault,
                                    maxPriority));
  interStrmEventSyncCapture(stream1, stream2);
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 6.1
 */
TEST_CASE("Unit_hipStreamBeginCapture_ColligatedStrmCapture_defaultflag") {
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  colligatedStrmCapture(stream1, stream2);
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 6.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_ColligatedStrmCapture_blockingflag") {
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));
  colligatedStrmCapture(stream1, stream2);
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 6.3
 */
TEST_CASE("Unit_hipStreamBeginCapture_ColligatedStrmCapture_diffflags") {
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamDefault));
  colligatedStrmCapture(stream1, stream2);
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 6.4
 */
TEST_CASE("Unit_hipStreamBeginCapture_ColligatedStrmCapture_diffprio") {
  hipStream_t stream1, stream2;
  int minPriority = 0, maxPriority = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
  HIP_CHECK(hipStreamCreateWithPriority(&stream1, hipStreamDefault,
                                    minPriority));
  HIP_CHECK(hipStreamCreateWithPriority(&stream2, hipStreamDefault,
                                    maxPriority));
  colligatedStrmCapture(stream1, stream2);
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 7
 */
TEST_CASE("Unit_hipStreamBeginCapture_multiplestrms") {
  hipStream_t stream1, stream2, stream3;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  hipGraph_t graph1, graph2, graph3;
  size_t numNodes1 = 0, numNodes2 = 0, numNodes3 = 0;
  SECTION("Capture Multiple stream with interdependent events") {
    hipEvent_t event1, event2;
    HIP_CHECK(hipEventCreate(&event1));
    HIP_CHECK(hipEventCreate(&event2));
    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipEventRecord(event1, stream1));
    HIP_CHECK(hipStreamWaitEvent(stream2, event1, 0));
    dummyKernel<<<1, 1, 0, stream1>>>();
    HIP_CHECK(hipStreamEndCapture(stream1, &graph1));
    HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipEventRecord(event2, stream2));
    HIP_CHECK(hipStreamWaitEvent(stream3, event2, 0));
    dummyKernel<<<1, 1, 0, stream2>>>();
    HIP_CHECK(hipStreamEndCapture(stream2, &graph2));
    HIP_CHECK(hipStreamBeginCapture(stream3, hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, stream3>>>();
    HIP_CHECK(hipStreamEndCapture(stream3, &graph3));
    HIP_CHECK(hipGraphGetNodes(graph1, nullptr, &numNodes1));
    HIP_CHECK(hipGraphGetNodes(graph2, nullptr, &numNodes2));
    HIP_CHECK(hipGraphGetNodes(graph3, nullptr, &numNodes3));
    REQUIRE(numNodes1 == 1);
    REQUIRE(numNodes2 == 1);
    REQUIRE(numNodes3 == 1);
    HIP_CHECK(hipEventDestroy(event2));
    HIP_CHECK(hipEventDestroy(event1));
  }
  SECTION("Capture Multiple stream with single event") {
    hipEvent_t event1;
    HIP_CHECK(hipEventCreate(&event1));
    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipEventRecord(event1, stream1));
    HIP_CHECK(hipStreamWaitEvent(stream2, event1, 0));
    HIP_CHECK(hipStreamWaitEvent(stream3, event1, 0));
    dummyKernel<<<1, 1, 0, stream1>>>();
    HIP_CHECK(hipStreamEndCapture(stream1, &graph1));
    HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, stream2>>>();
    HIP_CHECK(hipStreamEndCapture(stream2, &graph2));
    HIP_CHECK(hipStreamBeginCapture(stream3, hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, stream3>>>();
    HIP_CHECK(hipStreamEndCapture(stream3, &graph3));
    HIP_CHECK(hipGraphGetNodes(graph1, nullptr, &numNodes1));
    HIP_CHECK(hipGraphGetNodes(graph2, nullptr, &numNodes2));
    HIP_CHECK(hipGraphGetNodes(graph3, nullptr, &numNodes3));
    REQUIRE(numNodes1 == 1);
    REQUIRE(numNodes2 == 1);
    REQUIRE(numNodes3 == 1);
    HIP_CHECK(hipEventDestroy(event1));
  }
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 8
 */
TEST_CASE("Unit_hipStreamBeginCapture_ColligatedStrmCapture_func") {
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  colligatedStrmCaptureFunc(stream1, stream2);
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 9.1
 */
TEST_CASE("Unit_hipStreamBeginCapture_Multithreaded_Global") {
  multithreadedTest(hipStreamCaptureModeGlobal);
}
/* Test scenario 9.2
 */
TEST_CASE("Unit_hipStreamBeginCapture_Multithreaded_ThreadLocal") {
  multithreadedTest(hipStreamCaptureModeThreadLocal);
}
/* Test scenario 9.3
 */
TEST_CASE("Unit_hipStreamBeginCapture_Multithreaded_Relaxed") {
  multithreadedTest(hipStreamCaptureModeRelaxed);
}
/* Test scenario 10
 */
TEST_CASE("Unit_hipStreamBeginCapture_CapturingFromWithinStrms") {
  hipGraph_t graph;
  hipStream_t stream1, stream2, stream3;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  hipEvent_t e1, e2, e3;
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventCreate(&e2));
  HIP_CHECK(hipEventCreate(&e3));
  // Create a device memory of size int and initialize it to 0
  int *devMem{nullptr}, *hostMem{nullptr};
  hostMem = reinterpret_cast<int*>(malloc(sizeof(int)));
  HIP_CHECK(hipMalloc(&devMem, sizeof(int)));
  HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Start Capturing stream1
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(e1, stream1));
  incrementKernel<<<1, 1, 0, stream2>>>(devMem);
  incrementKernel<<<1, 1, 0, stream2>>>(devMem);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem);
  HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
  HIP_CHECK(hipStreamWaitEvent(stream3, e1, 0));
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  incrementKernel<<<1, 1, 0, stream2>>>(devMem);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem);
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  incrementKernel<<<1, 1, 0, stream2>>>(devMem);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem);
  HIP_CHECK(hipEventRecord(e2, stream2));
  HIP_CHECK(hipEventRecord(e3, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream1, e2, 0));
  HIP_CHECK(hipStreamWaitEvent(stream1, e3, 0));
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, sizeof(int),
                           hipMemcpyDefault, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));  // End Capture
  // Reset device memory
  HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Create Executable Graphs
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  REQUIRE((*hostMem) == INCREMENT_KERNEL_FINALEXP_VAL);
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(e3));
  HIP_CHECK(hipEventDestroy(e2));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 11
 */
TEST_CASE("Unit_hipStreamBeginCapture_DetectingInvalidCapture") {
  hipStream_t stream1, stream2;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(event, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, event, 0));
  dummyKernel<<<1, 1, 0, stream1>>>();
  // Since stream2 is already in capture mode due to event wait
  // hipStreamBeginCapture on stream2 is expected to return error.
  REQUIRE(hipSuccess != hipStreamBeginCapture(stream2,
        hipStreamCaptureModeGlobal));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 12
 */
TEST_CASE("Unit_hipStreamBeginCapture_CapturingMultGraphsFrom1Strm") {
  hipStream_t stream1;
  HIP_CHECK(hipStreamCreate(&stream1));
  hipGraph_t graph[3];
  // Create a device memory of size int and initialize it to 0
  int *devMem{nullptr}, *hostMem{nullptr};
  hostMem = reinterpret_cast<int*>(malloc(sizeof(int)));
  HIP_CHECK(hipMalloc(&devMem, sizeof(int)));
  HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Capture Graph1
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, sizeof(int),
                           hipMemcpyDefault, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph[0]));
  // Capture Graph2
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, sizeof(int),
                           hipMemcpyDefault, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph[1]));
  // Capture Graph3
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, sizeof(int),
                           hipMemcpyDefault, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph[2]));
  // Instantiate and execute all graphs
  for (int i = 0; i < 3; i++) {
    hipGraphExec_t graphExec{nullptr};
    HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph[i], nullptr,
                                nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, stream1));
    HIP_CHECK(hipStreamSynchronize(stream1));
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    REQUIRE((*hostMem) == (i + 1));
  }
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
  for (int i = 0; i < 3; i++) {
    HIP_CHECK(hipGraphDestroy(graph[i]));
  }
  HIP_CHECK(hipStreamDestroy(stream1));
}
#if HT_NVIDIA
/* Test scenario 13
 */
TEST_CASE("Unit_hipStreamBeginCapture_CheckingSyncDuringCapture") {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  SECTION("Synchronize stream during capture") {
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    REQUIRE(hipErrorStreamCaptureUnsupported ==
            hipStreamSynchronize(stream));
  }
  SECTION("Synchronize device during capture") {
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    REQUIRE(hipErrorStreamCaptureUnsupported == hipDeviceSynchronize());
  }
  SECTION("Synchronize event during capture") {
    hipEvent_t e;
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipEventRecord(e, stream));
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    REQUIRE(hipErrorStreamCaptureUnsupported == hipEventSynchronize(e));
    HIP_CHECK(hipEventDestroy(e));
  }
  SECTION("Wait for an event during capture") {
    hipEvent_t e;
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipEventRecord(e, stream));
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    REQUIRE(hipErrorStreamCaptureIsolation ==
            hipStreamWaitEvent(stream, e, 0));
    HIP_CHECK(hipEventDestroy(e));
  }
  SECTION("Query stream during capture") {
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    REQUIRE(hipErrorStreamCaptureUnsupported == hipStreamQuery(stream));
  }
  SECTION("Query for an event during capture") {
    hipEvent_t e;
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipEventRecord(e, stream));
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    REQUIRE(hipSuccess != hipEventQuery(e));
    HIP_CHECK(hipEventDestroy(e));
  }
  HIP_CHECK(hipStreamDestroy(stream));
}
#endif
/* Test scenario 14
 */
TEST_CASE("Unit_hipStreamBeginCapture_EndingCapturewhenCaptureInProgress") {
  hipStream_t stream1, stream2;
  hipGraph_t graph;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  SECTION("Abruptly end strm capture when in progress in forked strm") {
    hipEvent_t e;
    HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, stream1>>>();
    HIP_CHECK(hipEventRecord(e, stream1));
    HIP_CHECK(hipStreamWaitEvent(stream2, e, 0));
    dummyKernel<<<1, 1, 0, stream2>>>();
    REQUIRE(hipErrorStreamCaptureUnjoined ==
            hipStreamEndCapture(stream1, &graph));
    HIP_CHECK(hipEventDestroy(e));
  }
  SECTION("End strm capture when forked strm still has operations") {
    hipEvent_t e1, e2;
    HIP_CHECK(hipEventCreate(&e1));
    HIP_CHECK(hipEventCreate(&e2));
    HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
    dummyKernel<<<1, 1, 0, stream1>>>();
    HIP_CHECK(hipEventRecord(e1, stream1));
    HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
    dummyKernel<<<1, 1, 0, stream2>>>();
    HIP_CHECK(hipEventRecord(e2, stream2));
    HIP_CHECK(hipStreamWaitEvent(stream1, e2, 0));
    dummyKernel<<<1, 1, 0, stream2>>>();
    REQUIRE(hipErrorStreamCaptureUnjoined ==
            hipStreamEndCapture(stream1, &graph));
    HIP_CHECK(hipEventDestroy(e2));
    HIP_CHECK(hipEventDestroy(e1));
  }
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}

/* Test scenario 15
 */
TEST_CASE("Unit_hipStreamBeginCapture_MultiGPU") {
  int devcount = 0;
  HIP_CHECK(hipGetDeviceCount(&devcount));
  // If only single GPU is detected then return
  if (devcount < 2) {
    SUCCEED("skipping the testcases as numDevices < 2");
    return;
  }
  hipStream_t* stream = reinterpret_cast<hipStream_t*>(malloc(
                        devcount*sizeof(hipStream_t)));
  REQUIRE(stream != nullptr);
  hipGraph_t* graph = reinterpret_cast<hipGraph_t*>(malloc(
                        devcount*sizeof(hipGraph_t)));
  REQUIRE(graph != nullptr);
  int **devMem{nullptr}, **hostMem{nullptr};
  hostMem = reinterpret_cast<int**>(malloc(sizeof(int*)*devcount));
  REQUIRE(hostMem != nullptr);
  devMem = reinterpret_cast<int**>(malloc(sizeof(int*)*devcount));
  REQUIRE(devMem != nullptr);
  hipGraphExec_t* graphExec = reinterpret_cast<hipGraphExec_t*>(malloc(
                        devcount*sizeof(hipGraphExec_t)));
  // Capture stream in each device
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreate(&stream[dev]));
    hostMem[dev] = reinterpret_cast<int*>(malloc(sizeof(int)));
    HIP_CHECK(hipMalloc(&devMem[dev], sizeof(int)));
    HIP_CHECK(hipStreamBeginCapture(stream[dev],
              hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemsetAsync(devMem[dev], 0, sizeof(int), stream[dev]));
    for (int i = 0; i < (dev + 1); i++) {
      incrementKernel<<<1, 1, 0, stream[dev]>>>(devMem[dev]);
    }
    HIP_CHECK(hipMemcpyAsync(hostMem[dev], devMem[dev], sizeof(int),
                            hipMemcpyDefault, stream[dev]));
    HIP_CHECK(hipStreamEndCapture(stream[dev], &graph[dev]));
  }
  // Launch the captured graphs in the respective device
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipGraphInstantiate(&graphExec[dev], graph[dev], nullptr,
                                nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec[dev], stream[dev]));
  }
  // Validate output
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(stream[dev]));
    REQUIRE((*hostMem[dev]) == (dev + 1));
  }
  // Destroy all device resources
  for (int dev = 0; dev < devcount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipGraphExecDestroy(graphExec[dev]));
    HIP_CHECK(hipStreamDestroy(stream[dev]));
  }
  free(graphExec);
  free(hostMem);
  free(devMem);
  free(stream);
  free(graph);
}
/* Test scenario 16
 */
TEST_CASE("Unit_hipStreamBeginCapture_nestedStreamCapture") {
  hipGraph_t graph;
  hipStream_t stream1, stream2, stream3;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  hipEvent_t e1, e2, e3, e4;
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventCreate(&e2));
  HIP_CHECK(hipEventCreate(&e3));
  HIP_CHECK(hipEventCreate(&e4));
  // Create a device memory of size int and initialize it to 0
  int *devMem{nullptr}, *hostMem{nullptr};
  hostMem = reinterpret_cast<int*>(malloc(sizeof(int)));
  REQUIRE(hostMem != nullptr);
  HIP_CHECK(hipMalloc(&devMem, sizeof(int)));
  HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Start Capturing stream1
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(e1, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
  HIP_CHECK(hipEventRecord(e2, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream3, e2, 0));
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  incrementKernel<<<1, 1, 0, stream2>>>(devMem);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem);
  incrementKernel<<<1, 1, 0, stream1>>>(devMem);
  incrementKernel<<<1, 1, 0, stream2>>>(devMem);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem);
  HIP_CHECK(hipEventRecord(e3, stream2));
  HIP_CHECK(hipEventRecord(e4, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream1, e4, 0));
  HIP_CHECK(hipStreamWaitEvent(stream1, e3, 0));
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, sizeof(int),
                           hipMemcpyDefault, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));  // End Capture
  // Reset device memory
  HIP_CHECK(hipMemset(devMem, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Create Executable Graphs
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  REQUIRE((*hostMem) == INCREMENT_KERNEL_FINALEXP_VAL);
  HIP_CHECK(hipFree(devMem));
  free(hostMem);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(e4));
  HIP_CHECK(hipEventDestroy(e3));
  HIP_CHECK(hipEventDestroy(e2));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 17
 */
TEST_CASE("Unit_hipStreamBeginCapture_streamReuse") {
  hipGraph_t graph1, graph2, graph3;
  hipStream_t stream1, stream2, stream3;
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  hipEvent_t e1, e2, e3, e4;
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventCreate(&e2));
  HIP_CHECK(hipEventCreate(&e3));
  HIP_CHECK(hipEventCreate(&e4));
  // Create a device memory of size int and initialize it to 0
  int *devMem1{nullptr}, *hostMem1{nullptr}, *devMem2{nullptr},
  *hostMem2{nullptr}, *devMem3{nullptr}, *hostMem3{nullptr};
  HipTest::initArrays<int>(&devMem1, &devMem2, &devMem3,
               &hostMem1, &hostMem2, &hostMem3, 1, false);
  HIP_CHECK(hipMemset(devMem1, 0, sizeof(int)));
  HIP_CHECK(hipMemset(devMem2, 0, sizeof(int)));
  HIP_CHECK(hipMemset(devMem3, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Start Capturing stream1
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(e1, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
  HIP_CHECK(hipEventRecord(e2, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream3, e2, 0));
  incrementKernel<<<1, 1, 0, stream1>>>(devMem1);
  incrementKernel<<<1, 1, 0, stream2>>>(devMem1);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem1);
  incrementKernel<<<1, 1, 0, stream1>>>(devMem1);
  incrementKernel<<<1, 1, 0, stream2>>>(devMem1);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem1);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem1);
  HIP_CHECK(hipEventRecord(e3, stream2));
  HIP_CHECK(hipEventRecord(e4, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream1, e4, 0));
  HIP_CHECK(hipStreamWaitEvent(stream1, e3, 0));
  HIP_CHECK(hipMemcpyAsync(hostMem1, devMem1, sizeof(int),
                           hipMemcpyDefault, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph1));  // End Capture
  // Start capturing graph2 from stream 2
  HIP_CHECK(hipStreamBeginCapture(stream2, hipStreamCaptureModeGlobal));
  incrementKernel<<<1, 1, 0, stream2>>>(devMem2);
  incrementKernel<<<1, 1, 0, stream2>>>(devMem2);
  incrementKernel<<<1, 1, 0, stream2>>>(devMem2);
  HIP_CHECK(hipMemcpyAsync(hostMem2, devMem2, sizeof(int),
                           hipMemcpyDefault, stream2));
  HIP_CHECK(hipStreamEndCapture(stream2, &graph2));  // End Capture
  // Start capturing graph3 from stream 3
  HIP_CHECK(hipStreamBeginCapture(stream3, hipStreamCaptureModeGlobal));
  incrementKernel<<<1, 1, 0, stream3>>>(devMem3);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem3);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem3);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem3);
  incrementKernel<<<1, 1, 0, stream3>>>(devMem3);
  HIP_CHECK(hipMemcpyAsync(hostMem3, devMem3, sizeof(int),
                           hipMemcpyDefault, stream3));
  HIP_CHECK(hipStreamEndCapture(stream3, &graph3));  // End Capture
  // Reset device memory
  HIP_CHECK(hipMemset(devMem1, 0, sizeof(int)));
  HIP_CHECK(hipMemset(devMem2, 0, sizeof(int)));
  HIP_CHECK(hipMemset(devMem3, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());
  // Create Executable Graphs
  hipGraphExec_t graphExec{nullptr};
  // Verify graph1
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  REQUIRE((*hostMem1) == INCREMENT_KERNEL_FINALEXP_VAL);
  // Verify graph2
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph2, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream2));
  HIP_CHECK(hipStreamSynchronize(stream2));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  REQUIRE((*hostMem2) == 3);
  // Verify graph3
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph3, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream3));
  HIP_CHECK(hipStreamSynchronize(stream3));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  REQUIRE((*hostMem3) == 5);
  HipTest::freeArrays<int>(devMem1, devMem2, devMem3,
                   hostMem1, hostMem2, hostMem3, false);
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph3));
  HIP_CHECK(hipEventDestroy(e4));
  HIP_CHECK(hipEventDestroy(e3));
  HIP_CHECK(hipEventDestroy(e2));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}

/* Test scenario 18
 */
TEST_CASE("Unit_hipStreamBeginCapture_captureComplexGraph") {
  hipGraph_t graph;
  hipStream_t stream1, stream2, stream3, stream4, stream5;
  // Stream and event create
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  HIP_CHECK(hipStreamCreate(&stream4));
  HIP_CHECK(hipStreamCreate(&stream5));
  hipEvent_t e0, e1, e2, e3, e4, e5, e6;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventCreate(&e2));
  HIP_CHECK(hipEventCreate(&e3));
  HIP_CHECK(hipEventCreate(&e4));
  HIP_CHECK(hipEventCreate(&e5));
  HIP_CHECK(hipEventCreate(&e6));
  // Allocate Device memory and Host memory
  size_t N = GRIDSIZE*BLOCKSIZE;
  int *Ah{nullptr}, *Bh{nullptr}, *Ch{nullptr}, *Ad{nullptr}, *Bd{nullptr};
  HipTest::initArrays<int>(&Ad, &Bd, nullptr, &Ah, &Bh, &Ch, N, false);
  // Capture streams into graph
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(e0, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream4, e0, 0));
  HIP_CHECK(hipStreamWaitEvent(stream5, e0, 0));
  HIP_CHECK(hipMemcpyAsync(Ad, Ah, (N*sizeof(int)),
                        hipMemcpyDefault, stream1));
  HIP_CHECK(hipMemcpyAsync(Bd, Bh, (N*sizeof(int)),
                        hipMemcpyDefault, stream5));
  hipHostFn_t fn = hostNodeCallback;
  HIPCHECK(hipLaunchHostFunc(stream4, fn, nullptr));
  HIP_CHECK(hipEventRecord(e1, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e1, 0));
  int *Ad_2nd_half = Ad + N/2;
  int *Ad_1st_half = Ad;
  mymul<<<GRIDSIZE/2, BLOCKSIZE, 0, stream1>>>(Ad_2nd_half, CONST_KER2_VAL);
  mymul<<<GRIDSIZE/2, BLOCKSIZE, 0, stream2>>>(Ad_1st_half, CONST_KER1_VAL);
  HIP_CHECK(hipEventRecord(e2, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream3, e2, 0));
  mymul<<<GRIDSIZE/2, BLOCKSIZE, 0, stream2>>>(Ad_1st_half, CONST_KER3_VAL);
  HIPCHECK(hipLaunchHostFunc(stream3, fn, nullptr));
  HIP_CHECK(hipEventRecord(e6, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, e6, 0));
  HIP_CHECK(hipEventRecord(e5, stream5));
  HIP_CHECK(hipStreamWaitEvent(stream1, e5, 0));
  myadd<<<GRIDSIZE, BLOCKSIZE, 0, stream1>>>(Ad, Bd);
  HIP_CHECK(hipEventRecord(e3, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream1, e3, 0));
  HIP_CHECK(hipEventRecord(e4, stream4));
  HIP_CHECK(hipStreamWaitEvent(stream1, e4, 0));
  HIP_CHECK(hipMemcpyAsync(Ch, Ad, (N*sizeof(int)),
                        hipMemcpyDefault, stream1));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));  // End Capture
  // Execute and test the graph
  // Create Executable Graphs
  hipGraphExec_t graphExec{nullptr};
  // Verify graph1
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  for (int iter = 0; iter < LAUNCH_ITERS; iter++) {
    init_input(Ah, N);
    init_input(Bh, N);
    HIP_CHECK(hipGraphLaunch(graphExec, stream1));
    HIP_CHECK(hipStreamSynchronize(stream1));
    for (size_t i = 0; i < N; i++) {
      if (i > (N/2 - 1)) {
        REQUIRE(Ch[i] == (Bh[i] + Ah[i]*CONST_KER2_VAL));
      } else {
        REQUIRE(Ch[i] == (Bh[i] + Ah[i]*CONST_KER1_VAL*CONST_KER3_VAL));
      }
    }
  }
  REQUIRE(gCbackIter == (2*LAUNCH_ITERS));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  // Free Device memory and Host memory
  HipTest::freeArrays<int>(Ad, Bd, nullptr, Ah, Bh, Ch, false);
  // Destroy graph, events and streams
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(e6));
  HIP_CHECK(hipEventDestroy(e5));
  HIP_CHECK(hipEventDestroy(e4));
  HIP_CHECK(hipEventDestroy(e3));
  HIP_CHECK(hipEventDestroy(e2));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipEventDestroy(e0));
  HIP_CHECK(hipStreamDestroy(stream5));
  HIP_CHECK(hipStreamDestroy(stream4));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}
/* Test scenario 19
 */
TEST_CASE("Unit_hipStreamBeginCapture_captureEmptyStreams") {
  hipGraph_t graph;
  hipStream_t stream1, stream2, stream3;
  // Stream and event create
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipStreamCreate(&stream3));
  hipEvent_t e0, e1, e2;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventCreate(&e2));
  // Capture streams into graph
  HIP_CHECK(hipStreamBeginCapture(stream1, hipStreamCaptureModeGlobal));
  HIP_CHECK(hipEventRecord(e0, stream1));
  HIP_CHECK(hipStreamWaitEvent(stream2, e0, 0));
  HIP_CHECK(hipStreamWaitEvent(stream3, e0, 0));
  HIP_CHECK(hipEventRecord(e1, stream2));
  HIP_CHECK(hipStreamWaitEvent(stream1, e1, 0));
  HIP_CHECK(hipEventRecord(e2, stream3));
  HIP_CHECK(hipStreamWaitEvent(stream1, e2, 0));
  HIP_CHECK(hipStreamEndCapture(stream1, &graph));  // End Capture
  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  REQUIRE(numNodes == 0);
  // Destroy graph, events and streams
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(e2));
  HIP_CHECK(hipEventDestroy(e1));
  HIP_CHECK(hipEventDestroy(e0));
  HIP_CHECK(hipStreamDestroy(stream3));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream1));
}

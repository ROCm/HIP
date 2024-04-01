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

#include <hip_test_common.hh>

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <random>
#include <thread>

__global__ void addVal(unsigned long long* ptr, size_t index,
                       unsigned long long val) {
  atomicAdd(ptr + index, val);
}

// Create a copy constructible AtomicWrap around std::atomic so that
// we can put it in a vector
template <typename T> struct AtomicWrap {
  std::atomic<T> data;

  AtomicWrap() : data() {}

  AtomicWrap(T i) : data(i) {}

  AtomicWrap(const std::atomic<T>& a) : data(a.load()) {}

  AtomicWrap(const AtomicWrap& other) : data(other.data.load()) {}

  AtomicWrap& operator=(const AtomicWrap& other) {
    data.store(other.data.load());
    return *this;
  }
};

// Have multiple threads and enqueue commands from them on a single stream
// Validate at the end that all commands have completed successfully
TEST_CASE("Stress_StreamEnqueue_DifferentThreads") {
  auto hwThreads = std::thread::hardware_concurrency();
  hwThreads = (hwThreads >= 2) ? hwThreads : 2;  // Run atleast 2 threads

  std::vector<AtomicWrap<unsigned long long>> hostData(hwThreads, 0);

  unsigned long long* dPtr{nullptr};
  HIP_CHECK(hipMalloc(&dPtr, sizeof(unsigned long long) * hwThreads));
  REQUIRE(dPtr != nullptr);

  HIP_CHECK(hipMemset(dPtr, 0, sizeof(unsigned long long) * hwThreads));

  std::random_device device;
  std::mt19937 engine(device());

  constexpr size_t maxWork = 10000;
  constexpr size_t maxVal = 10;

  std::uniform_int_distribution<std::mt19937::result_type> genIndex(0,
                                                           hwThreads - 1);
  std::uniform_int_distribution<std::mt19937::result_type> genWork(0, maxWork);
  std::uniform_int_distribution<std::mt19937::result_type> genVal(0, maxVal);

  auto enqueueKernelThread = [&](hipStream_t stream) {
    auto iter = genWork(engine);  // Generate work to be done via thread
    for (unsigned long i = 0; i < iter; i++) {
      auto index = genIndex(engine);  // Generate Index to add to
      auto val = genVal(engine);  // Generate value to add to the destination
      hostData[index].data += val;    // Replicate it on host
      addVal<<<1, 1, 0, stream>>>(dPtr, static_cast<size_t>(index),
            static_cast<unsigned long long>(val));  // And on device
    }
  };

  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  std::vector<std::thread> threadPool{};
  threadPool.reserve(hwThreads);

  // Launch work
  for (size_t i = 0; i < hwThreads; i++) {
    threadPool.emplace_back(std::thread(enqueueKernelThread, stream));
  }

  // Wait for work to finish
  for (auto& i : threadPool) {
    i.join();
  }

  HIP_CHECK(hipStreamDestroy(stream));

  auto hPtr = std::make_unique<unsigned long long[]>(hwThreads);
  HIP_CHECK(hipMemcpy(hPtr.get(), dPtr, sizeof(unsigned long long) * hwThreads,
            hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(dPtr));

  // Validate that CPU and GPU has the same results
  for (size_t i = 0; i < hwThreads; i++) {
    INFO("Check for Index " << i);
    REQUIRE(hostData[i].data.load() == hPtr[i]);
  }
}

__global__ void doOperation(int* dPtr, int val) {
  auto i = threadIdx.x;
  atomicAdd(dPtr + i, val);
}

// Allocate mulitple stream for same device.
// Same device stream operate on same memory
TEST_CASE("Stress_StreamEnqueue_DifferentThreads_MultiGPU") {
  int deviceCount{0};
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount > 0);

  // Skip the test if devices less than 2
  if (deviceCount <= 1) {
    HipTest::HIP_SKIP_TEST("Skipping because devices <= 1");
    return;
  }

  constexpr size_t streamPerGPU{3};  // Stream per gpu

  std::vector<hipStream_t> streamPool{};
  streamPool.reserve(deviceCount * streamPerGPU);
  // Map of stream and device memory
  std::map<hipStream_t, int*> streamToDeviceMemory;
  // Map of stream and host result
  std::map<hipStream_t, AtomicWrap<int>> streamToHostMemory;
  // Map of stream and device it was created on
  std::map<hipStream_t, size_t> streamToDeviceIndex;
  constexpr size_t size = 1024;

  for (int i = 0; i < deviceCount; i++) {
    HIP_CHECK(hipSetDevice(i));

    for (size_t j = 0; j < streamPerGPU; j++) {
      hipStream_t stream{nullptr};
      HIP_CHECK(hipStreamCreate(&stream));
      REQUIRE(stream != nullptr);
      streamPool.push_back(stream);

      int* dPtr{nullptr};
      HIP_CHECK(hipMalloc(&dPtr, sizeof(int) * size));
      REQUIRE(dPtr != nullptr);
      HIP_CHECK(hipMemset(dPtr, 0, sizeof(int) * size));
      // All streams work on exclusive memory
      streamToDeviceMemory[stream] = dPtr;

      streamToHostMemory[stream] = AtomicWrap<int>(0);  // CPU result

      streamToDeviceIndex[stream] = i;  // Capture device id for stream
    }
  }

  constexpr size_t maxVal = 5;
  constexpr size_t maxWorkPerThread = 10000;

  // Boiler plate code to generate a random number
  std::random_device device;
  std::mt19937 engine(device());

  std::uniform_int_distribution<std::mt19937::result_type> genVal(-maxVal,
                                                           maxVal);
  std::uniform_int_distribution<std::mt19937::result_type> genStream(0,
                                              streamPool.size() - 1);

#if HT_NVIDIA
  std::mutex ness;  // On nvidia, current device needs to match stream's device
#endif

  auto enqueueKernelThread = [&]() {
    for (size_t i = 0; i < maxWorkPerThread; i++) {
#if HT_NVIDIA
      std::unique_lock<std::mutex> lock(ness);  // Lock on creation
#endif
      // Get a random stream
      hipStream_t stream = streamPool[genStream(engine)];

      // TODO use HIP_CHECK_THREAD when PR#2664 is merged
      if (hipSuccess != hipSetDevice(streamToDeviceIndex[stream])) {
        return;
      }

      int val = genVal(engine);  // Generate Value to add/sub to
      // Replicate result on CPU
      streamToHostMemory[stream].data.fetch_add(val);
      auto dPtr = streamToDeviceMemory[stream];
      doOperation<<<1, 1024, 0, stream>>>(dPtr, val);  // On GPU
    }
  };

  auto maxThreads = std::thread::hardware_concurrency();
  maxThreads = (maxThreads >= 2) ? maxThreads : 2;  // Run atleast 2 threads

  std::vector<std::thread> threadPool{};
  threadPool.reserve(maxThreads);

  // Launch Threads
  for (size_t i = 0; i < maxThreads; i++) {
    threadPool.emplace_back(std::thread(enqueueKernelThread));
  }

  // Wait for them to stop
  for (auto& i : threadPool) {
    i.join();
  }

  // Sync and check results
  for (auto& i : streamPool) {
    HIP_CHECK(hipStreamSynchronize(i));
    auto dResult = std::make_unique<int[]>(size);
    HIP_CHECK(hipMemcpy(dResult.get(), streamToDeviceMemory[i],
              sizeof(int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(streamToDeviceMemory[i]));
    HIP_CHECK(hipStreamDestroy(i));
    auto res = streamToHostMemory[i].data.load();
    INFO("Matching CPU: " << res << " GPU: " << dResult[0] << " Dev Ptr: "
    << streamToDeviceMemory[i] << " on Device: " << streamToDeviceIndex[i]);
    REQUIRE(std::all_of(dResult.get(), dResult.get() + size,
            [=](int r) { return r == res; }));
  }
}

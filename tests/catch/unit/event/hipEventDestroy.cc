#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <hip_test_common.hh>

static constexpr size_t vectorSize{16384};


static inline void launchVectorAdd(float*& A_h, float*& B_h, float*& C_h) {
  float* A_d{nullptr};
  float* B_d{nullptr};
  float* C_d{nullptr};
  HipTest::initArraysForHost(&A_h, &B_h, &C_h, vectorSize, true);
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&B_d), B_h, 0));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&C_d), C_h, 0));
  HipTest::vectorADD<<<1, 1>>>(A_d, B_d, C_d, vectorSize);
}


/**
 * @brief Check that destroying an event before the kernel has finished running causes no errors.
 *
 */
TEST_CASE("Unit_hipEventDestroy_Unfinished") {
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  float *A_h, *B_h, *C_h;
  launchVectorAdd(A_h, B_h, C_h);

  HIP_CHECK(hipEventRecord(event));
  HIP_CHECK_ERROR(hipEventQuery(event), hipErrorNotReady);
  HIP_CHECK(hipEventDestroy(event));

  HIP_CHECK(hipDeviceSynchronize());
  HipTest::checkVectorADD(A_h, B_h, C_h, vectorSize);
  HipTest::freeArraysForHost(A_h, B_h, C_h, true);
}

/**
 * @brief Check that destroying an event enqueued to a stream causes no errors.
 *
 */
TEST_CASE("Unit_hipEventDestroy_WithWaitingStream") {
  float *A_d, *B_d, *C_d;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), vectorSize));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&B_d), vectorSize));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&C_d), vectorSize));

  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  float *A_h, *B_h, *C_h;
  launchVectorAdd(A_h, B_h, C_h);

  HIP_CHECK(hipEventRecord(event, stream));
  HIP_CHECK_ERROR(hipEventQuery(event), hipErrorNotReady);
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK_ERROR(hipStreamQuery(stream), hipErrorNotReady);
  HIP_CHECK(hipStreamSynchronize(stream));

  HipTest::checkVectorADD(A_h, B_h, C_h, vectorSize);
  HipTest::freeArraysForHost(A_h, B_h, C_h, true);
}

TEST_CASE("Unit_hipEventDestroy_Negative") {
  
  SECTION("Invalid Event") {
    hipEvent_t event{nullptr};
    HIP_CHECK_ERROR(hipEventDestroy(event), hipErrorInvalidResourceHandle);
  }

#ifndef HT_AMD
  SECTION("Destroy twice") {
    hipEvent_t event;
    HIP_CHECK(hipEventCreate(&event));
    HIP_CHECK(hipEventDestroy(event));
    HIP_CHECK_ERROR(hipEventDestroy(event), hipErrorContextIsDestroyed);
  }
#endif
}

#include <hip_test_common.hh>
#include <iostream>

template <typename T> __global__ void add(T* a, T* b, T* c, size_t size) {
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

TEMPLATE_TEST_CASE("Add Kernel", "[kernel][add]", int, long, float, long long, double) {
  auto addKernel = add<TestType>;
  auto size = GENERATE(as<size_t>{}, 100, 500, 1000);
  TestType *d_a, *d_b, *d_c;
  auto res = hipMalloc(&d_a, sizeof(TestType) * size);
  REQUIRE(res == hipSuccess);
  res = hipMalloc(&d_b, sizeof(TestType) * size);
  REQUIRE(res == hipSuccess);
  res = hipMalloc(&d_c, sizeof(TestType) * size);
  REQUIRE(res == hipSuccess);

  std::vector<TestType> a, b, c;
  for (int i = 0; i < size; i++) {
    a.push_back(i + 1);
    b.push_back(i + 1);
    c.push_back(2 * (i + 1));
  }

  res = hipMemcpy(d_a, a.data(), sizeof(TestType) * size, hipMemcpyHostToDevice);
  REQUIRE(res == hipSuccess);
  res = hipMemcpy(d_b, b.data(), sizeof(TestType) * size, hipMemcpyHostToDevice);
  REQUIRE(res == hipSuccess);

  hipLaunchKernelGGL(addKernel, 1, size, 0, 0, d_a, d_b, d_c, size);

  res = hipMemcpy(a.data(), d_c, sizeof(TestType) * size, hipMemcpyDeviceToHost);
  REQUIRE(res == hipSuccess);

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);
  REQUIRE(a == c);
}

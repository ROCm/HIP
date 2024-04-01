#include "../device_tests_common.hh"

__global__ void testKernel_max(int* a) {
  const unsigned long long cull_int = 1;
  const long long int cll_int = 1;
  unsigned long long int ull_int;
  long long int ll_int;
  const unsigned long int cul_int = 1;
  const long int cl_int = 1;
  unsigned long int ul_int;
  long int l_int;
  const unsigned int cu_int = 1;
  const int c_int = 1;
  unsigned int u_int;
  int _int;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  ull_int = max(cull_int, cll_int);
  ull_int = max(cll_int, cull_int);
  ull_int = max(cull_int, cull_int);
  ll_int = max(cll_int, cll_int);
  ul_int = max(cul_int, cl_int);
  ul_int = max(cl_int, cul_int);
  ul_int = max(cul_int, cul_int);
  l_int = max(cl_int, cl_int);
  u_int = max(cu_int, c_int);
  u_int = max(c_int, cu_int);
  u_int = max(cu_int, cu_int);
  _int = max(c_int, c_int);
  ull_int = ullmax(cull_int, cull_int);
  u_int = umax(cu_int, cu_int); 
}

TEST_CASE("Unit_deviceFunctions_CompileTest_max_int") {
  int* Outd;
  auto res = hipMalloc((void**)&Outd, SIZE);
  REQUIRE(res == hipSuccess);
  hipLaunchKernelGGL(testKernel_max, dim3(LEN, 1, 1), dim3(1, 1, 1), 0, 0, Outd);
  HIP_CHECK(hipGetLastError());
  res = hipDeviceSynchronize();
  REQUIRE(res == hipSuccess);
  res = hipGetLastError();
  REQUIRE(res == hipSuccess);
  HIP_CHECK(hipFree(Outd));
}
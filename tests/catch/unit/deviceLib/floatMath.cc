#include <hip_test_common.hh>

#define LEN 512
#define SIZE LEN << 2

__global__ void floatMath(float* In, float* Out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  Out[tid] = __cosf(In[tid]);
  Out[tid] = __exp10f(Out[tid]);
  Out[tid] = __expf(Out[tid]);
  Out[tid] = __frsqrt_rn(Out[tid]);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  Out[tid] = __fsqrt_rd(Out[tid]);
#endif
  Out[tid] = __fsqrt_rn(Out[tid]);
#if defined OCML_BASIC_ROUNDED_OPERATIONS
  Out[tid] = __fsqrt_ru(Out[tid]);
  Out[tid] = __fsqrt_rz(Out[tid]);
#endif
  Out[tid] = __log10f(Out[tid]);
  Out[tid] = __log2f(Out[tid]);
  Out[tid] = __logf(Out[tid]);
  Out[tid] = __powf(2.0f, Out[tid]);
  __sincosf(Out[tid], &In[tid], &Out[tid]);
  Out[tid] = __sinf(Out[tid]);
  Out[tid] = __cosf(Out[tid]);
  Out[tid] = __tanf(Out[tid]);
}

TEST_CASE("FloatMathTest") {
  float *Ind, *Outd;
  auto res = hipMalloc((void**)&Ind, SIZE);
  REQUIRE(res == hipSuccess);
  res = hipMalloc((void**)&Outd, SIZE);
  REQUIRE(res == hipSuccess);
  hipLaunchKernelGGL(floatMath, dim3(LEN, 1, 1), dim3(1, 1, 1), 0, 0, Ind, Outd);
  res = hipDeviceSynchronize();
  REQUIRE(res == hipSuccess);
  res = hipGetLastError();
  REQUIRE(res == hipSuccess);
}

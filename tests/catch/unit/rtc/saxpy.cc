#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>


#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

static constexpr auto NUM_THREADS{128};
static constexpr auto NUM_BLOCKS{32};

static constexpr auto saxpy{
    R"(
extern "C"
__global__
void saxpy(float a, float* x, float* y, float* out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
           out[tid] = a * x[tid] + y[tid];
    }
}
)"};

TEST_CASE("Unit_hiprtc_saxpy") {
  using namespace std;
  hiprtcProgram prog;
  hiprtcCreateProgram(&prog,       // prog
                      saxpy,       // buffer
                      "saxpy.cu",  // name
                      0, nullptr, nullptr);
  hipDeviceProp_t props;
  int device = 0;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
#ifdef __HIP_PLATFORM_AMD__
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
#else
  std::string sarg = std::string("--fmad=false");
#endif
  const char* options[] = {sarg.c_str()};
  hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  vector<char> code(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));

  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  // Do hip malloc first so that we donot need to do a cuInit manually before calling hipModule APIs
  size_t n = NUM_THREADS * NUM_BLOCKS;
  size_t bufferSize = n * sizeof(float);

  float *dX, *dY, *dOut;
  HIP_CHECK(hipMalloc(&dX, bufferSize));
  HIP_CHECK(hipMalloc(&dY, bufferSize));
  HIP_CHECK(hipMalloc(&dOut, bufferSize));

  hipModule_t module;
  hipFunction_t kernel;
  HIP_CHECK(hipModuleLoadData(&module, code.data()));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "saxpy"));

  float a = 5.1f;
  unique_ptr<float[]> hX{new float[n]};
  unique_ptr<float[]> hY{new float[n]};
  unique_ptr<float[]> hOut{new float[n]};
  for (size_t i = 0; i < n; ++i) {
    hX[i] = static_cast<float>(i);
    hY[i] = static_cast<float>(i * 2);
  }

  HIP_CHECK(hipMemcpy(dX, hX.get(), bufferSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dY, hY.get(), bufferSize, hipMemcpyHostToDevice));

  struct {
    float a_;
    float* b_;
    float* c_;
    float* d_;
    size_t e_;
  } args{a, dX, dY, dOut, n};

  auto size = sizeof(args);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  hipModuleLaunchKernel(kernel, NUM_BLOCKS, 1, 1, NUM_THREADS, 1, 1, 0, nullptr, nullptr, config);

  HIP_CHECK(hipMemcpy(hOut.get(), dOut, bufferSize, hipMemcpyDeviceToHost));

  hipFree(dX);
  hipFree(dY);
  hipFree(dOut);

  HIP_CHECK(hipModuleUnload(module));

  for (size_t i = 0; i < n; ++i) {
    INFO("For " << i << " Value: " << fabs(a * hX[i] + hY[i] - hOut[i])
                << " with: " << (fabs(hOut[i] * 1.0f) * 1e-6));
    REQUIRE(fabs(a * hX[i] + hY[i] - hOut[i]) <= fabs(hOut[i]) * 1e-6);
  }
}

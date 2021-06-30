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
#include <hip/hip_runtime.h>
extern "C"
__global__
void saxpy(float a, float* x, float* y, float* out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
               out[tid] = a * x[tid] + y[tid] ;
                   
        }
        
}
)"};

TEST_CASE("saxpy", "[hiprtc][saxpy]") {
  using namespace std;
  hiprtcProgram prog;
  hiprtcCreateProgram(&prog,       // prog
                      saxpy,       // buffer
                      "saxpy.cu",  // name
                      0, nullptr, nullptr);
  hipDeviceProp_t props;
  int device = 0;
  hipGetDeviceProperties(&props, device);
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
  const char* options[] = {sarg.c_str()};
  hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};
  size_t logSize;
  hiprtcGetProgramLogSize(prog, &logSize);
  if (logSize) {
    string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);

    std::cout << log << '\n';
  }
  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  hiprtcGetCodeSize(prog, &codeSize);

  vector<char> code(codeSize);
  hiprtcGetCode(prog, code.data());

  hiprtcDestroyProgram(&prog);

  hipModule_t module;
  hipFunction_t kernel;
  hipModuleLoadData(&module, code.data());
  hipModuleGetFunction(&kernel, module, "saxpy");

  size_t n = NUM_THREADS * NUM_BLOCKS;
  size_t bufferSize = n * sizeof(float);

  float a = 5.1f;
  unique_ptr<float[]> hX{new float[n]};
  unique_ptr<float[]> hY{new float[n]};
  unique_ptr<float[]> hOut{new float[n]};
for (size_t i = 0; i < n; ++i) {
        hX[i] = static_cast<float>(i);
        hY[i] = static_cast<float>(i * 2);
    }

    hipDeviceptr_t dX, dY, dOut;
    hipMalloc(&dX, bufferSize);
    hipMalloc(&dY, bufferSize);
    hipMalloc(&dOut, bufferSize);
    hipMemcpyHtoD(dX, hX.get(), bufferSize);
    hipMemcpyHtoD(dY, hY.get(), bufferSize);

    struct {
        float a_;
        hipDeviceptr_t b_;
        hipDeviceptr_t c_;
        hipDeviceptr_t d_;
        size_t e_;
    } args{a, dX, dY, dOut, n};

    auto size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    hipModuleLaunchKernel(kernel, NUM_BLOCKS, 1, 1, NUM_THREADS, 1, 1,
                          0, nullptr, nullptr, config);
    hipMemcpyDtoH(hOut.get(), dOut, bufferSize);

    for (size_t i = 0; i < n; ++i) {
      REQUIRE(fabs(a * hX[i] + hY[i] - hOut[i]) > fabs(hOut[i]) * 1e-6);
    }

    hipFree(dX);
    hipFree(dY);
    hipFree(dOut);

    hipModuleUnload(module);
}

#include <hip_test_common.hh>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>


#include <cassert>
#include <cstddef>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>
#include <map>

static constexpr auto program{
    R"(
extern "C"
__global__ void kernel(int* a) {
  // C++17 feature
  if (int j = 10; *a % 2 == 0)
    *a = 10 + j;
  else
    *a = 20 + j;
}
)"};

TEST_CASE("Unit_hiprtc_cpp17") {
  using namespace std;
  hiprtcProgram prog;
  hiprtcCreateProgram(&prog,         // prog
                      program,       // buffer
                      "program.cu",  // name
                      0, nullptr, nullptr);
  hipDeviceProp_t props;
  int device = 0;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
#ifdef __HIP_PLATFORM_AMD__
  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
#else
  std::string sarg = std::string("--fmad=false");
#endif
  const char* options[] = {sarg.c_str(), "-std=c++17", "-Werror"};
  hiprtcResult compileResult{hiprtcCompileProgram(prog, 3, options)};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }
  hiprtcDestroyProgram(&prog);
  REQUIRE(compileResult == HIPRTC_SUCCESS);
}

static constexpr const char template_kernel[]{R"(
template <typename T> struct complex {
 public:
  typedef T value_type;
  inline __host__ __device__ complex(const T& re, const T& im);
  __host__ __device__ inline complex<T>& operator*=(const complex<T> z);
  __host__ __device__ inline T real() const volatile { return m_data[0];  }
  __host__ __device__ inline T imag() const volatile { return m_data[1];  }
  __host__ __device__ inline T real() const { return m_data[0];  }
  __host__ __device__ inline T imag() const { return m_data[1];  }
  __host__ __device__ inline void real(T re) volatile { m_data[0] = re;  }
  __host__ __device__ inline void imag(T im) volatile { m_data[1] = im;  }
  __host__ __device__ inline void real(T re) { m_data[0] = re;  }
  __host__ __device__ inline void imag(T im) { m_data[1] = im;  }

 private:
  T m_data[2];
};
template <typename T> inline __host__ __device__ complex<T>::complex(const T& re, const T& im) {
  real(re);
  imag(im);
}
template <typename T>
__host__ __device__ inline complex<T>& complex<T>::operator*=(const complex<T> z) {
  *this = *this * z;
  return *this;
}
template <typename T>
__host__ __device__ inline complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
  return complex<T>(lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
                    lhs.real() * rhs.imag() + lhs.imag() * rhs.real());
}

template <typename T> __global__ void my_sqrt(T* input, int N) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    input[x] *= input[x];
  }
}
)"};

TEST_CASE("Unit_hiprtc_namehandling") {
  using namespace std;
  hiprtcProgram prog;
  hiprtcCreateProgram(&prog,                 // prog
                      template_kernel,       // buffer
                      "template_kernel.cu",  // name
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

  std::vector<std::string> name_expressions;
  name_expressions.push_back("my_sqrt<int>");
  name_expressions.push_back("my_sqrt<float>");
  name_expressions.push_back("my_sqrt<complex<double>>");
  name_expressions.push_back("my_sqrt<complex<double> >");
  for (size_t i = 0; i < name_expressions.size(); i++) {
    REQUIRE(HIPRTC_SUCCESS == hiprtcAddNameExpression(prog, name_expressions[i].c_str()));
  }

  hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};

  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    std::cout << log << '\n';
  }

  std::map<std::string, std::string> mangled_names;

  for (size_t i = 0; i < name_expressions.size(); i++) {
    const char* mangled_instantiation_cstr;
    REQUIRE(HIPRTC_SUCCESS == hiprtcGetLoweredName(prog, name_expressions[i].c_str(), &mangled_instantiation_cstr));

    std::string mangled_name_str = mangled_instantiation_cstr;
    mangled_names[name_expressions[i]] = mangled_name_str;
    REQUIRE(mangled_name_str.size() > 0);
  }

  // Match the last two names
  REQUIRE(mangled_names["my_sqrt<complex<double>>"] == mangled_names["my_sqrt<complex<double> >"]);

  hiprtcDestroyProgram(&prog);
  REQUIRE(compileResult == HIPRTC_SUCCESS);
}

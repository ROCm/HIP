#include "CUDA2HIP.h"

// Maps CUDA header names to HIP header names
const std::map <llvm::StringRef, hipCounter> CUDA_INCLUDE_MAP{
  // CUDA includes
  {"cuda.h",                    {"hip/hip_runtime.h",        CONV_INCLUDE_CUDA_MAIN_H, API_DRIVER}},
  {"cuda_runtime.h",            {"hip/hip_runtime.h",        CONV_INCLUDE_CUDA_MAIN_H, API_RUNTIME}},
  {"cuda_runtime_api.h",        {"hip/hip_runtime_api.h",    CONV_INCLUDE,             API_RUNTIME}},
  {"channel_descriptor.h",      {"hip/channel_descriptor.h", CONV_INCLUDE,             API_RUNTIME}},
  {"device_functions.h",        {"hip/device_functions.h",   CONV_INCLUDE,             API_RUNTIME}},
  {"driver_types.h",            {"hip/driver_types.h",       CONV_INCLUDE,             API_RUNTIME}},
  {"cuda_fp16.h",               {"hip/hip_fp16.h",           CONV_INCLUDE,             API_RUNTIME}},
  {"cuda_texture_types.h",      {"hip/hip_texture_types.h",  CONV_INCLUDE,             API_RUNTIME}},
  {"vector_types.h",            {"hip/hip_vector_types.h",   CONV_INCLUDE,             API_RUNTIME}},
  // cuBLAS includes
  {"cublas.h",                  {"hipblas.h",                CONV_INCLUDE_CUDA_MAIN_H, API_BLAS}},
  {"cublas_v2.h",               {"hipblas.h",                CONV_INCLUDE_CUDA_MAIN_H, API_BLAS}},
  // cuRAND includes
  {"curand.h",                  {"hiprand.h",                CONV_INCLUDE_CUDA_MAIN_H, API_RAND}},
  {"curand_kernel.h",           {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_discrete.h",         {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_discrete2.h",        {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_globals.h",          {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_lognormal.h",        {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_mrg32k3a.h",         {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_mtgp32.h",           {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_mtgp32_host.h",      {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_mtgp32_kernel.h",    {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_mtgp32dc_p_11213.h", {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_normal.h",           {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_normal_static.h",    {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_philox4x32_x.h",     {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_poisson.h",          {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_precalc.h",          {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  {"curand_uniform.h",          {"hiprand_kernel.h",         CONV_INCLUDE,             API_RAND}},
  // cuDNN includes
  {"cudnn.h",                   {"hipDNN.h",                 CONV_INCLUDE_CUDA_MAIN_H, API_DNN}},
  // cuFFT includes
  {"cufft.h",                   {"hipfft.h",                 CONV_INCLUDE_CUDA_MAIN_H, API_FFT}},
  // cuComplex includes
  {"cuComplex.h",               {"hip/hip_complex.h",        CONV_INCLUDE_CUDA_MAIN_H, API_COMPLEX}},
};

const std::map<llvm::StringRef, hipCounter>& CUDA_RENAMES_MAP() {
  static std::map<llvm::StringRef, hipCounter> ret;
  if (!ret.empty()) {
    return ret;
  }
  // First run, so compute the union map.
  ret.insert(CUDA_DRIVER_TYPE_NAME_MAP.begin(), CUDA_DRIVER_TYPE_NAME_MAP.end());
  ret.insert(CUDA_DRIVER_FUNCTION_MAP.begin(), CUDA_DRIVER_FUNCTION_MAP.end());
  ret.insert(CUDA_RUNTIME_TYPE_NAME_MAP.begin(), CUDA_RUNTIME_TYPE_NAME_MAP.end());
  ret.insert(CUDA_RUNTIME_FUNCTION_MAP.begin(), CUDA_RUNTIME_FUNCTION_MAP.end());
  ret.insert(CUDA_COMPLEX_TYPE_NAME_MAP.begin(), CUDA_COMPLEX_TYPE_NAME_MAP.end());
  ret.insert(CUDA_COMPLEX_FUNCTION_MAP.begin(), CUDA_COMPLEX_FUNCTION_MAP.end());
  ret.insert(CUDA_BLAS_TYPE_NAME_MAP.begin(), CUDA_BLAS_TYPE_NAME_MAP.end());
  ret.insert(CUDA_BLAS_FUNCTION_MAP.begin(), CUDA_BLAS_FUNCTION_MAP.end());
  ret.insert(CUDA_RAND_TYPE_NAME_MAP.begin(), CUDA_RAND_TYPE_NAME_MAP.end());
  ret.insert(CUDA_RAND_FUNCTION_MAP.begin(), CUDA_RAND_FUNCTION_MAP.end());
  ret.insert(CUDA_DNN_TYPE_NAME_MAP.begin(), CUDA_DNN_TYPE_NAME_MAP.end());
  ret.insert(CUDA_DNN_FUNCTION_MAP.begin(), CUDA_DNN_FUNCTION_MAP.end());
  ret.insert(CUDA_FFT_TYPE_NAME_MAP.begin(), CUDA_FFT_TYPE_NAME_MAP.end());
  ret.insert(CUDA_FFT_FUNCTION_MAP.begin(), CUDA_FFT_FUNCTION_MAP.end());
  return ret;
};

// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// CHECK: #include <hip/hip_runtime.h>
// CHECK: #include <memory>

// CHECK-NOT: #include <cuda_runtime.h>
// CHECK-NOT: #include <hip/hip_runtime.h>

// CHECK: #include "hip/hip_runtime_api.h"
// CHECK: #include "hip/channel_descriptor.h"
// CHECK: #include "hip/device_functions.h"
// CHECK: #include "hip/driver_types.h"
// CHECK: #include "hip/hip_complex.h"
// CHECK: #include "hip/hip_fp16.h"
// CHECK: #include "hip/hip_texture_types.h"
// CHECK: #include "hip/hip_vector_types.h"

// CHECK: #include <iostream>

// CHECK: #include "hipblas.h"
// CHECK-NOT: #include "cublas.h"

// CHECK: #include <stdio.h>

// CHECK: #include "hiprand.h"
// CHECK: #include "hiprand_kernel.h"

// CHECK: #include <algorithm>

// CHECK-NOT: #include "hiprand.h"
// CHECK-NOT: #include "hiprand_kernel.h"
// CHECK-NOT: #include "curand_discrete.h"
// CHECK-NOT: #include "curand_discrete2.h"
// CHECK-NOT: #include "curand_globals.h"
// CHECK-NOT: #include "curand_lognormal.h"
// CHECK-NOT: #include "curand_mrg32k3a.h"
// CHECK-NOT: #include "curand_mtgp32.h"
// CHECK-NOT: #include "curand_mtgp32_host.h"
// CHECK-NOT: #include "curand_mtgp32_kernel.h"
// CHECK-NOT: #include "curand_mtgp32dc_p_11213.h"
// CHECK-NOT: #include "curand_normal.h"
// CHECK-NOT: #include "curand_normal_static.h"
// CHECK-NOT: #include "curand_philox4x32_x.h"
// CHECK-NOT: #include "curand_poisson.h"
// CHECK-NOT: #include "curand_precalc.h"
// CHECK-NOT: #include "curand_uniform.h"

// CHECK: #include <string>

// CHECK: #include "hipfft.h"
// CHECK: #include "hipsparse.h"

#include <cuda.h>

#include <memory>

#include <cuda_runtime.h>

#include "cuda_runtime_api.h"
#include "channel_descriptor.h"
#include "device_functions.h"
#include "driver_types.h"
#include "cuComplex.h"
#include "cuda_fp16.h"
#include "cuda_texture_types.h"
#include "vector_types.h"

#include <iostream>

#include "cublas_v2.h"
#include "cublas.h"

#include <stdio.h>

#include "curand.h"
#include "curand_kernel.h"

#include <algorithm>

#include "curand_discrete.h"
#include "curand_discrete2.h"
#include "curand_globals.h"
#include "curand_lognormal.h"
#include "curand_mrg32k3a.h"
#include "curand_mtgp32.h"
#include "curand_mtgp32_host.h"
#include "curand_mtgp32_kernel.h"
#include "curand_mtgp32dc_p_11213.h"
#include "curand_normal.h"
#include "curand_normal_static.h"
#include "curand_philox4x32_x.h"
#include "curand_poisson.h"
#include "curand_precalc.h"
#include "curand_uniform.h"

#include <string>

#include "cufft.h"

#include "cusparse.h"

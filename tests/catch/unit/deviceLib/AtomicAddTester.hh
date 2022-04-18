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

#pragma once

#include<hip_test_common.hh>
#include<hip_test_checkers.hh>
#include<hip_test_isavalidation.hh>
#include <filesystem>


#define INC_VAL 10
#define INITIAL_VAL 5

static __global__ void AtomicCheckFloat(float * Ad, float * result) {
  float inc_val = INC_VAL;
  *result = atomicAdd(Ad, inc_val);
}

static __global__ void AtomicCheckDouble(double * Ad, double * result) {
  double inc_val = INC_VAL;
  *result = atomicAdd(Ad, inc_val);
}

static __global__ void UnsafeAtomicCheckFloat(float * Ad, float * result) {
  float inc_val = INC_VAL;
  *result = unsafeAtomicAdd(Ad, inc_val);
}

static __global__ void UnsafeAtomicCheckDouble(double * Ad, double * result) {
  double inc_val = INC_VAL;
  *result = unsafeAtomicAdd(Ad, inc_val);
}

static __global__ void SafeAtomicCheckFloat(float * Ad, float * result) {
  float inc_val = INC_VAL;
  *result = safeAtomicAdd(Ad, inc_val);
}

static __global__ void SafeAtomicCheckDouble(double * Ad, double * result) {
  double inc_val = INC_VAL;
  *result = safeAtomicAdd(Ad, inc_val);
}

__attribute__((used))
__device__
void FlatAtomicCheckFloat(float * Ad, float * result) {
  float inc_val = INC_VAL;
  *result = atomicAdd(Ad, inc_val);
}

__attribute__((used))
__device__
void FlatAtomicCheckDouble(double * Ad, double * result) {
  double inc_val = INC_VAL;
  *result = atomicAdd(Ad, inc_val);
}

static std::string get_isa_name(std::string filename) {
  // testtype, coherent, unsafe, flag
  std::filesystem::path path(filename);
  path = path.filename();
  std::string sname = path.string();
  // remove ".cc"
  sname = sname.substr(0, sname.size()-3);
  // add ISA checker
  sname += "-hip-amdgcn(.*)\\.s";
  // and word boundary up front
  sname = "\\W" + sname;
  return sname;
}

enum atomicSafety : int {
  none = 0,
  safe = 1,
  unsafe = 2
};

template<typename TestType, bool coherent, atomicSafety safety, bool flag,
         bool global=true>
static void run(std::string filename) {
  filename = get_isa_name(filename);
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);
  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    if (prop.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      TestType *A_h{nullptr}, *result{nullptr};
      TestType *A_d{nullptr}, *result_d{nullptr};
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(TestType),
                              coherent ? hipHostMallocCoherent : hipHostMallocNonCoherent));
      A_h[0] = INITIAL_VAL;
      HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result),
                              sizeof(TestType),
                              coherent ? hipHostMallocCoherent : hipHostMallocNonCoherent));
      result[0] = INITIAL_VAL;
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d),
                                        A_h, 0));
      HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d),
                                        result, 0));

      // when the compiler can prove we are in global, it can use
      // just the global add
      std::vector<std::string> unsafe_f32;
      if (global) unsafe_f32 = {"global_atomic_add_f32"};
      else unsafe_f32 = {"flat_store_dword", "global_atomic_add_f32", "ds_add_rtn_f32"};
      std::vector<std::string> safe_f32 = {"global_atomic_cmpswap"};
      // it does not do this for flat, as global < flat
      std::vector<std::string> unsafe_f64 = {"flat_atomic_add_f64"};
      std::vector<std::string> safe_f64 = {"global_atomic_cmpswap_x2"};

      typedef void (*pkern)(TestType *, TestType *);
      pkern knl;
      std::string kname;
      std::vector<std::string> check;
      if (safety == atomicSafety::unsafe) {
        if constexpr (std::is_same<TestType, float>::value) {
          knl = UnsafeAtomicCheckFloat;
          kname = "UnsafeAtomicCheckFloat";
          check = unsafe_f32;
        } else {
          knl = UnsafeAtomicCheckDouble;
          kname = "UnsafeAtomicCheckDouble";
          check = unsafe_f64;
        }
      } else if (safety == atomicSafety::none) {
        if constexpr (std::is_same<TestType, float>::value) {
          knl = AtomicCheckFloat;
          kname = "AtomicCheckFloat";
          check = flag ? unsafe_f32 : safe_f32;
        } else {
          knl = AtomicCheckDouble;
          kname = "AtomicCheckDouble";
          check = flag ? unsafe_f64 : safe_f64;
        }
      } else {
        // (safety == atomicSafety::safe)
        if constexpr (std::is_same<TestType, float>::value) {
          knl = SafeAtomicCheckFloat;
          kname = "SafeAtomicCheckFloat";
          check = safe_f32;
        } else {
          knl = SafeAtomicCheckDouble;
          kname = "SafeAtomicCheckDouble";
          check = safe_f64;
        }
      }

      if (!global) {
        // mark flat, don't run "kernel"
        kname = "Flat" + kname;
      } else {
        // only run kernels
        hipLaunchKernelGGL(knl,
                           dim3(1), dim3(1),
                           0, 0, A_d,
                           result_d);
        HIP_CHECK(hipDeviceSynchronize());
        if (coherent && (safety == atomicSafety::unsafe || (
            flag && safety != atomicSafety::safe))){
          // expect this to fail
          REQUIRE(A_h[0] != INITIAL_VAL + INC_VAL);
          REQUIRE(result[0] != INITIAL_VAL);
        } else {
          // should pass
          REQUIRE(A_h[0] == INITIAL_VAL + INC_VAL);
          REQUIRE(result[0] == INITIAL_VAL);
        }
      }
      bool isaresult = HipTest::assemblyFile_Verification(filename, check, kname);
      REQUIRE(isaresult);
      HIP_CHECK(hipHostFree(A_h));
      HIP_CHECK(hipHostFree(result));
    }
  } else {
    WARN("Memory model feature is only supported for gfx90a, Hence"
         "skipping the testcase for this GPU " << device);
  }
}

/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
This testcase verifies the hipModuleLoad API On
1. Single code object
2. Multi Target architecture code object
*/
#include <fstream>
#include "hip_test_common.hh"
#include "hip_test_checkers.hh"
#ifdef __linux__
#include <unistd.h>
#endif
#define LEN 64
#define SIZE (LEN << 2)
#define COMMAND_LEN 256
#define CODE_OBJ_SINGLEARCH "module_kernels.code"
#define kernel_name "hello_world"
#define CODE_OBJ_MULTIARCH "vcpy_kernel_multarch.code"

/*
This API loads the kernel function, Launches the kernel
which copies one variable to another and validates both
the device variables for the current GPU architecture
*/
void testCodeObjFile(const char *codeObjFile) {
  float *A, *B;
  float *Ad, *Bd;
  HipTest::initArrays<float>(&Ad, &Bd, nullptr,
                             &A, &B, nullptr, LEN, false);

  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(Ad), A, SIZE));
  HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(Bd), B, SIZE));

  hipModule_t Module;
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, codeObjFile));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  struct {
    void* _Ad;
    void* _Bd;
  } args;
  args._Ad = reinterpret_cast<void*>(Ad);
  args._Bd = reinterpret_cast<void*>(Bd);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0,
                                 stream, NULL,
                                 reinterpret_cast<void**>(&config)));

  HIP_CHECK(hipStreamDestroy(stream));

  HIP_CHECK(hipMemcpyDtoH(B, reinterpret_cast<hipDeviceptr_t>(Bd), SIZE));

  for (uint32_t i = 0; i < LEN; i++) {
    REQUIRE(A[i] == B[i]);
  }

  HipTest::freeArrays<float>(Ad, Bd, nullptr,
                             A, B, nullptr,
                             false);
  HIP_CHECK(hipModuleUnload(Module));
}

#ifdef __linux__
/**
 * Check if environment variable $ROCM_PATH is defined
 *
 */
bool isRocmPathSet() {
  FILE *fpipe;
  char const *command = "echo $ROCM_PATH";
  fpipe = popen(command, "r");

  if (fpipe == nullptr) {
    WARN("Unable to create command");
    return false;
  }
  char command_op[COMMAND_LEN];
  if (fgets(command_op, COMMAND_LEN, fpipe)) {
    size_t len = strlen(command_op);
    if (len > 1) {  // This is because fgets always adds newline character
      pclose(fpipe);
      return true;
    }
  }
  pclose(fpipe);
  return false;
}
#endif
/*
This testcase checks the hipModuleLoadData API for the
current GPU architecture.
*/
TEST_CASE("Unit_hipModule_TestCodeObjFile") {
  testCodeObjFile(CODE_OBJ_SINGLEARCH);
}

/*
This testcases
1. Creates kernel file and copies to tmp folder
2. Checks for Rocm path and generates code file for
   multiple target architectures.
*/
TEST_CASE("Unit_hipModule_MultiTargArchCodeObj") {
#ifdef __linux__
  char command[COMMAND_LEN];
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  // Hardcoding the codeobject lines in multiple string to avoid cpplint warning
  std::string CodeObjL1 = "#include \"hip/hip_runtime.h\"\n";
  std::string CodeObjL2 =
       "extern \"C\" __global__ void hello_world(float* a, float* b) {\n";
  std::string CodeObjL3 = "  int tx = hipThreadIdx_x;\n";
  std::string CodeObjL4 = "  b[tx] = a[tx];\n";
  std::string CodeObjL5 = "}";
  // Creating the full code object string
  static std::string CodeObj = CodeObjL1 + CodeObjL2 + CodeObjL3 +
                               CodeObjL4 + CodeObjL5;
  std::ofstream ofs("/tmp/vcpy_kernel.cpp", std::ofstream::out);
  ofs << CodeObj;
  ofs.close();
  // Copy the file into current working location if not available
  if (access("/tmp/vcpy_kernel.cpp", F_OK) == -1) {
    INFO("Code Object File: /tmp/vcpy_kernel.cpp not found");
    REQUIRE(true);
  }
  // Generate the command to generate multi architecture code object file
  const char* hipcc_path = nullptr;
  if (isRocmPathSet()) {
    hipcc_path = "$ROCM_PATH/bin/hipcc";
  } else {
    hipcc_path = "/opt/rocm/bin/hipcc";
  }
  /* Putting these command parameters into a variable to shorten the string
    literal length in order to avoid multiline string literal cpplint warning
  */
  const char* genco_option = "--offload-arch";
  const char* input_codeobj = "/tmp/vcpy_kernel.cpp";
  snprintf(command, COMMAND_LEN,
  "%s --genco %s=gfx801,gfx802,gfx803,gfx900,gfx908,gfx1030,gfx90a,%s %s -o %s",
  hipcc_path, genco_option, props.gcnArchName, input_codeobj,
  CODE_OBJ_MULTIARCH);

  system((const char*)command);
  // Check if the code object file is created
  snprintf(command, COMMAND_LEN, "./%s",
           CODE_OBJ_MULTIARCH);

  if (access(command, F_OK) == -1) {
    INFO("Code Object File not found");
    REQUIRE(true);
  } else {
    testCodeObjFile(CODE_OBJ_MULTIARCH);
  }
#else
  SUCCEED("This test is skipped due to non linux environment");
#endif
}

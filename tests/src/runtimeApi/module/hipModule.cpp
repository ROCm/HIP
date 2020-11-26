/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD_CMD: vcpy_kernel.code %hc --genco %S/vcpy_kernel.cpp -o vcpy_kernel.code
 * BUILD: %t %s ../../test_common.cpp NVCC_OPTIONS -std=c++11
 * TEST: %t --tests 0x1
 * TEST: %t --tests 0x2 EXCLUDE_HIP_PLATFORM nvidia
 * HIT_END
 */

#include <iostream>
#include <fstream>
#ifdef __linux__
#include <unistd.h>
#endif
#include "test_common.h"

#define LEN 64
#define SIZE (LEN << 2)
#define COMMAND_LEN 256
#define CODE_OBJ_SINGLEARCH "vcpy_kernel.code"
#define kernel_name "hello_world"
#define CODE_OBJ_MULTIARCH "vcpy_kernel_multarch.code"

bool testCodeObjFile(const char *codeObjFile) {
  float *A, *B;
  hipDeviceptr_t Ad, Bd;
  A = new float[LEN];
  B = new float[LEN];

  for (uint32_t i = 0; i < LEN; i++) {
    A[i] = i * 1.0f;
    B[i] = 0.0f;
  }

  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Ad), SIZE));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&Bd), SIZE));
  HIPCHECK(hipMemcpyHtoD(Ad, A, SIZE));
  HIPCHECK(hipMemcpyHtoD(Bd, B, SIZE));

  hipModule_t Module;
  hipFunction_t Function;
  HIPCHECK(hipModuleLoad(&Module, codeObjFile));
  HIPCHECK(hipModuleGetFunction(&Function, Module, kernel_name));

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

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
  HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0,
                                 stream, NULL,
                                 reinterpret_cast<void**>(&config)));

  HIPCHECK(hipStreamDestroy(stream));

  HIPCHECK(hipMemcpyDtoH(B, Bd, SIZE));

  bool btestPassed = true;
  for (uint32_t i = 0; i < LEN; i++) {
    if (A[i] != B[i]) {
      btestPassed = false;
      break;
    }
  }
  HIPCHECK(hipFree(reinterpret_cast<void*>(Bd)));
  HIPCHECK(hipFree(reinterpret_cast<void*>(Ad)));
  delete[] B;
  delete[] A;
  HIPCHECK(hipModuleUnload(Module));
  return btestPassed;
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
    printf("Unable to create command\n");
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

bool testMultiTargArchCodeObj() {
  bool btestPassed = true;
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
    printf("Code Object File: /tmp/vcpy_kernel.cpp not found \n");
    return true;
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
  "%s --genco %s=gfx801,gfx802,gfx803,gfx900,gfx908,%s %s -o %s",
  hipcc_path, genco_option, props.gcnArchName, input_codeobj,
  CODE_OBJ_MULTIARCH);

  printf("command = %s\n", command);
  system((const char*)command);
  // Check if the code object file is created
  snprintf(command, COMMAND_LEN, "./%s",
           CODE_OBJ_MULTIARCH);

  if (access(command, F_OK) == -1) {
    printf("Code Object File not found \n");
    return true;
  }
  btestPassed = testCodeObjFile(CODE_OBJ_MULTIARCH);
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
  return btestPassed;
}

int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = true;
  if (p_tests == 0x1) {
    /* In this test scenario a code object file for the current
    GPU architecture is generated, loaded and executed. */
    TestPassed = testCodeObjFile(CODE_OBJ_SINGLEARCH);
  } else if (p_tests == 0x2) {
    /* In this test scenario a code object file for the multiple
    GPU architectures (including the current) is generated, loaded
    and executed. */
    TestPassed = testMultiTargArchCodeObj();
  } else {
    printf("Invalid Test Case \n");
    exit(1);
  }
  if (TestPassed) {
    passed();
  } else {
    failed("Test Case %x Failed!", p_tests);
  }
}

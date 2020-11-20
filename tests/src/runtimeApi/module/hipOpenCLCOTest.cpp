/*
Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
 * BUILD: %t %s ../../test_common.cpp EXCLUDE_HIP_PLATFORM nvidia
 * TEST: %t --tests 0x1
 * TEST: %t --tests 0x2
 * HIT_END
 */
#ifdef __linux__
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "test_common.h"

#define OPENCL_OBJ_FILE "opencl_add.cpp"
#define HIP_CODEOBJ_FILE_DEFAULT "opencl_add.co"
#define HIP_CODEOBJ_FILE_V3 "opencl_add_v3.co"
#define COMMAND_LEN 256
#define BUFFER_LEN 256
/**
 * Validates OpenCL Static Lds Code Object
 *
 */
bool testStaticLdsCodeObj(const char* pCodeObjFile) {
  hipDevice_t device;
  hipModule_t Module;
  hipFunction_t Function;
  printf("Executing %s \n", __func__);
  HIPCHECK(hipDeviceGet(&device, 0));
  HIPCHECK(hipModuleLoad(&Module, pCodeObjFile));
  HIPCHECK(hipModuleGetFunction(&Function, Module, "add"));

  float *Ah, *Bh;
  Ah = new float[BUFFER_LEN];
  Bh = new float[BUFFER_LEN];
  for (uint32_t i = 0; i < BUFFER_LEN; i++) {
    Ah[i] = i * 1.0f;
    Bh[i] = 0.0f;
  }

  float *Ad, *Bd;
  HIPCHECK(hipMalloc(&Ad, sizeof(float) * BUFFER_LEN));
  HIPCHECK(hipMalloc(&Bd, sizeof(float) * BUFFER_LEN));
  HIPCHECK(hipMemcpy(Ad, Ah, sizeof(float) * BUFFER_LEN,
                     hipMemcpyHostToDevice));

  struct {
    void* _Bd;
    void* _Ad;
  } args;
  args._Ad = static_cast<void*>(Ad);
  args._Bd = static_cast<void*>(Bd);
  size_t size = sizeof(args);

  void *config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
    HIP_LAUNCH_PARAM_END
  };

  HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1, BUFFER_LEN, 1, 1, 0, 0,
                        NULL, reinterpret_cast<void**>(&config)));
  HIPCHECK(hipMemcpy(Bh, Bd, sizeof(float) * BUFFER_LEN,
                     hipMemcpyDeviceToHost));

  bool TestPassed = true;
  for (uint32_t i = 0; i < BUFFER_LEN; i++) {
    if (Ah[i] != Bh[i]) {
      TestPassed = false;
      break;
    }
  }
  hipFree(Ad);
  hipFree(Bd);
  delete[] Ah;
  delete[] Bh;
  return TestPassed;
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
  char command_op[BUFFER_LEN];
  if (fgets(command_op, BUFFER_LEN, fpipe)) {
    size_t len = strlen(command_op);
    if (len > 1) {  // This is because fgets always adds newline character
      pclose(fpipe);
      return true;
    }
  }
  pclose(fpipe);
  return false;
}
/**
 * Gets the sramecc/xnack settings from rocm info
 *
 */
int getV3TargetIdFeature(char* feature, bool rocmPathSet) {
  FILE *fpipe;
  char command[COMMAND_LEN] = "";
  const char *rocmpath = nullptr;
  if (rocmPathSet) {
    // For STG2 testing where /opt/rocm path is not present
    rocmpath = "$ROCM_PATH/bin/rocminfo";
  } else {
    // Check if the rocminfo tool exists
    rocmpath = "/opt/rocm/bin/rocminfo";
  }
  snprintf(command, COMMAND_LEN, "%s", rocmpath);
  strncat(command, " | grep -m1 \"sramecc.:xnack.\"", COMMAND_LEN);
  fpipe = popen(command, "r");

  if (fpipe == nullptr) {
    printf("Unable to create command file\n");
    return -1;
  }
  char command_op[BUFFER_LEN];
  const char* pOpt1 = nullptr;
  const char *pOpt2 = nullptr;
  if (fgets(command_op, BUFFER_LEN, fpipe)) {
    if (strstr(command_op, "sramecc+")) {
      pOpt1 = "-msram-ecc";
    } else if (strstr(command_op, "sramecc-")) {
      pOpt1 = "-mno-sram-ecc";
    } else {
      pclose(fpipe);
      return -1;
    }
    if (strstr(command_op, "xnack+")) {
      pOpt2 = " -mxnack";
    } else if (strstr(command_op, "xnack-")) {
      pOpt2 = " -mno-xnack";
    } else {
      pclose(fpipe);
      return -1;
    }
  } else {
    printf("No sramecc/xnack settings found.\n");
    pclose(fpipe);
    return -1;
  }
  strncpy(feature, pOpt1, strlen(pOpt1));
  strncat(feature, pOpt2, strlen(pOpt2));
  pclose(fpipe);
  return 0;
}
#endif

int main(int argc, char* argv[]) {
  HipTest::parseStandardArguments(argc, argv, true);
  bool TestPassed = true;
  int version = HIP_VERSION_MAJOR;
#ifdef __linux__
  char command[COMMAND_LEN] = "";
  char v3option[32] = "";
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  if (access("./opencl_add.cpp", F_OK) == -1) {
    system("cp ../tests/src/runtimeApi/module/opencl_add.cpp .");
  }
  // Generate the command to translate the OpenCL code object to hip code object
  const char *pCodeObjVer = nullptr;
  const char *pCodeObjFile = nullptr;
  bool rocmPathSet = isRocmPathSet();
  if (p_tests == 0x1) {
    pCodeObjVer = "";
    pCodeObjFile = HIP_CODEOBJ_FILE_DEFAULT;
  } else if ((p_tests == 0x2) && (version >= 4)) {
    pCodeObjVer = "-mcode-object-version=3";
    if (-1 == getV3TargetIdFeature(v3option, rocmPathSet)) {
      printf("Error getting V3 Option. Skipping Test. \n");
      passed();
    }
    pCodeObjFile = HIP_CODEOBJ_FILE_V3;
  } else {
    printf("Invalid Test Case \n");
    passed();
  }
  printf("v3option = %s\n", v3option);
  /* The command string is created using multiple concatenation instead of one go
    to avoid the following cpplint error:
    " Multi-line string ("...") found.  This lint script doesn't do well with such strings,
    and may give bogus warnings.  Use C++11 raw strings or concatenation instead."
  */
  if (rocmPathSet) {
    // For STG2 testing where /opt/rocm path is not present
    snprintf(command, COMMAND_LEN,
    "$ROCM_PATH/llvm/bin/clang -target amdgcn-amd-amdhsa -x cl ");
  } else {
    snprintf(command, COMMAND_LEN,
    "/opt/rocm/llvm/bin/clang -target amdgcn-amd-amdhsa -x cl ");
  }
  char command_temp[COMMAND_LEN] = "";
  snprintf(command_temp, COMMAND_LEN,
  "-include `find /opt/rocm* -name opencl-c.h` %s %s -mcpu=%s -o %s %s",
  pCodeObjVer, v3option, props.gcnArchName, pCodeObjFile, OPENCL_OBJ_FILE);

  strncat(command, command_temp, COMMAND_LEN);
  printf("command executed = %s\n", command);

  system((const char*)command);
  // Check if the code object file is created
  snprintf(command, COMMAND_LEN, "./%s",
           pCodeObjFile);

  if (access(command, F_OK) == -1) {
    printf("Code Object File not found \n");
    passed();
  }
  TestPassed = testStaticLdsCodeObj(pCodeObjFile);
  if (TestPassed) {
    passed();
  } else {
    failed("Test Case %x Failed!", p_tests);
  }
#else
  printf("This test is skipped due to non linux environment.\n");
  passed();
#endif
}

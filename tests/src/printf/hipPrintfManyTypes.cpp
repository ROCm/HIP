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

#include "hipPrintfUtil.h"
#include "../test_common.h"

__global__ void test_kernel_int(const char* format, int val) {
  printf(format, val);
}


__global__ void test_kernel_float(const char* format, float val) {
  printf(format, val);
}


__global__ void test_kernel_float_limits(int TestNum) {
  if (TestNum == 0) {
    printf("%f", 1.0f/0.0f);
  }
  if (TestNum == 1) {
    printf("%f", sqrt(-1.0f));
  }
  if (TestNum == 2) {
    printf("%f", acos(2.0f));
  }
}

__global__ void test_kernel_octal(const char* format, const unsigned long int val) {
  printf(format, val);
}

__global__ void test_kernel_unsigned(const char* format, const unsigned long int val) {
  printf(format, val);
}

__global__ void test_kernel_char(int TestNum) {
  if (TestNum == 0) {
    printf("%4c", '1');
  }
  if (TestNum == 1) {
    printf("%-4c", '1');
  }
  if (TestNum == 2) {
    printf("%c", 66);
  }
}

__global__ void test_kernel_string(int TestNum) {
  if (TestNum == 0) {
    printf("%4s", "foo");
  }
  if (TestNum == 1) {
    printf("%.1s", "foo");
  }
  if (TestNum == 2) {
    printf("%s","%%");
  }
}

void BuildCorrectOutput(std::vector<std::string>* CorrectBuff, size_t TestId) {
  for (auto &params:allTestCase[TestId]->_genParameters) {
    char refResult[256];
    if (allTestCase[TestId]->printFN == NULL)
      continue;
    allTestCase[TestId]->printFN(params, refResult, 256);
    CorrectBuff->push_back(refResult);
  }
}

bool TestKernel(const unsigned int TestId, const unsigned int TestNum) {

  CaptureStream capture(stdout);

  capture.Begin();

  switch (allTestCase[TestId]->_type) {
    case TYPE_INT: {
      const char* Format;
      size_t FormatLen = std::strlen(allTestCase[TestId]->_genParameters[TestNum].genericFormat);
      hipMalloc((void**)&Format, sizeof(char)*FormatLen);
      hipMemcpyHtoD((void*)Format, (void*)allTestCase[TestId]->_genParameters[TestNum].genericFormat , sizeof(char)*FormatLen);
      hipLaunchKernelGGL(test_kernel_int, dim3(1), dim3(1), 0, 0, Format,
                         atoi(allTestCase[TestId]->_genParameters[TestNum].dataRepresentation));
      break;
    }
    case TYPE_FLOAT: {
      const char* Format;
      size_t FormatLen = std::strlen(allTestCase[TestId]->_genParameters[TestNum].genericFormat);
      hipMalloc((void**)&Format, sizeof(char)*FormatLen);
      hipMemcpyHtoD((void*)Format, (void*)allTestCase[TestId]->_genParameters[TestNum].genericFormat , sizeof(char)*FormatLen);
      float val = strtof(allTestCase[TestId]->_genParameters[TestNum].dataRepresentation, NULL);
      hipLaunchKernelGGL(test_kernel_float, dim3(1), dim3(1), 0, 0, Format, val);
      break;
    }
    case TYPE_FLOAT_LIMITS: {
      hipLaunchKernelGGL(test_kernel_float_limits, dim3(1), dim3(1), 0, 0, TestNum);
      break;
    }
    case TYPE_OCTAL: {
      const char* Format;
      size_t FormatLen = std::strlen(allTestCase[TestId]->_genParameters[TestNum].genericFormat);
      hipMalloc((void**)&Format, sizeof(char)*FormatLen);
      hipMemcpyHtoD((void*)Format, (void*)allTestCase[TestId]->_genParameters[TestNum].genericFormat , sizeof(char)*FormatLen);
      const unsigned long int data = strtoul(allTestCase[TestId]->_genParameters[TestNum].dataRepresentation, NULL, 10);
      hipLaunchKernelGGL(test_kernel_octal, dim3(1), dim3(1), 0, 0, Format, data);
      break;
    }
    case TYPE_UNSIGNED: {
      const char* Format;
      size_t FormatLen = std::strlen(allTestCase[TestId]->_genParameters[TestNum].genericFormat);
      hipMalloc((void**)&Format, sizeof(char)*FormatLen);
      hipMemcpyHtoD((void*)Format, (void*)allTestCase[TestId]->_genParameters[TestNum].genericFormat , sizeof(char)*FormatLen);
      const unsigned long int data = strtoul(allTestCase[TestId]->_genParameters[TestNum].dataRepresentation, NULL, 10);
      hipLaunchKernelGGL(test_kernel_unsigned, dim3(1), dim3(1), 0, 0, Format, data);
      break;
    }
    case TYPE_HEXADEC: {
      const char* Format;
      size_t FormatLen = std::strlen(allTestCase[TestId]->_genParameters[TestNum].genericFormat);
      hipMalloc((void**)&Format, sizeof(char)*FormatLen);
      hipMemcpyHtoD((void*)Format, (void*)allTestCase[TestId]->_genParameters[TestNum].genericFormat , sizeof(char)*FormatLen);
      const unsigned long int data = strtoul(allTestCase[TestId]->_genParameters[TestNum].dataRepresentation, NULL, 0);
      hipLaunchKernelGGL(test_kernel_unsigned, dim3(1), dim3(1), 0, 0, Format, data);
      break;
    }
    case TYPE_CHAR: {
      hipLaunchKernelGGL(test_kernel_char, dim3(1), dim3(1), 0, 0, TestNum);
      break;
    }
    case TYPE_STRING: {
      hipLaunchKernelGGL(test_kernel_string, dim3(1), dim3(1), 0, 0, TestNum);
      break;
    }
    default: {
      return false;
    }
  }

  hipDeviceSynchronize();
  capture.End();
  std::string device_output = capture.getData();
  char* exp;
  //Exponenent representation
  if ((exp = std::strstr((char*)device_output.c_str(),"E+")) != NULL || (exp = std::strstr((char*)device_output.c_str(),"e+")) != NULL
     || (exp = std::strstr((char*)device_output.c_str(),"E-")) != NULL || (exp = std::strstr((char*)device_output.c_str(),"e-")) != NULL) {

    char correctExp[3]={0};
    std::strncpy(correctExp,exp,2);

    char* eCorrectBuffer = strstr((char*)allTestCase[TestId]->_correctBuffer[TestNum].c_str(),correctExp);
    if (eCorrectBuffer == NULL) {
      return false;
    }
    eCorrectBuffer+=2;
    exp += 2;
    //Exponent always contains at least two digits
    if (strlen(exp) < 2) {
      return false;
    }
    //Skip leading zeros in the exponent
    while (*exp == '0') {
      ++exp;
    }
    while (*eCorrectBuffer == '0') {
      ++eCorrectBuffer;
    }
    if (std::strcmp(eCorrectBuffer,exp)) {
      return false;
    }
  }
  if (!std::strcmp(allTestCase[TestId]->_correctBuffer[TestNum].c_str(),"inf")) {
    if (!std::strcmp(device_output.c_str(),"inf")||!std::strcmp(device_output.c_str(),"infinity")
        || !std::strcmp(device_output.c_str(),"1.#INF00")&&std::strcmp(device_output.c_str(),"Inf")) {
             return true;
      }
  }
  if (!std::strcmp(allTestCase[TestId]->_correctBuffer[TestNum].c_str(),"nan")
       || !std::strcmp(allTestCase[TestId]->_correctBuffer[TestNum].c_str(),"-nan")) {
    if (!std::strcmp(device_output.c_str(),"nan")||!std::strcmp(device_output.c_str(),"-nan")
        || !std::strcmp(device_output.c_str(),"1.#IND00")||!std::strcmp(device_output.c_str(),"-1.#IND00")
        || !std::strcmp(device_output.c_str(),"NaN")||!std::strcmp(device_output.c_str(),"nan(ind)")
        || !std::strcmp(device_output.c_str(),"nan(snan)")||!std::strcmp(device_output.c_str(),"-nan(ind)")) {
      return true;
      }
  }
  std::cout<<device_output<<std::endl;
  std::cout<<allTestCase[TestId]->_correctBuffer[TestNum]<<std::endl;
  fflush(stdout);
  if (std::strcmp(device_output.c_str(), allTestCase[TestId]->_correctBuffer[TestNum].c_str())) {
    return false;
  }
  return true;
}

#define MAX_TYPES 8
int main(int argc, char **argv) {

  const char* arg = argv[1];
  bool TestpPass;
  if (!std::strcmp(arg,"--All") || !std::strcmp(arg," ")) {
      for (int i=0; i < TYPE_COUNT; i++) {
        BuildCorrectOutput(&allTestCase[i]->_correctBuffer, i);
        for (int j=0; j < allTestCase[i]->numOfTests; j++) {
          TestpPass = TestKernel(i,j);
          HIPASSERT(TestpPass == true);
        }
      }
  } else {
    char* input[] = {"--INT", "--FLOAT", "--FLOAT_LIMITS", "--OCTAL",
                     "--UNSIGNED", "--HEXADEC", "--CHAR", "--STRING"};
    for (int i=0; i < MAX_TYPES; i++) {
      if (!std::strcmp(arg, input[i])) {
        BuildCorrectOutput(&allTestCase[i]->_correctBuffer, i);
        for (int j=0; j < allTestCase[i]->numOfTests; j++) {
          TestpPass = TestKernel(i,j);
          HIPASSERT(TestpPass == true);
        }
      }
    }
  }
  passed();
}

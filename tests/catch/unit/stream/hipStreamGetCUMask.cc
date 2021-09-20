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

/**
Testcase Scenarios :
1) Test to verify hipExtStreamGetCUMask api returning default CU Mask or global CU Mask.
2) Test to verify hipExtStreamGetCUMask api returns custom mask set.
3) Negative tests for hipExtStreamGetCUMask api.
*/

#include <hip_test_common.hh>
#include <vector>


/**
 * Scenario to verify hipExtStreamGetCUMask api returning default CU Mask or global CU Mask.
 * Scenario to verify hipExtStreamGetCUMask api returns custom mask set.
 */
TEST_CASE("Unit_hipExtStreamGetCUMask_verifyDefaultAndCustomMask") {
  constexpr int maxNum = 6;
  std::vector<uint32_t> cuMask(maxNum);
  hipDeviceProp_t props;
  std::stringstream ss;
  char* gCUMask{nullptr};
  std::string globalCUMask("");
  std::vector<uint32_t> defaultCUMask;

  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    INFO("info: didn't find any GPU! skipping the test!");
    return;
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  INFO("info: running on bus " << "0x" << props.pciBusID << " " <<
         props.name << " with " << props.multiProcessorCount << " CUs");

  // Get global CU Mask if exists
  gCUMask = getenv("ROC_GLOBAL_CU_MASK");
  if (gCUMask != nullptr && gCUMask[0] != '\0') {
    globalCUMask.assign(gCUMask);

    for_each(globalCUMask.begin(), globalCUMask.end(), [](char & c) {
      c = ::tolower(c);
    });
  }

  // Create default CU Mask
  uint32_t temp = 0;
  uint32_t bit_index = 0;
  for (uint32_t i = 0; i < (uint32_t)props.multiProcessorCount; i++) {
    temp |= 1UL << bit_index;
    if (bit_index >= 32) {
      defaultCUMask.push_back(temp);
      temp = 0;
      bit_index = 0;
      temp |= 1UL << bit_index;
    }
    bit_index += 1;
  }
  if (bit_index != 0) {
    defaultCUMask.push_back(temp);
  }

  SECTION("Verify with default CU Mask or global CU Mask") {
    // make a default CU mask bit-array where all CUs are active
    // this default mask is expected to be returned when there is no
    // custom or global CU mask defined

    HIP_CHECK(hipExtStreamGetCUMask(0, cuMask.size(), &cuMask[0]));

    ss << std::hex;
    for (int i = cuMask.size() - 1; i >= 0; i--) {
      ss << cuMask[i];
    }

    // remove extra 0 from ss if any present
    size_t found = ss.str().find_first_not_of("0");
    if (found != std::string::npos) {
      ss.str(ss.str().substr(found, ss.str().length()));
    }

    INFO("info: CU mask for the default stream is: 0x" << ss.str().c_str());
    if (globalCUMask.size() > 0) {
      if (ss.str().compare(globalCUMask) != 0) {
        INFO("Expected CU mask:" << globalCUMask.c_str() <<
                                     ", api returned:" << ss.str().c_str());
        REQUIRE(false);
      }
    } else {
      for (int i = 0 ; i < min(cuMask.size(), defaultCUMask.size()); i++) {
        if (cuMask[i] != defaultCUMask[i]) {
          INFO("Expected CU mask " << defaultCUMask[i] <<
                                      ", api returned:" << cuMask[i]);
          REQUIRE(false);
        }
      }
    }
  }

  SECTION("Verify with custom mask set") {
    std::vector<uint32_t> customMask(defaultCUMask);
    hipStream_t stream;
    customMask[0] = 0xe;

    HIP_CHECK(hipExtStreamCreateWithCUMask(&stream, customMask.size(),
                                                      customMask.data()));
    ss.str("");
    for (int i = customMask.size() - 1; i >= 0; i--) {
      ss << customMask[i];
    }
    INFO("info: setting a custom CU mask 0x" << ss.str());

    HIP_CHECK(hipExtStreamGetCUMask(stream, cuMask.size(), &cuMask[0]));
    ss.str("");
    for (int i = cuMask.size() - 1; i >= 0; i--) {
      ss << cuMask[i];
    }

    size_t found = ss.str().find_first_not_of("0");
    if (found != std::string::npos) {
      ss.str(ss.str().substr(found, ss.str().length()));
    }

    INFO("info: reading back CU mask 0x" << ss.str() <<
                                                " for stream " << stream);

    if (!gCUMask) {
      for (size_t i = 0; i < customMask.size(); i++) {
        if (customMask[i] != cuMask[i]) {
          INFO("Error! expected CU mask:" << customMask[i]
                            << ", Received CU mask:" << cuMask[i]);
          REQUIRE(false);
        }
      }
    }

    HIP_CHECK(hipStreamDestroy(stream));
  }
}

/**
 * Negative tests for hipExtStreamGetCUMask.
 */
TEST_CASE("Unit_hipExtStreamGetCUMask_Negative") {
  hipError_t ret;
  constexpr int maxNum = 6;
  std::vector<uint32_t> cuMask(maxNum);

  SECTION("cuMask is nullptr") {
    ret = hipExtStreamGetCUMask(0, cuMask.size(), nullptr);
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("cuMaskSize is 0") {
    ret = hipExtStreamGetCUMask(0, 0, &cuMask[0]);
    REQUIRE(ret == hipErrorInvalidValue);
  }
}

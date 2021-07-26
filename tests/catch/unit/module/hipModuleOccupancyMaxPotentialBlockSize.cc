
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
#include <hip_test_common.hh>
#include <limits>

#define fileName "module_kernels.code"
#define kernel_name "hello_world"
/**
 * hipModuleOccupancyMaxPotentialBlockSize and hipModuleOccupancyMaxPotentialBlockSizeWithFlags
 * corner tests.
 * Scenario1:
 * Validates the value of gridSize, which should be always non zero +ve integer and blockSize
 * range returned for dynSharedMemPerBlk = 0 and blockSizeLimit = 0.
 * Scenario2:
 * Validates the value of gridSize, which should be always non zero +ve integer and blockSize
 * range returned for dynSharedMemPerBlk = devProp.sharedMemPerBlock and
 * blockSizeLimit = devProp.maxThreadsPerBlock.
 */
TEST_CASE("Unit_hipModuleOccupancyMaxPotentialBlockSize_FuncTst") {
  // Initialize
  hipDeviceProp_t devProp;
  int gridSize = 0;
  int blockSize = 0;
  hipModule_t Module;
  CTX_CREATE()
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  // Scenario1
  SECTION("without flag - gridSize when input params are 0") {
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize,
            &blockSize, Function, 0, 0));
  }
  // Scenario2
  SECTION("without flag - gridSize when input params are maximum") {
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize,
             &blockSize, Function,
             devProp.sharedMemPerBlock, devProp.maxThreadsPerBlock));
  }
  // Scenario1
  SECTION("with flag - gridSize when input params are 0") {
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize,
            &blockSize, Function, 0, 0, 0));
  }
  // Scenario2
  SECTION("with flag - gridSize when input params are maximum") {
    HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize,
            &blockSize, Function, devProp.sharedMemPerBlock,
            devProp.maxThreadsPerBlock, 0));
  }
  // Check if blockSize doen't exceed maxThreadsPerBlock
  REQUIRE_FALSE(gridSize <= 0);
  REQUIRE_FALSE(blockSize <= 0);
  REQUIRE_FALSE(blockSize > devProp.maxThreadsPerBlock);
  // Un-initialize
  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY()
}
/**
 * hipModuleOccupancyMaxActiveBlocksPerMultiprocessor and
 * hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags Corner tests.
 * Scenario1:
 * Validates numBlock value range is within expected limit when sharedMemPerBlock
 * is 0.
 * Scenario2:
 * Validates numBlock value range is within expected limit when
 * dynSharedMemPerBlk = devProp.sharedMemPerBlock.
 */
TEST_CASE("Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_FuncTst") {
  // Initialize
  hipDeviceProp_t devProp;
  int gridSize = 0;
  int blockSize = 0;
  int numBlock = 0;
  hipModule_t Module;
  CTX_CREATE()
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize,
          &blockSize, Function, 0, 0));
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  // Scenario1
  SECTION("without flag - gridSize when input params are 0") {
    HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock,
             Function, blockSize, 0));
  }
  // Scenario2
  SECTION("without flag - gridSize when input params are maximum") {
    HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlock,
             Function, blockSize, devProp.sharedMemPerBlock));
  }
  // Scenario1
  SECTION("with flag - gridSize when input params are 0") {
    HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &numBlock, Function, blockSize, 0, 0));
  }
  // Scenario2
  SECTION("with flag - gridSize when input params are maximum") {
    HIP_CHECK(hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            &numBlock, Function, blockSize, devProp.sharedMemPerBlock, 0))
  }
  // Check if numBlocks are within limits
  int temp_val = (numBlock * blockSize);
  REQUIRE_FALSE(numBlock <= 0);
  REQUIRE_FALSE(temp_val > devProp.maxThreadsPerMultiProcessor);
  // Un-initialize
  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY()
}
/**
 * hipModuleOccupancyMaxPotentialBlockSize negative tests.
 * Scenario1: gridSize is nullptr.
 * Scenario2: blocksize is nullptr.
 * Scenario3: blockSizeLimit < 0.
 */
TEST_CASE("Unit_hipModuleOccupancyMaxPotentialBlockSize_NegTst") {
  int gridSize = 0;
  int blockSize = 0;
  hipModule_t Module;
  hipFunction_t Function;
  CTX_CREATE()
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  // Scenario1
  SECTION("without flag - gridSize is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipModuleOccupancyMaxPotentialBlockSize(
                  nullptr, &blockSize, Function, 0, 0));
  }
  // Scenario2
  SECTION("without flag - blocksize is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipModuleOccupancyMaxPotentialBlockSize(
                  &gridSize, nullptr, Function, 0, 0));
  }
  // Scenario3
  SECTION("without flag - blockSizeLimit is less than 0") {
    hipDeviceProp_t devProp;
    HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
#if HT_NVIDIA
    REQUIRE_FALSE(hipSuccess == hipModuleOccupancyMaxPotentialBlockSize(
                  &gridSize, &blockSize, Function, 0, -1));
#else
    // As discussed in SWDEV-269400
    // with developers this difference in behavior between NVIDIA and AMD
    // is retained.
    REQUIRE_FALSE(hipSuccess != hipModuleOccupancyMaxPotentialBlockSize(
                  &gridSize, &blockSize, Function, 0, -1));
#endif
  }
  // Scenario1
  SECTION("with flag - gridSize is nullptr") {
    REQUIRE_FALSE(hipSuccess ==
      hipModuleOccupancyMaxPotentialBlockSizeWithFlags(nullptr,
      &blockSize, Function, 0, 0, 0));
  }
  // Scenario2
  SECTION("with flag - blocksize is nullptr") {
    REQUIRE_FALSE(hipSuccess ==
      hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize,
      nullptr, Function, 0, 0, 0));
  }
  // Scenario3
  SECTION("with flag - blockSizeLimit is less than 0") {
#if HT_NVIDIA
    REQUIRE_FALSE(hipSuccess ==
      hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize,
      &blockSize, Function, 0, -1, 0));
#else
    // As discussed in SWDEV-269400
    // with developers this difference in behavior between NVIDIA and AMD
    // is retained.
    REQUIRE_FALSE(hipSuccess !=
      hipModuleOccupancyMaxPotentialBlockSizeWithFlags(&gridSize,
      &blockSize, Function, 0, -1, 0));
#endif
  }
  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY()
}
/**
 * hipModuleOccupancyMaxActiveBlocksPerMultiprocessor negative tests.
 * Scenario1: numBlocks is nullptr.
 * Scenario2: Check the behavior for blockSize < 0.
 * Scenario3: Check error code returned for dynSharedMemPerBlk = 0 and blockSize = 0.
 * Scenario4: dynSharedMemPerBlk = size_t numeric limit.
 */
TEST_CASE("Unit_hipModuleOccupancyMaxActiveBlocksPerMultiprocessor_NegTst") {
  int gridSize = 0;
  int blockSize = 0;
  int numBlocks = 0;
  hipModule_t Module;
  hipFunction_t Function;
  CTX_CREATE()
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  HIP_CHECK(hipModuleOccupancyMaxPotentialBlockSize(&gridSize, &blockSize,
                                                   Function, 0, 0));
  // Scenario1
  SECTION("without flag - numBlocks is nullptr") {
    REQUIRE_FALSE(hipSuccess ==
       hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(nullptr,
       Function, blockSize, 0));
  }
  // Scenario3
  SECTION("without flag - dynSharedMemPerBlk = 0 and blockSize = 0") {
    REQUIRE_FALSE(hipSuccess ==
       hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
       Function, 0, 0));
  }
  // Scenario2
  SECTION("without flag - blockSize is less than 0") {
    REQUIRE_FALSE(hipSuccess ==
       hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
       Function, -1, 0));
  }
  // Scenario4
  SECTION("without flag - dynSharedMemPerBlk = max_numerical_limit") {
    REQUIRE_FALSE(hipSuccess ==
       hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
       Function, 0, std::numeric_limits<std::size_t>::max()));
  }
  // Scenario1
  SECTION("with flag - numBlocks is nullptr") {
    REQUIRE_FALSE(hipSuccess ==
       hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(nullptr,
       Function, blockSize, 0, 0));
  }
  // Scenario3
  SECTION("with flag - dynSharedMemPerBlk = 0 and blockSize = 0") {
    REQUIRE_FALSE(hipSuccess ==
      hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks,
      Function, 0, 0, 0));
  }
  // Scenario2
  SECTION("with flag - blockSize is less than 0") {
    REQUIRE_FALSE(hipSuccess ==
      hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks,
      Function, -1, 0, 0));
  }
  // Scenario4
  SECTION("with flag - dynSharedMemPerBlk = max_numerical_limit") {
    REQUIRE_FALSE(hipSuccess ==
      hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(&numBlocks,
      Function, 0, std::numeric_limits<std::size_t>::max(), 0));
  }
  HIP_CHECK(hipModuleUnload(Module));
  CTX_DESTROY()
}


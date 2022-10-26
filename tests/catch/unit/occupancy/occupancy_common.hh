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

#include <hip_test_common.hh>

template <typename F>
void MaxPotentialBlockSize(F func, int maxThreadsPerBlock) {
  int gridSize = 0;
  int blockSize = 0;

  // Get potential blocksize
  HIP_CHECK(func(&gridSize, &blockSize));

  // Check if blockSize doesn't exceed maxThreadsPerBlock
  REQUIRE(gridSize > 0); REQUIRE(blockSize > 0);
  REQUIRE(blockSize <= maxThreadsPerBlock);
  REQUIRE(gridSize * blockSize <  static_cast<int64_t>(std::pow(2, 32)));
}

template <typename F>
void MaxPotentialBlockSizeNegative(F func) {
  int blockSize = 0;
  int gridSize = 0;

  // Validate common arguments
  SECTION("gridSize is nullptr") {
    HIP_CHECK_ERROR(func(nullptr, &blockSize), hipErrorInvalidValue);
  }
  SECTION("blockSize is nullptr") {
    HIP_CHECK_ERROR(func(&gridSize, nullptr), hipErrorInvalidValue);
  }
}

template <typename F>
void MaxActiveBlocksPerMultiprocessor(F func, int blockSize, int maxThreadsPerMultiProcessor) {
  int numBlocks = 0;

  // Validate maximum active block pre multiprocessor
  HIP_CHECK(func(&numBlocks));

  // Check if numBlocks and blockSize are within limits
  REQUIRE(numBlocks > 0);
  REQUIRE((numBlocks * blockSize) <= maxThreadsPerMultiProcessor);
}

template <typename F>
void MaxActiveBlocksPerMultiprocessorNegative(F func, int blockSize) {
  int numBlocks = 0;

  // Validate common arguments
  SECTION("numBlocks is nullptr") {
    HIP_CHECK_ERROR(func(nullptr, blockSize, 0), hipErrorInvalidValue);
  }
  SECTION("Block size is 0") {
    HIP_CHECK_ERROR(func(&numBlocks, 0, 0), hipErrorInvalidValue);
  }
}

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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include "hip_test_common.hh"

#include <string>
#include <array>
#include <cstdlib>
#include <random>
#include <fstream>
#include <streambuf>
#include "hip_test_filesystem.hh"

namespace hip {
class SpawnProc {
  std::string exeName;
  std::string resultStr;
  std::string tmpFileName;
  bool captureOutput;

  std::string getRandomString(size_t len = 6) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 25);

    std::string res;
    for (size_t i = 0; i < len; i++) {
      res += 'a' + dist(rng);
    }
    return res;
  }

 public:
  SpawnProc(std::string exeName_, bool captureOutput_ = false)
      : exeName(exeName_), captureOutput(captureOutput_) {
    auto dir = fs::path(TestContext::get().currentPath()).parent_path();
    dir /= exeName;
    exeName = dir.string();
    if (captureOutput) {
      auto path = fs::temp_directory_path();
      path /= getRandomString();
      tmpFileName = path.string();
    }
  }

  int run() {
    std::string execCmd = exeName;
    if (captureOutput) {
      execCmd += " > ";
      execCmd += tmpFileName;
    }

    auto res = std::system(execCmd.c_str());

    if (captureOutput) {
      std::ifstream t(tmpFileName.c_str());
      resultStr =
          std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    }
    return res;
  }

  std::string getOutput() { return resultStr; }
};
}  // namespace hip

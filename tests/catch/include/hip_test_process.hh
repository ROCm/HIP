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

#include "hip_test_common.hh"
#include "hip_test_filesystem.hh"

#include <string>
#include <array>
#include <cstdlib>
#include <random>
#include <fstream>
#include <streambuf>

namespace hip {
/*
Class to spawn a process in isolation and test its standard output and return status
Good for printf tests and environment variable tests

How to use:
Have the stand alone exe in the same folder
Init a class using hip::SpawnProc proc("ExeName", yes_or_no_to_capture_output);
proc.run("Optional command line args");
*/
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
    auto dir = fs::path(TestContext::get().currentPath());
    dir /= exeName;
    exeName = dir.string();

    INFO("Testing that exe exists: " << exeName);
    REQUIRE(fs::exists(exeName));

    if (captureOutput) {
      auto path = fs::temp_directory_path();
      path /= getRandomString();
      tmpFileName = path.string();
      INFO("Testing that capture file does not exist already: " << tmpFileName);
      REQUIRE(!fs::exists(tmpFileName));
    }
  }

  int run(std::string commandLineArgs = "") {
    std::string execCmd = exeName;

    // Append command line args
    if (commandLineArgs.size() > 0) {
      execCmd += " ";  // Add space for command line args
      execCmd += commandLineArgs;
    }

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
#if HT_LINUX
    return WEXITSTATUS(res);
#else
    return res;
#endif
  }

  std::string getOutput() { return resultStr; }
};
}  // namespace hip

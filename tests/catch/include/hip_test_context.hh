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
#include <hip/hip_runtime.h>
#include <vector>
#include <string>
#include <set>

#if defined(_WIN32)
#define HT_WIN 1
#define HT_LINUX 0
#elif defined(__linux__)
#define HT_WIN 0
#define HT_LINUX 1
#else
#error "OS not recognized"
#endif

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
#define HT_AMD 1
#define HT_NVIDIA 0
#elif defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__)
#define HT_AMD 0
#define HT_NVIDIA 1
#else
#error "Platform not recognized"
#endif

static int _log_enable = (std::getenv("HT_LOG_ENABLE") ? 1 : 0);

#define LogPrintf(format, ...)                                                                     \
  {                                                                                                \
    if (_log_enable) {                                                                             \
      printf(format, __VA_ARGS__);                                                                 \
      printf("%c", '\n');                                                                          \
    }                                                                                              \
  }

typedef struct Config_ {
  std::string json_file;              // Json file
  std::string platform;               // amd/nvidia
  std::string os;                     // windows/linux
} Config;

class TestContext {
  bool p_windows = false, p_linux = false;  // OS
  bool amd = false, nvidia = false;         // HIP Platform
  std::string exe_path;
  std::string current_test;
  std::set<std::string> skip_test;
  std::string json_file_;
  std::vector<std::string>  platform_list_ = {"amd" , "nvidia"};
  std::vector<std::string>  os_list_ = {"windows", "linux", "all"};
  std::vector<std::string>  amd_arch_list_ = {};

  Config config_;
  std::string& getJsonFile();
  std::string substringFound( std::vector<std::string> list,
                              std::string filename);
  void detectOS();
  void detectPlatform();
  void fillConfig();
  void setExePath(int, char**);
  void parseOptions(int, char**);
  bool parseJsonFile();
  std::string getMatchingConfigFile(std::string config_dir);
  const Config& getConfig() const { return config_; }

  TestContext(int argc, char** argv);

 public:
  static const TestContext& get(int argc = 0, char** argv = nullptr) {
    static TestContext instance(argc, argv);
    return instance;
  }

  bool isWindows() const;
  bool isLinux() const;
  bool isNvidia() const;
  bool isAmd() const;
  bool skipTest() const;

  const std::string& getCurrentTest() const { return current_test; }
  std::string currentPath() const;

  TestContext(const TestContext&) = delete;
  void operator=(const TestContext&) = delete;
};

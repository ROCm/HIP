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
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <atomic>
#include <mutex>
#include <vector>
#include <iostream>
#include <string>
#include <set>
#include <unordered_map>

// OS Check
#if defined(_WIN32)
#define HT_WIN 1
#define HT_LINUX 0
#elif defined(__linux__)
#define HT_WIN 0
#define HT_LINUX 1
#else
#error "OS not recognized"
#endif

// Platform check
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
#define HT_AMD 1
#define HT_NVIDIA 0
#elif defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__)
#define HT_AMD 0
#define HT_NVIDIA 1
#else
#error "Platform not recognized"
#endif

typedef struct Config_ {
  std::vector<std::string> json_files;  // Json files
  std::string platform;   // amd/nvidia
  std::string os;         // windows/linux
} Config;

// Store Multi threaded results
struct HCResult {
  size_t line;            // Line of check (HIP_CHECK_THREAD or REQUIRE_THREAD)
  std::string file;       // File name of the check
  hipError_t result;      // hipResult for HIP_CHECK_THREAD, for conditions its hipSuccess
  std::string call;       // Call of HIP API or a bool condition
  bool conditionsResult;  // If bool condition, result of call. For HIP Calls its true
  HCResult(size_t l, std::string f, hipError_t r, std::string c, bool b = true)
      : line(l), file(f), result(r), call(c), conditionsResult(b) {}
};


class TestContext {
  bool p_windows = false, p_linux = false;  // OS
  bool amd = false, nvidia = false;         // HIP Platform
  std::string exe_path;
  std::string current_test;
  std::set<std::string> skip_test;
  std::string json_file_;
  std::vector<std::string> platform_list_ = {"amd", "nvidia"};
  std::vector<std::string> os_list_ = {"windows", "linux", "all"};
  std::vector<std::string> amd_arch_list_ = {};

  struct rtcState {
    hipModule_t module;
    hipFunction_t kernelFunction;
  };

  std::unordered_map<std::string, rtcState> compiledKernels{};

  Config config_;
  std::string& getCommonJsonFile();
  std::string substringFound(std::vector<std::string> list, std::string filename);
  void detectOS();
  void detectPlatform();
  void getConfigFiles();
  void setExePath(int, char**);
  void parseOptions(int, char**);
  bool parseJsonFiles();
  std::string getMatchingConfigFile(std::string config_dir);
  const Config& getConfig() const { return config_; }


  TestContext(int argc, char** argv);

  // Multi threaded checks helpers
  std::mutex resultMutex;
  std::vector<HCResult> results;  // Multi threaded test results buffer
  std::atomic<bool> hasErrorOccured_{false};

 public:
  static TestContext& get(int argc = 0, char** argv = nullptr) {
    static TestContext instance(argc, argv);
    return instance;
  }

  static std::string getEnvVar(std::string var) {
    #if defined(_WIN32)
    rsize_t MAX_LEN = 4096;
    char dstBuf[MAX_LEN];
    size_t dstSize;
    if (!::getenv_s(&dstSize, dstBuf, MAX_LEN, var.c_str())) {
      return std::string(dstBuf);
    }
    #elif defined(__linux__)
    char* val = std::getenv(var.c_str());
    if (val != NULL) {
      return std::string(val);
    }
    #else
    #error "OS not recognized"
    #endif
    return std::string("");
  }


  bool isWindows() const;
  bool isLinux() const;
  bool isNvidia() const;
  bool isAmd() const;
  bool skipTest() const;

  const std::string& getCurrentTest() const { return current_test; }
  std::string currentPath() const;

  // Multi threaded results helpers
  void addResults(HCResult r);  // Add multi threaded results
  void finalizeResults();       // Validate on all results
  bool hasErrorOccured();       // Query if error has occured

  /**
   * @brief Unload all loaded modules.
   * Note: This function needs to be called at the end of each test that uses RTC.
   *       It is not possible to unload the loaded modules without adding explicit code to the end
   * of each test. This function exists only to provide a clean way to exit a test when using RTC.
   *       However, not unloading a module explicitly shouldn't have any effect on the outcome of
   * the test.
   */
  void cleanContext();

  /**
   * @brief Keeps track of all the already compiled rtc kernels.
   *
   * @param kernelNameExpression The name expression (e.g. hipTest::vectorADD<float>).
   * @param loadedModule  The loaded module.
   * @param kernelFunction The hipFunction that will be used to run the kernel in the future.
   */
  void trackRtcState(std::string kernelNameExpression, hipModule_t loadedModule,
                     hipFunction_t kernelFunction);

  /**
   * @brief Get the already compiled hip rtc kernel function if it exists.
   *
   * @param kernelNameExpression The name expression (e.g. hipTest::vectorADD<float>).
   * @return the hipFunction if it exists. nullptr otherwise
   */
  hipFunction_t getFunction(const std::string kernelNameExpression);

  TestContext(const TestContext&) = delete;
  void operator=(const TestContext&) = delete;

  ~TestContext();
};

static bool _log_enable = (!TestContext::getEnvVar("HT_LOG_ENABLE").empty() ? true : false);

// printing logs
#define LogPrintf(format, ...)                                                                   \
{                                                                                                \
  if(_log_enable) {                                                                              \
    printf(format, __VA_ARGS__);                                                                 \
    printf("%c", '\n');                                                                          \
  }                                                                                              \
}
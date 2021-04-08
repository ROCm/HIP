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
  std::vector<std::string> devices;   // gfx906, etc
  std::vector<std::string> targetId;  // Target Ids, only for AMD, gfx906:sramecc+:xnack-
  std::string os;                     // windows/linux
} Config;

class TestContext {
  bool p_windows = false, p_linux = false;  // OS
  bool amd = false, nvidia = false;         // HIP Platform
  std::string exe_path;
  std::string current_test;
  std::set<std::string> skip_test;

  Config config_;

  void detectOS();
  void detectPlatform();
  void fillConfig();
  void setExePath(int, char**);
  void parseOptions(int, char**);
  bool parseJsonFile();
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
  const std::vector<std::string>& getDevices() const;
  const std::vector<std::string>& getTargetId() const;

  const std::string& getCurrentTest() const { return current_test; }
  std::string currentPath();

  TestContext(const TestContext&) = delete;
  void operator=(const TestContext&) = delete;
};

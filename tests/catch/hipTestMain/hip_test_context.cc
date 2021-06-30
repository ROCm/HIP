#include <hip_test_common.hh>
#include <picojson.h>
#include <fstream>
#include <sstream>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "gg filesystem"
#endif

#include <regex>

static std::string getValue(std::string option, const std::string& opt_str) {
  std::string s_opt = opt_str;
  return s_opt.erase(0, option.size());
}

static std::string trimName(std::string input, char trim) {
  auto pos_ = input.find(trim);
  auto res = input;
  if (pos_ == std::string::npos) {
    input = "";
  } else {
    res = input.substr(0, pos_);
    input = input.substr(pos_);
  }
  return res;
}

const std::vector<std::string>& TestContext::getDevices() const { return config_.devices; }
const std::vector<std::string>& TestContext::getTargetId() const { return config_.targetId; }

void TestContext::detectOS() {
#if (HT_WIN == 1)
  p_windows = true;
#elif (HT_LINUX == 1)
  p_linux = true;
#endif
}

void TestContext::detectPlatform() {
#if (HT_AMD == 1)
  amd = true;
#elif (HT_NVIDIA == 1)
  nvidia = true;
#endif
}

void TestContext::fillConfig() {
  const char* env_config = std::getenv("HT_CONFIG_FILE");
  LogPrintf("Env Config file: %s",
            (env_config != nullptr) ? env_config : "Not found, using default config");

  // Check if path has been provided
  std::string config_str;
  if (env_config != nullptr) {
    config_str = env_config;
  } else {
    config_str = "config.json";
  }

  fs::path config_path = config_str;
  if (config_path.has_parent_path()) {
    config_.json_file = config_str;
  } else {
    config_.json_file = exe_path + config_str;
  }
  LogPrintf("Config file path: %s", config_.json_file.c_str());

  config_.platform = (amd ? "amd" : (nvidia ? "nvidia" : "unknown"));
  config_.os = (p_windows ? "windows" : (p_linux ? "linux" : "unknown"));

  if (config_.os == "unknown" || config_.platform == "unknown") {
    LogPrintf("%s", "Either Config or Os is unknown, this wont end well");
    abort();
  }

  int deviceCount = 0;
  auto res = hipGetDeviceCount(&deviceCount);
  if (res != hipSuccess) {
    LogPrintf("HIP Device Count query failed with: %s", hipGetErrorString(res));
    abort();
  }
  if (deviceCount == 0) {
    LogPrintf("%s", "No hip devices found");
    abort();
  }
  config_.devices.reserve(deviceCount);
  for (int i = 0; i < deviceCount; i++) {
    hipDeviceProp_t props;
    res = hipGetDeviceProperties(&props, i);
    if (res != hipSuccess) {
      LogPrintf("HIP Device Count query failed with: %s", hipGetErrorString(res));
      abort();
    }
    if (amd) {
      std::string tid = std::string(props.gcnArchName);
      config_.targetId.push_back(tid);
      std::string dev = trimName(tid, ':');
      config_.devices.push_back(dev);
    } else if (nvidia) {
      config_.devices.push_back(std::string(props.name));
    } else {
      LogPrintf("%s", "Unknown platform");
      abort();
    }
  }
}

TestContext::TestContext(int argc, char** argv) {
  detectOS();
  detectPlatform();
  setExePath(argc, argv);
  fillConfig();
  parseJsonFile();
  parseOptions(argc, argv);
}

void TestContext::setExePath(int argc, char** argv) {
  fs::path p = std::string(argv[0]);
  if (p.has_filename()) p.remove_filename();
  exe_path = p.string();
}

bool TestContext::isWindows() const { return p_windows; }
bool TestContext::isLinux() const { return p_linux; }

bool TestContext::isNvidia() const { return nvidia; }
bool TestContext::isAmd() const { return amd; }

void TestContext::parseOptions(int argc, char** argv) {
  // Test name is at [1] position
  if (argc != 2) return;
  current_test = std::string(argv[1]);
}

bool TestContext::skipTest() const {
  // Direct Match
  auto flags = std::regex::ECMAScript;
  for (const auto& i : skip_test) {
    auto regex = std::regex(i.c_str(), flags);
    if (std::regex_match(current_test, regex)) {
      return true;
    }
  }
  // TODO add test case skip as well
  return false;
}

std::string TestContext::currentPath() { return fs::current_path().string(); }

bool TestContext::parseJsonFile() {
  // Check if file exists
  if (!fs::exists(config_.json_file)) {
    LogPrintf("Unable to find the file: %s", config_.json_file.c_str());
    return true;
  }

  // Open the file
  std::ifstream js_file(config_.json_file);
  std::string json_str((std::istreambuf_iterator<char>(js_file)), std::istreambuf_iterator<char>());
  LogPrintf("Json contents:: %s", json_str.data());

  picojson::value v;
  std::string err;
  const char* json_end =
      picojson::parse(v, json_str.data(), json_str.data() + json_str.size(), &err);
  if (err.size() > 1) {
    LogPrintf("Error from PicoJson: %s", err.data());
    return false;
  }

  if (!v.is<picojson::object>()) {
    LogPrintf("%s", "Data in json is not in correct format, it should be an object");
    return false;
  }
  const picojson::object &o = v.get<picojson::object>();
  for (picojson::object::const_iterator i = o.begin(); i != o.end(); ++i) {
    // Processing for DisabledTests
    if (i->first == "DisabledTests") {
      // Value should contain list of values
      if (!i->second.is<picojson::array>()) return false;

      auto& val = i->second.get<picojson::array>();
      for (auto ai = val.begin(); ai != val.end(); ai++) {
        std::string tmp = ai->get<std::string>();
        std::string newRegexName;
        for(const auto &c : tmp) {
          if (c == '*')
            newRegexName += ".*";
          else
            newRegexName += c;
        }
        skip_test.insert(newRegexName);
      }
    }
  }

  return true;
}

#include <cstdlib>
#include <hip_test_common.hh>
#include <picojson.h>
#include <fstream>
#include <sstream>
#include <regex>
#include "hip_test_filesystem.hh"

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

std::string TestContext::substringFound(std::vector<std::string> list, std::string filename) {
  std::string match = "";
  for (unsigned int i = 0; i < list.size(); i++) {
    if (filename.find(list.at(i)) != std::string::npos) {
      match = list.at(i);
      break;
    }
  }
  return match;
}


std::string TestContext::getMatchingConfigFile(std::string config_dir) {
  std::string configFileToUse;
  for (auto& p : fs::recursive_directory_iterator(config_dir)) {
    fs::path filename = p.path();
    std::string cur_arch = "TODO";
    std::string arch = substringFound(amd_arch_list_, filename.filename().string());
    std::string platform = substringFound(platform_list_, filename.filename().string());
    std::string os = substringFound(os_list_, filename.filename().string());
    std::string common_arch = "common";
    std::vector<std::string> default_arch_vec {common_arch};
    std::string common = substringFound(default_arch_vec, filename.filename().string());
    // if arch found then use that exit from loop
    if (arch == cur_arch) {
      configFileToUse = filename.string();
      break;
      // match the platform/os and continue to look
    } else if ((platform == config_.platform) && (os == config_.os || os == "all") &&
                common == common_arch) { // ensures only common file is returned
      configFileToUse = filename.string();
    }
  }
  return configFileToUse;
}


std::string& TestContext::getCommonJsonFile() {
  fs::path config_dir = exe_path;
  config_dir = config_dir.parent_path();
  int levels = 0;
  bool configFolderFound = false;
  std::vector<std::string> configList;
  std::string configFile;
  // check a max of 5 levels down the executable path
  while (levels < 5) {
    fs::path temp_path = config_dir;
    temp_path /= "hipTestMain";
    temp_path /= "config";
    if (fs::exists(temp_path)) {
      config_dir = fs::absolute(temp_path);
      configFolderFound = true;
      break;
    } else {
      config_dir = config_dir.parent_path();
      levels++;
    }
  }

  // get config.json files if config folder.
  if (configFolderFound) {
    json_file_ = getMatchingConfigFile(config_dir.string());
  }
  return json_file_;
}


void TestContext::getConfigFiles() {
  config_.platform = (amd ? "amd" : (nvidia ? "nvidia" : "unknown"));
  config_.os = (p_windows ? "windows" : (p_linux ? "linux" : "unknown"));

  if (config_.os == "unknown" || config_.platform == "unknown") {
    LogPrintf("%s", "Either Config or Os is unknown, this wont end well");
    abort();
  }

  const char* env_config = std::getenv("HIP_CATCH_EXCLUDE_FILE");
  LogPrintf("Env Config file: %s",
            (env_config != nullptr) ? env_config : "Not found, using common config");
  // HIP_CATCH_EXCLUDE_FILE is set for custom file path
  if (env_config != nullptr) {
    if(fs::exists(env_config)) {
      config_.json_files.push_back(env_config);
    }
  } else {
    // get common json file
    config_.json_files.push_back(getCommonJsonFile());
  }

  for (const auto& fl : config_.json_files) {
    LogPrintf("Config file path: %s", fl.c_str());
  }
}

TestContext::TestContext(int argc, char** argv) {
  detectOS();
  detectPlatform();
  setExePath(argc, argv);
  getConfigFiles();
  parseJsonFiles();
  parseOptions(argc, argv);
}

void TestContext::setExePath(int argc, char** argv) {
  if (argc == 0) return;
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

std::string TestContext::currentPath() const { return fs::current_path().string(); }

bool TestContext::parseJsonFiles() {
  // Check if file exists
  for (const auto& fl : config_.json_files) {
    if (!fs::exists(fl)) {
      LogPrintf("Unable to find the file: %s", fl.c_str());
      return true;
    }
    // Open the file
    std::ifstream js_file(fl);
    std::string json_str((std::istreambuf_iterator<char>(js_file)), std::istreambuf_iterator<char>());
    LogPrintf("Json contents:: %s", json_str.data());

    picojson::value v;
    std::string err = picojson::parse(v, json_str);
    if (err.size() > 1) {
      LogPrintf("Error from PicoJson: %s", err.data());
      return false;
    }

    if (!v.is<picojson::object>()) {
      LogPrintf("%s", "Data in json is not in correct format, it should be an object");
      return false;
    }

    const picojson::object& o = v.get<picojson::object>();
    for (picojson::object::const_iterator i = o.begin(); i != o.end(); ++i) {
      // Processing for DisabledTests
      if (i->first == "DisabledTests") {
        // Value should contain list of values
        if (!i->second.is<picojson::array>()) return false;

        auto& val = i->second.get<picojson::array>();
        for (auto ai = val.begin(); ai != val.end(); ai++) {
          std::string tmp = ai->get<std::string>();
          std::string newRegexName;
          for (const auto& c : tmp) {
            if (c == '*')
              newRegexName += ".*";
            else
              newRegexName += c;
          }
          skip_test.insert(newRegexName);
        }
      }
    }
  }
  return true;
}

void TestContext::cleanContext() {
  for (auto& pair : compiledKernels) {
    hipError_t error = hipModuleUnload(pair.second.module);
    if (error != hipSuccess) {
      throw std::runtime_error("Unable to unload rtc module");
    }
  }
}

void TestContext::trackRtcState(std::string kernelNameExpression, hipModule_t loadedModule,
                                hipFunction_t kernelFunction) {
  rtcState state{loadedModule, kernelFunction};
  compiledKernels[kernelNameExpression] = state;
}

hipFunction_t TestContext::getFunction(const std::string kernelNameExpression) {
  auto it{compiledKernels.find(kernelNameExpression)};

  if (it != compiledKernels.end()) {
    return it->second.kernelFunction;
  } else {
    return nullptr;
  }
}

void TestContext::addResults(HCResult r) {
  std::unique_lock<std::mutex> lock(resultMutex);
  results.push_back(r);
  if ((!r.conditionsResult) ||
      ((r.result != hipSuccess) && (r.result != hipErrorPeerAccessAlreadyEnabled))) {
    hasErrorOccured_.store(true);
  }
}

void TestContext::finalizeResults() {
  std::unique_lock<std::mutex> lock(resultMutex);
  // clear the results whatever happens
  std::shared_ptr<void> emptyVec(nullptr, [this](auto) { results.clear(); });

  for (const auto& i : results) {
    INFO("HIP API Result check\n    File:: "
         << i.file << "\n    Line:: " << i.line << "\n    API:: " << i.call
         << "\n    Result:: " << i.result << "\n    Result Str:: " << hipGetErrorString(i.result));
    REQUIRE(((i.result == hipSuccess) || (i.result == hipErrorPeerAccessAlreadyEnabled)));
    REQUIRE(i.conditionsResult);
  }
  hasErrorOccured_.store(false);  // Clear the flag
}

bool TestContext::hasErrorOccured() { return hasErrorOccured_.load(); }

TestContext::~TestContext() {
  // Show this message when there are unchecked results
  if (results.size() != 0) {
    std::cerr << "HIP_CHECK_THREAD_FINALIZE() has not been called after HIP_CHECK_THREAD\n"
              << "Please call HIP_CHECK_THREAD_FINALIZE after joining threads\n"
              << "There is/are " << results.size() << " unchecked results from threads."
              << std::endl;
    std::abort();  // Crash to bring users attention to this message and avoid accidental passing of
                   // tests without checking for errors
  }
}
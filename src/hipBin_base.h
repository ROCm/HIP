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

#ifndef SRC_HIPBIN_BASE_H_
#define SRC_HIPBIN_BASE_H_
// We haven't checked which filesystem to include yet
#ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL

// Check for feature test macro for <filesystem>
#   if defined(__cpp_lib_filesystem)
#       define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0

// Check for feature test macro for <experimental/filesystem>
#   elif defined(__cpp_lib_experimental_filesystem)
#       define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// We can't check if headers exist...
// Let's assume experimental to be safe
#   elif !defined(__has_include)
#       define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// Check if the header "<filesystem>" exists
#   elif __has_include(<filesystem>)

// If we're compiling on Visual Studio and are not compiling with C++17, we need to use experimental
#       ifdef _MSC_VER

// Check and include header that defines "_HAS_CXX17"
#           if __has_include(<yvals_core.h>)
#               include <yvals_core.h>

// Check for enabled C++17 support
#               if defined(_HAS_CXX17) && _HAS_CXX17
// We're using C++17, so let's use the normal version
#                   define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
#               endif
#           endif

// If the marco isn't defined yet, that means any of the other VS specific checks failed, so we need to use experimental
#           ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
#               define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1
#           endif

// Not on Visual Studio. Let's use the normal version
#       else  // #ifdef _MSC_VER
#           define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
#       endif

// Check if the header "<filesystem>" exists
#   elif __has_include(<experimental/filesystem>)
#       define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// Fail if neither header is available with a nice error message
#   else
#       error Could not find system header "<filesystem>" or "<experimental/filesystem>"
#   endif

// We priously determined that we need the exprimental version
#   if INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
// Include it
#       include <experimental/filesystem>

// We need the alias from std::experimental::filesystem to std::filesystem
namespace fs = std::experimental::filesystem;

// We have a decent compiler and can use the normal version
#   else
// Include it
#       include <filesystem>
namespace fs = std::filesystem;
#   endif

#endif  // #ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL

#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <fstream>
#include <regex>
#include <algorithm>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
  #include "tchar.h"
  #include "windows.h"
  #include "io.h"
  #ifdef _UNICODE
    typedef wchar_t TCHAR;
    typedef std::wstring TSTR;
    typedef std::wstring::size_type TSIZE;
    # define ENDLINE L"/\\"
  #else
    typedef char TCHAR;
    typedef std::string TSTR;
    typedef std::string::size_type TSIZE;
    # define ENDLINE "/\\"
#endif
#else
  #include <unistd.h>
#endif





using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::regex;
using std::map;
using std::smatch;
using std::stringstream;

// All envirnoment variables used in the code
# define PATH                       "PATH"
# define HIP_ROCCLR_HOME            "HIP_ROCCLR_HOME"
# define HIP_PATH                   "HIP_PATH"
# define ROCM_PATH                  "ROCM_PATH"
# define CUDA_PATH                  "CUDA_PATH"
# define HSA_PATH                   "HSA_PATH"
# define HIP_CLANG_PATH             "HIP_CLANG_PATH"
# define HIP_PLATFORM               "HIP_PLATFORM"
# define HIP_COMPILER               "HIP_COMPILER"
# define HIP_RUNTIME                "HIP_RUNTIME"
# define LD_LIBRARY_PATH            "LD_LIBRARY_PATH"

// hipcc
# define HIPCC_COMPILE_FLAGS_APPEND     "HIPCC_COMPILE_FLAGS_APPEND"
# define HIPCC_LINK_FLAGS_APPEND        "HIPCC_LINK_FLAGS_APPEND"
# define HIP_LIB_PATH                   "HIP_LIB_PATH"
# define DEVICE_LIB_PATH                "DEVICE_LIB_PATH"
# define HIP_CLANG_HCC_COMPAT_MODE      "HIP_CLANG_HCC_COMPAT_MODE"
# define HIP_COMPILE_CXX_AS_HIP         "HIP_COMPILE_CXX_AS_HIP"
# define HIPCC_VERBOSE                  "HIPCC_VERBOSE"
# define HCC_AMDGPU_TARGET              "HCC_AMDGPU_TARGET"

# define HIP_BASE_VERSION_MAJOR     "4"
# define HIP_BASE_VERSION_MINOR     "4"
# define HIP_BASE_VERSION_PATCH     "0"
# define HIP_BASE_VERSION_GITHASH   "0"

// Known HIP target names.
vector<string> knownTargets = { "gfx700", "gfx701", "gfx702", "gfx703", "gfx704", "gfx705",
                                "gfx801", "gfx802", "gfx803", "gfx805", "gfx810",
                                "gfx900", "gfx902", "gfx904", "gfx906", "gfx908", "gfx909", "gfx90a",
                                "gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032" };


// Known Features
vector<string> knownFeatures = { "sramecc - " , "sramecc + ", "xnack - ", "xnack + " };


enum PlatformType {
  amd = 0,
  nvidia,
  intel
};

string PlatformTypeStr(PlatformType platform) {
  switch (platform) {
  case amd:
    return "amd";
  case nvidia:
    return "nvidia";
  case intel:
    return "intel";
  default:
    return "invalid platform";
  }
}

enum CompilerType {
  clang = 0,
  nvcc
};


string CompilerTypeStr(CompilerType compiler) {
  switch (compiler) {
  case clang:
    return "clang";
  case nvcc:
    return "nvcc";
  default:
    return "invalid CompilerType";
  }
}



enum RuntimeType {
  rocclr = 0,
  cuda
};

string RuntimeTypeStr(RuntimeType runtime) {
  switch (runtime) {
  case rocclr:
    return "rocclr";
  case cuda:
    return "cuda";
  default:
    return "invalid RuntimeType";
  }
}



enum OsType {
  lnx = 0,
  windows
};

string OsTypeStr(OsType os) {
  switch (os) {
  case lnx:
    return "linux";
  case windows:
    return "windows";
  default:
    return "invalid OsType";
  }
}


struct PlatformInfo {
  PlatformType platform;
  CompilerType compiler;
  RuntimeType runtime;
  OsType os;
};



struct EnvVariables {
  string path_ = "";
  string hipPathEnv_ = "";
  string hipRocclrPathEnv_ = "";
  string roccmPathEnv_ = "";
  string cudaPathEnv_ = "";
  string hsaPathEnv_ = "";
  string hipClangPathEnv_ = "";
  string hipPlatformEnv_ = "";
  string hipCompilerEnv_ = "";
  string hipRuntimeEnv_ = "";
  string ldLibraryPathEnv_ = "";
  string verboseEnv_ = "";
  string hipccCompileFlagsAppendEnv_ = "";
  string hipccLinkFlagsAppendEnv_ = "";
  string hipLibPathEnv_ = "";
  string deviceLibPathEnv_ = "";
  string hipClangHccCompactModeEnv_ = "";
  string hipCompileCxxAsHipEnv_ = "";
  string hccAmdGpuTargetEnv_ = "";
  void operator =(const EnvVariables& var) {
    path_ = var.path_;
    hipPathEnv_ = var.hipPathEnv_;
    hipRocclrPathEnv_ = var.hipRocclrPathEnv_;
    roccmPathEnv_ = var.roccmPathEnv_;
    cudaPathEnv_ = var.cudaPathEnv_;
    hsaPathEnv_ = var.hsaPathEnv_;
    hipClangPathEnv_ = var.hipClangPathEnv_;
    hipPlatformEnv_ = var.hipPlatformEnv_;
    hipCompilerEnv_ = var.hipCompilerEnv_;
    hipRuntimeEnv_ = var.hipRuntimeEnv_;
    ldLibraryPathEnv_ = var.ldLibraryPathEnv_;
    verboseEnv_ = var.verboseEnv_;
    hipccCompileFlagsAppendEnv_ = var.hipccCompileFlagsAppendEnv_;
    hipccLinkFlagsAppendEnv_ = var.hipccLinkFlagsAppendEnv_;
    hipLibPathEnv_ = var.hipLibPathEnv_;
    deviceLibPathEnv_ = var.deviceLibPathEnv_;
    hipClangHccCompactModeEnv_ = var.hipClangHccCompactModeEnv_;
    hipCompileCxxAsHipEnv_ = var.hipCompileCxxAsHipEnv_;
    hccAmdGpuTargetEnv_ = var.hccAmdGpuTargetEnv_;
  }
  friend std::ostream& operator <<(std::ostream& os, const EnvVariables& var) {
    os << "Path: "                           << var.path_ << endl;
    os << "Hip Path: "                       << var.hipPathEnv_ << endl;
    os << "Hip Rocclr Path: "                << var.hipRocclrPathEnv_ << endl;
    os << "Roccm Path: "                     << var.roccmPathEnv_ << endl;
    os << "Cuda Path: "                      << var.cudaPathEnv_ << endl;
    os << "Hsa Path: "                       << var.hsaPathEnv_ << endl;
    os << "Hip Clang Path: "                 << var.hipClangPathEnv_ << endl;
    os << "Hip Platform: "                   << var.hipPlatformEnv_ << endl;
    os << "Hip Compiler: "                   << var.hipCompilerEnv_ << endl;
    os << "Hip Runtime: "                    << var.hipRuntimeEnv_ << endl;
    os << "LD Library Path: "                << var.ldLibraryPathEnv_ << endl;
    os << "Verbose: "                        << var.verboseEnv_ << endl;
    os << "Hipcc Compile Flags Append: "     << var.hipccCompileFlagsAppendEnv_ << endl;
    os << "Hipcc Link Flags Append: "        << var.hipccLinkFlagsAppendEnv_ << endl;
    os << "Hip lib Path: "                   << var.hipLibPathEnv_ << endl;
    os << "Device lib Path: "                << var.deviceLibPathEnv_ << endl;
    os << "Hip Clang HCC Compact mode: "     << var.hipClangHccCompactModeEnv_ << endl;
    os << "Hip Compile Cxx as Hip: "         << var.hipCompileCxxAsHipEnv_ << endl;
    os << "Hcc Amd Gpu Target: "             << var.hccAmdGpuTargetEnv_ << endl;
    return os;
  }
};


enum HipBinCommand {
  unknown = -1,
  path,
  roccmpath,
  cpp_config,
  compiler,
  platform,
  runtime,
  hipclangpath,
  full,
  version,
  check,
  newline,
  help,
};


struct SystemCmdOut {
  string out;
  int exitCode = 0;
};


class HipBinBase {
 public:
  HipBinBase();
  virtual ~HipBinBase() {}
  // Common helper functions
  std::string getSelfPath() const;
  vector<string> splitStr(string fullStr, char delimiter) const;
  string replaceStr(const string& s, const string& toReplace, const string& replaceWith) const;
  string replaceRegex(const string& s, std::regex toReplace, string replaceWith) const;
  SystemCmdOut exec(const char* cmd, bool printConsole) const;
  string getTempDir();
  void deleteTempFiles();
  string mktempFile(string name);
  string chomp(string str) const;
  void getSystemInfo() const;
  void printEnvirnomentVariables() const;
  void readEnvVariables();
  const EnvVariables& getEnvVariables() const;
  void readOSInfo();
  const OsType& getOSInfo() const;
  void constructHipPath();
  void constructRoccmPath();
  void constructHsaPath();
  void readHipVersion();
  const string& getHipPath() const;
  const string& getRoccmPath() const;
  const string& getHsaPath() const;
  const string& getHipVersion() const;
  void constructRocclrHomePath();
  const string& getRocclrHomePath() const;
  string readConfigMap(map<string, string> hipVersionMap, string keyName, string defaultValue) const;
  void printUsage() const;
  bool canRun(string exeName, string& cmdOut);
  map<string, string> parseConfigFile(fs::path configPath) const;
  bool substringPresent(string fullString, string subString) const;
  bool stringRegexMatch(string fullString, string pattern) const;

  // Interface functions
  virtual void constructCompilerPath() = 0;
  virtual void printFull() = 0;
  virtual bool detectPlatform() = 0;
  virtual const string& getCompilerPath() const = 0;
  virtual void printCompilerInfo() const = 0;
  virtual string getCompilerVersion() = 0;
  virtual string getCompilerIncludePath() = 0;
  virtual const PlatformInfo& getPlatformInfo() const = 0;
  virtual string getCppConfig() = 0;
  virtual void checkHipconfig() = 0;
  virtual string getDeviceLibPath() const = 0;
  virtual string getHipLibPath() const = 0;
  virtual string getHipCC() const = 0;
  virtual string getHipInclude() const = 0;
  virtual void initializeHipCXXFlags() = 0;
  virtual void initializeHipCFlags() = 0;
  virtual void initializeHipLdFlags() = 0;
  virtual const string& getHipCXXFlags() const = 0;
  virtual const string& getHipCFlags() const = 0;
  virtual const string& getHipLdFlags() const = 0;
  virtual void executeHipCCCmd(vector<string> argv) = 0;

 private:
  EnvVariables envVariables_, variables_;
  OsType osInfo_;
  string hipVersion_;
  vector<string> tmpDirs_;
  vector<string> tmpFiles_;
};

HipBinBase::HipBinBase() {
  readOSInfo();                 // detects if windows or linux
  readEnvVariables();           // reads the envirnoment variables
  constructHipPath();           // constructs HIP Path
  constructRoccmPath();         // constructs Roccm Path
  constructRocclrHomePath();    // constructs RocclrHomePath
  constructHsaPath();           // constructs hsa path
  readHipVersion();             // stores the hip version
}

// create temp file with the template name
string HipBinBase::mktempFile(string name) {
  string fileName;
#if defined(_WIN32) || defined(_WIN64)
  fileName = _mktemp(&name[0]);
# else
  fileName = mktemp(&name[0]);
# endif
  tmpFiles_.push_back(fileName);
  return fileName;
}

// gets the path of the executable name
std::string HipBinBase::getSelfPath() const {
  int MAX_PATH_CHAR = 1024;
  int bufferSize = 0;
  string path;
  #if defined(_WIN32) || defined(_WIN64)
    TCHAR buffer[MAX_PATH] = { 0 };
    bufferSize = GetModuleFileName(NULL, buffer, MAX_PATH_CHAR);
    TSIZE pos = TSTR(buffer).find_last_of(ENDLINE);
    TSTR wide = TSTR(buffer).substr(0, pos);
    path = std::string(wide.begin(), wide.end());
  #else
    char buff[MAX_PATH_CHAR];
    ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff) - 1);
    if (len != -1) {
      buff[len] = '\0';
      path = std::string(buff);
      fs::path exePath(path);
      path = exePath.parent_path().string();
    }
  #endif
  return path;
}


// removes the empty spaces and end lines
string HipBinBase::chomp(string str) const {
  string strChomp = str;
  strChomp.erase(str.find_last_not_of(" \n\r\t")+1);
  return strChomp;
}

// matches the pattern in the string
bool HipBinBase::stringRegexMatch(string fullString, string pattern) const {
  bool found = false;
  std::regex e(pattern);
  if (std::regex_match(fullString, e)) {
    found = true;
  }
  return found;
}

// subtring is present in string
bool HipBinBase::substringPresent(string fullString, string subString) const {
  bool found = false;
  if (fullString.find(subString) != std::string::npos) {
    found = true;
  }
  return found;
}

// splits the string with the delimiter
vector<string> HipBinBase::splitStr(string fullStr, char delimiter) const {
  vector <string> tokens;
  stringstream check1(fullStr);
  string intermediate;
  while (getline(check1, intermediate, delimiter)) {
    tokens.push_back(intermediate);
  }
  return tokens;
}


// replaces the toReplace string with replaceWith string. Returns the new string
string HipBinBase::replaceStr(const string& s, const string& toReplace, const string& replaceWith) const {
  string out = s;
  std::size_t pos = out.find(toReplace);
  if (pos == std::string::npos) return out;
  return out.replace(pos, toReplace.length(), replaceWith);
}

// replaces the toReplace regex pattern with replaceWith string. Returns the new string
string HipBinBase::replaceRegex(const string& s, std::regex toReplace, string replaceWith) const {
  std::string out = s;
  while (std::regex_search(out, toReplace)) {
    out = std::regex_replace(out, toReplace, replaceWith);
  }
  return out;
}

// prints the envirnoment variables
void HipBinBase::printEnvirnomentVariables() const {
  const OsType& os = getOSInfo();
  if (os == windows) {
    cout << "PATH=" << envVariables_.path_ << "\n" << endl;
    system("set | findstr /B /C:\"HIP\" /C:\"HSA\" /C:\"CUDA\" /C:\"LD_LIBRARY_PATH\"");
  } else {
    string cmd = "echo PATH =";
    cmd += envVariables_.path_;
    system(cmd.c_str());
    system("env | egrep '^HIP|^HSA|^CUDA|^LD_LIBRARY_PATH'");
  }
}

// prints system information
void HipBinBase::getSystemInfo() const {
  const OsType& os = getOSInfo();
  if (os == windows) {
    cout << endl << "== Windows Display Drivers" << endl;
    cout << "Hostname      :";
    system("hostname");
    system("wmic path win32_VideoController get AdapterCompatibility,"
    "InstalledDisplayDrivers,Name | findstr /B /C:\"Advanced Micro Devices\"");
  } else {
    cout << endl << "== Linux Kernel" << endl;
    cout << "Hostname      :" << endl;
    system("hostname");
    system("uname -a");
  }
}


// prints the help text
void HipBinBase::printUsage() const {
  cout << "usage: hipconfig [OPTIONS]\n";
  cout << "  --path,  -p        : print HIP_PATH (use env var if set, else determine from hipconfig path)\n";
  cout << "  --rocmpath,  -R    : print ROCM_PATH (use env var if set, else determine from hip path or /opt/rocm)\n";
  cout << "  --cpp_config, -C   : print C++ compiler options\n";
  cout << "  --compiler, -c     : print compiler (clang or nvcc)\n";
  cout << "  --platform, -P     : print platform (amd or nvidia)\n";
  cout << "  --runtime, -r      : print runtime (rocclr or cuda)\n";
  cout << "  --hipclangpath, -l : print HIP_CLANG_PATH\n";
  cout << "  --full, -f         : print full config\n";
  cout << "  --version, -v      : print hip version\n";
  cout << "  --check            : check configuration\n";
  cout << "  --newline, -n      : print newline\n";
  cout << "  --help, -h         : print help message\n";
}

// reads the config file and stores it in a map for access
map<string, string> HipBinBase::parseConfigFile(fs::path configPath) const {
  map<string, string> configMap;
  ifstream isFile(configPath.string());
  std::string line;
  if (isFile.is_open()) {
    while (std::getline(isFile, line)) {
      std::istringstream is_line(line);
      std::string key;
      if (std::getline(is_line, key, '=')) {
        std::string value;
        if (std::getline(is_line, value)) {
          configMap.insert({ key, value });
        }
      }
    }
    isFile.close();
  }
  return configMap;
}

// Delete all created temporary files
void HipBinBase::deleteTempFiles() {
  try {
    // Deleting temp files vs the temp directory
    for (unsigned int i = 0; i < tmpFiles_.size(); i++) {
      fs::remove(tmpFiles_.at(i));
    }
  }
  catch(...) {
    for (unsigned int i = 0; i < tmpFiles_.size(); i++) {
      cout << "temp tmpFiles_ name: "<< tmpFiles_.at(i) <<endl;
    }
  }
}

// Create a new temporary directory and return it
string HipBinBase::getTempDir() {
  // mkdtemp is only applicable for unix and not windows.
  // Using filesystem becasuse of windows limitation
  string tmpdir = fs::temp_directory_path().string();
  tmpDirs_.push_back(tmpdir);
  return tmpdir;
}

// executes the command, returns the status and return string
SystemCmdOut HipBinBase::exec(const char* cmd, bool printConsole = false) const {
  SystemCmdOut sysOut;
  try {
    char buffer[128];
    std::string result = "";
    #if defined(_WIN32) || defined(_WIN64)
      FILE* pipe = _popen(cmd, "r");
    #else
      FILE* pipe = popen(cmd, "r");
    #endif
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
      while (fgets(buffer, sizeof buffer, pipe) != NULL) {
        result += buffer;
      }
    } catch (...) {
      cout << "Error while executing the command: " <<cmd <<endl;
    }
    #if defined(_WIN32) || defined(_WIN64)
      sysOut.exitCode = _pclose(pipe);
    #else
      int closeStatus = pclose(pipe);
      sysOut.exitCode =  WEXITSTATUS(closeStatus);
    #endif
    if (printConsole == true) {
      cout << result << endl;
    }
    sysOut.out = result;
  }
  catch(...) {
    sysOut.exitCode = -1;
  }
  return sysOut;
}


// compiler canRun or not
bool HipBinBase::canRun(string exeName, string& cmdOut) {
  string complierName = exeName;
  string temp_dir = getTempDir();
  fs::path templateFs = temp_dir;
  templateFs /= "canRunXXXXXX";
  string tmpFileName = mktempFile(templateFs.string());
  complierName += " --version > " + tmpFileName + " 2>&1";
  bool executable = false;
  if (system(const_cast<char*>(complierName.c_str()))) {
    executable = false;
  } else {
    string myline;
    std::ifstream fp;
    fp.open(tmpFileName);
    if (fp.is_open()) {
      while (fp) {
        std::getline(fp, myline);
        cmdOut += myline;
      }
    }
    fp.close();
    if (remove(tmpFileName.c_str()) != 0) {
      cout << "Error deleting file: " << tmpFileName << endl;
    }
    executable = true;
  }
  return executable;
}


// reads envirnoment variables
void HipBinBase::readEnvVariables() {
  if (const char* path = std::getenv(PATH))
    envVariables_.path_ = path;
  if (const char* hip = std::getenv(HIP_PATH))
    envVariables_.hipPathEnv_ = hip;
  if (const char* hip_rocclr = std::getenv(HIP_ROCCLR_HOME))
    envVariables_.hipRocclrPathEnv_ = hip_rocclr;
  if (const char* roccm = std::getenv(ROCM_PATH))
    envVariables_.roccmPathEnv_ = roccm;
  if (const char* cuda = std::getenv(CUDA_PATH))
    envVariables_.cudaPathEnv_ = cuda;
  if (const char* hsa = std::getenv(HSA_PATH))
    envVariables_.hsaPathEnv_ = hsa;
  if (const char* hipClang = std::getenv(HIP_CLANG_PATH))
    envVariables_.hipClangPathEnv_ = hipClang;
  if (const char* hipPlatform = std::getenv(HIP_PLATFORM))
    envVariables_.hipPlatformEnv_ = hipPlatform;
  if (const char* hipCompiler = std::getenv(HIP_COMPILER))
    envVariables_.hipCompilerEnv_ = hipCompiler;
  if (const char* hipRuntime = std::getenv(HIP_RUNTIME))
    envVariables_.hipRuntimeEnv_ = hipRuntime;
  if (const char* ldLibaryPath = std::getenv(LD_LIBRARY_PATH))
    envVariables_.ldLibraryPathEnv_ = ldLibaryPath;
  if (const char* hccAmdGpuTarget = std::getenv(HCC_AMDGPU_TARGET))
    envVariables_.hccAmdGpuTargetEnv_ = hccAmdGpuTarget;
  if (const char* verbose = std::getenv(HIPCC_VERBOSE))
    envVariables_.verboseEnv_ = verbose;
  if (const char* hipccCompileFlagsAppend = std::getenv(HIPCC_COMPILE_FLAGS_APPEND))
    envVariables_.hipccCompileFlagsAppendEnv_ = hipccCompileFlagsAppend;
  if (const char* hipccLinkFlagsAppend = std::getenv(HIPCC_LINK_FLAGS_APPEND))
    envVariables_.hipccLinkFlagsAppendEnv_ = hipccLinkFlagsAppend;
  if (const char* hipLibPath = std::getenv(HIP_LIB_PATH))
    envVariables_.hipLibPathEnv_ = hipLibPath;
  if (const char* deviceLibPath = std::getenv(DEVICE_LIB_PATH))
    envVariables_.deviceLibPathEnv_ = deviceLibPath;
  if (const char* hipClangHccCompactMode = std::getenv(HIP_CLANG_HCC_COMPAT_MODE))
    envVariables_.hipClangHccCompactModeEnv_ = hipClangHccCompactMode;
  if (const char* hipCompileCxxAsHip = std::getenv(HIP_COMPILE_CXX_AS_HIP))
    envVariables_.hipCompileCxxAsHipEnv_ = hipCompileCxxAsHip;
}

// returns envirnoment variables
const EnvVariables& HipBinBase::getEnvVariables() const {
  return envVariables_;
}

// detects the OS information
void HipBinBase::readOSInfo() {
#if defined _WIN32 || defined  _WIN64
  osInfo_ = windows;
#elif  defined __unix || defined __linux__
  osInfo_ = lnx;
#endif
}


// returns the os information
const OsType& HipBinBase::getOSInfo() const {
  return osInfo_;
}

// constructs the HIP path
void HipBinBase::constructHipPath() {
  fs::path full_path(getSelfPath());
  if (envVariables_.hipPathEnv_.empty())
    variables_.hipPathEnv_ = (full_path.parent_path()).string();
  else
    variables_.hipPathEnv_ = envVariables_.hipPathEnv_;
}


// constructs the ROCM path
void HipBinBase::constructRoccmPath() {
  if (envVariables_.roccmPathEnv_.empty()) {
    const string& hipPath = getHipPath();
    fs::path roccm_path(hipPath);
    roccm_path = roccm_path.parent_path();
    fs::path rocm_agent_enumerator_file(roccm_path);
    rocm_agent_enumerator_file /= "bin/rocm_agent_enumerator";
    if (!fs::exists(rocm_agent_enumerator_file)) {
      roccm_path = "/opt/rocm";
    }
    variables_.roccmPathEnv_ = roccm_path.string();
  } else {
    variables_.roccmPathEnv_ = envVariables_.roccmPathEnv_;}
}

// returns the HIP path
const string& HipBinBase::getHipPath() const {
  return variables_.hipPathEnv_;
}

// returns the Roccm path
const string& HipBinBase::getRoccmPath() const {
  return variables_.roccmPathEnv_;
}


// returns the Rocclr Home path
void HipBinBase::constructRocclrHomePath() {
  fs::path full_path(fs::current_path());
  fs::path hipvars_dir = full_path;
  fs::path bitcode = hipvars_dir;
  string rocclrHomePath;
  if (envVariables_.hipRocclrPathEnv_.empty()) {
    bitcode /= "../lib/bitcode";
    if (!fs::exists(bitcode)) {
      rocclrHomePath = getHipPath();
    } else {
      hipvars_dir /= "..";
      rocclrHomePath = hipvars_dir.string();
    }
  } else {
    rocclrHomePath = envVariables_.hipRocclrPathEnv_;
  }
  variables_.hipRocclrPathEnv_ = rocclrHomePath;
}




// returns the Rocclr Home path
const string& HipBinBase::getRocclrHomePath() const {
  return variables_.hipRocclrPathEnv_;
}

// construct hsa Path
void HipBinBase::constructHsaPath() {
  fs::path hsaPathfs;
  string hsaPath;
  if (envVariables_.hsaPathEnv_.empty()) {
    hsaPath = getRoccmPath();
    hsaPathfs = hsaPath;
    hsaPathfs /= "hsa";
    hsaPath = hsaPathfs.string();
    variables_.hsaPathEnv_ = hsaPath;
  } else {
    variables_.hsaPathEnv_ = envVariables_.hsaPathEnv_;
  }
}



// returns hsa Path
const string& HipBinBase::getHsaPath() const {
  return variables_.hsaPathEnv_;
}


// returns the value of the key from the Map passed
string HipBinBase::readConfigMap(map<string, string> hipVersionMap, string keyName, string defaultValue) const {
  string value;
  try {
    value = hipVersionMap.at(keyName);
  }
  catch (...) {
    value = defaultValue;
  }
  return value;
}


// reads the Hip Version
void HipBinBase::readHipVersion() {
  string hipVersion;
  const string& hipPath = getHipPath();
  fs::path hipVersionPath = hipPath;
  hipVersionPath /= "bin/.hipVersion";
  map<string, string> hipVersionMap;
  hipVersionMap = parseConfigFile(hipVersionPath);
  string hip_version_major, hip_version_minor, hip_version_patch, hip_version_githash;
  hip_version_major = readConfigMap(hipVersionMap, "HIP_VERSION_MAJOR", HIP_BASE_VERSION_MAJOR);
  hip_version_minor = readConfigMap(hipVersionMap, "HIP_VERSION_MINOR", HIP_BASE_VERSION_MINOR);
  hip_version_patch = readConfigMap(hipVersionMap, "HIP_VERSION_PATCH", HIP_BASE_VERSION_PATCH);
  hip_version_githash = readConfigMap(hipVersionMap, "HIP_VERSION_GITHASH", HIP_BASE_VERSION_GITHASH);
  hipVersion = hip_version_major + "." + hip_version_minor + "." + hip_version_patch + "-" + hip_version_githash;
  hipVersion_ = hipVersion;
}



// returns the Hip Version
const string& HipBinBase::getHipVersion() const {
  return hipVersion_;
}


#endif  // SRC_HIPBIN_BASE_H_

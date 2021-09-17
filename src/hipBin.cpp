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

#include "hipBin_amd.h"
#include "hipBin_nvidia.h"
#include <vector>
#include <string>

class HipBinBase;
class HipBinAmd;
class HipBinNvidia;
class HipBin;


class HipBin {
 private:
  vector<HipBinBase*> hipBinBasePtrs_;
  vector<PlatformInfo> platformVec_;
  HipBinBase* hipBinNVPtr_;
  HipBinBase* hipBinAMDPtr_;

 public:
  HipBin();
  ~HipBin();
  vector<HipBinBase*>& getHipBinPtrs();
  vector<PlatformInfo>& getPlaformInfo();

  HipBinCommand gethipconfigCmd(string argument);
  bool checkCmd(const vector<string>& commands, const string& argument);
  void executeHipConfig(int argc, char* argv[]);
  void executeHipCC(int argc, char* argv[]);
  bool substringPresent(string fullString, string subString) const;

};


// Implementation ================================================
//===========================================================================

HipBin::HipBin() {
  hipBinNVPtr_ = new HipBinNvidia();
  hipBinAMDPtr_ = new HipBinAmd();
  bool platformDetected = false;
  if (hipBinAMDPtr_->detectPlatform()) {
    const PlatformInfo& platformInfo = hipBinAMDPtr_->getPlatformInfo();   // populates the struct with AMD info
    platformVec_.push_back(platformInfo);
    hipBinBasePtrs_.push_back(hipBinAMDPtr_);
    platformDetected = true;
  } else if (hipBinNVPtr_->detectPlatform()) {
    const PlatformInfo& platformInfo = hipBinNVPtr_->getPlatformInfo();  // populates the struct with Nvidia info
    platformVec_.push_back(platformInfo);
    hipBinBasePtrs_.push_back(hipBinNVPtr_);
    platformDetected = true;
  }
  // if no device is detected, then it is defaulted to AMD
  if (!platformDetected) {
    cout << "Device not supported - Defaulting to AMD" << endl;
    const PlatformInfo& platformInfo = hipBinAMDPtr_->getPlatformInfo();   // populates the struct with AMD info
    platformVec_.push_back(platformInfo);
    hipBinBasePtrs_.push_back(hipBinAMDPtr_);
  }
}

HipBin::~HipBin() {
  delete hipBinNVPtr_;
  delete hipBinAMDPtr_;
  hipBinBasePtrs_.clear();  // clearing the vector so no one accesses the pointers
  platformVec_.clear();     // clearing the platform vector as the pointers are deleted
}


// subtring is present in string
bool HipBin::substringPresent(string fullString, string subString) const {
  bool found = false;
  if (fullString.find(subString) != std::string::npos) {
    found = true;
  }
  return found;
}

vector<PlatformInfo>& HipBin::getPlaformInfo() {
  return platformVec_;  // Return the populated platform info.
}


vector<HipBinBase*>& HipBin::getHipBinPtrs() {
  return hipBinBasePtrs_;  // Return the populated device pointers.
}


void HipBin::executeHipCC(int argc, char* argv[]) {
  cout << "HIPCC executable" <<endl;
  vector<HipBinBase*>& platformPtrs = getHipBinPtrs();
  std::vector<string> argvcc;
  for (int i = 0; i < argc; i++) {
    argvcc.push_back(argv[i]);
  }
  // 0th index points to the first platform detected.
  // In the near future this vector will contain mulitple devices
  platformPtrs.at(0)->executeHipCCCmd(argvcc);
}


void HipBin::executeHipConfig(int argc, char* argv[]) {
  vector<HipBinBase*>& platformPtrs = getHipBinPtrs();
  for (unsigned int j = 0; j < platformPtrs.size(); j++) {
    if (argc == 1) {
      platformPtrs.at(j)->printFull();
    }
    for (int i = 1; i < argc; ++i) {
      HipBinCommand cmd;
      cmd = gethipconfigCmd(argv[i]);
      switch (cmd) {
      case help: platformPtrs.at(j)->printUsage();
        break;
      case path: {
        string hipPath;
        hipPath = platformPtrs.at(j)->getHipPath();
        cout << hipPath;
        break;
      }
      case roccmpath: {
        string roccmPath;
        roccmPath = platformPtrs.at(j)->getRoccmPath();
        cout << roccmPath;
        break;
      }
      case cpp_config: {
        string cppConfig;
        cppConfig = platformPtrs.at(j)->getCppConfig();
        cout << cppConfig;
        break;
      }
      case compiler: {
        PlatformInfo platform;
        platform = platformPtrs.at(j)->getPlatformInfo();
        cout << CompilerTypeStr(platform.compiler);
        break;
      }
      case platform: {
        PlatformInfo platform;
        platform = platformPtrs.at(j)->getPlatformInfo();
        cout << PlatformTypeStr(platform.platform);
        break;
      }
      case runtime: {
        PlatformInfo platform;
        platform = platformPtrs.at(j)->getPlatformInfo();
        cout << RuntimeTypeStr(platform.runtime);
        break;
      }
      case hipclangpath: {
        string compilerPath;
        compilerPath = platformPtrs.at(j)->getCompilerPath();
        cout << compilerPath;
        break;
      }
      case full: platformPtrs.at(j)->printFull();
        break;
      case version: {
        string hipVersion;
        hipVersion = platformPtrs.at(j)->getHipVersion();
        cout << hipVersion;
        break;
      }
      case check: platformPtrs.at(j)->checkHipconfig();
        break;
      case newline: platformPtrs.at(j)->printFull();
        cout << endl;
        break;
      default:
        platformPtrs.at(j)->printUsage();
        break;
      }
    }
  }
}


bool HipBin::checkCmd(const vector<string>& commands, const string& argument) {
  bool found = false;
  for (unsigned int i = 0; i < commands.size(); i++) {
    if (argument.compare(commands.at(i)) == 0) {
      found = true;
      break;
    }
  }
  return found;
}


HipBinCommand HipBin::gethipconfigCmd(string argument) {
  vector<string> pathStrs = { "-p", "--path", "-path", "--p" };
  if (checkCmd(pathStrs, argument))
    return path;
  vector<string> rocmPathStrs = { "-R", "--rocmpath", "-rocmpath", "--R" };
  if (checkCmd(rocmPathStrs, argument))
    return roccmpath;
  vector<string> cppConfigStrs = { "-C", "--cpp_config", "-cpp_config", "--C", };
  if (checkCmd(cppConfigStrs, argument))
    return cpp_config;
  vector<string> CompilerStrs = { "-c", "--compiler", "-compiler", "--c" };
  if (checkCmd(CompilerStrs, argument))
    return compiler;
  vector<string> platformStrs = { "-P", "--platform", "-platform", "--P" };
  if (checkCmd(platformStrs, argument))
    return platform;
  vector<string> runtimeStrs = { "-r", "--runtime", "-runtime", "--r" };
  if (checkCmd(runtimeStrs, argument))
    return runtime;
  vector<string> hipClangPathStrs = { "-l", "--hipclangpath", "-hipclangpath", "--l" };
  if (checkCmd(hipClangPathStrs, argument))
    return hipclangpath;
  vector<string> fullStrs = { "-f", "--full", "-full", "--f" };
  if (checkCmd(fullStrs, argument))
    return full;
  vector<string> versionStrs = { "-v", "--version", "-version", "--v" };
  if (checkCmd(versionStrs, argument))
    return version;
  vector<string> checkStrs = { "--check", "-check" };
  if (checkCmd(checkStrs, argument))
    return check;
  vector<string> newlineStrs = { "--n", "-n", "--newline", "-newline" };
  if (checkCmd(newlineStrs, argument))
    return newline;
  vector<string> helpStrs = { "-h", "--help", "-help", "--h" };
  if (checkCmd(helpStrs, argument))
    return help;
  return full;  // default is full. return full if no commands are matched
}

//===========================================================================
//===========================================================================

int main(int argc, char* argv[]) {
  fs::path filename(argv[0]);
  filename = filename.filename();
  HipBin hipBin;
  // checking the exe name based on that we decide which one to execute
  if (hipBin.substringPresent(filename.string(), "hipconfig")) {
    hipBin.executeHipConfig(argc, argv);
  } else if (hipBin.substringPresent(filename.string(), "hipcc")) {
    hipBin.executeHipCC(argc, argv);
  } else {
    cout << "Command " << filename.string()
    << " not supported. Name the exe as hipconfig or hipcc and then try again ..." << endl;
  }
}

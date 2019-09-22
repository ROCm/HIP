/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <sstream>
#include <regex>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "CUDA2HIP.h"
#include "CUDA2HIP_Scripting.h"
#include "ArgParse.h"
#include "StringUtils.h"
#include "LLVMCompat.h"
#include "Statistics.h"

using namespace llvm;

namespace perl {

  const std::string space = "    ";
  const std::string double_space = space + space;
  const std::string triple_space = double_space + space;
  const std::string sSub = "sub";
  const std::string sReturn_0 = "return 0;\n";
  const std::string sReturn_m = "return $m;\n";
  const std::string sForeach = "foreach $func (\n";
  const std::string sMy = "my $m = 0;\n";

  void generateSymbolFunctions(std::unique_ptr<std::ostream>& perlStreamPtr) {
    *perlStreamPtr.get() << "\n" << sSub << " transformSymbolFunctions\n" << "{\n" << space << sMy;
    std::string sCommon = space + sForeach;
    std::set<std::string> &funcSet = DeviceSymbolFunctions0;
    for (int i = 0; i < 2; ++i) {
      *perlStreamPtr.get() << sCommon;
      if (i == 1) funcSet = DeviceSymbolFunctions1;
      unsigned int count = 0;
      for (auto& f : funcSet) {
        const auto found = CUDA_RUNTIME_FUNCTION_MAP.find(f);
        if (found != CUDA_RUNTIME_FUNCTION_MAP.end()) {
          *perlStreamPtr.get() << (count ? ",\n" : "") << double_space << "\"" << found->second.hipName.str() << "\"";
          count++;
        }
      }
      *perlStreamPtr.get() << "\n" << space << ")\n";
      *perlStreamPtr.get() << space << "{\n" << double_space;
      if (i ==0) *perlStreamPtr.get() << "$m += s/(?<!\\/\\/ CHECK: )($func)\\s*\\(\\s*([^,]+)\\s*,/$func\\(HIP_SYMBOL\\($2\\),/g\n";
      else *perlStreamPtr.get() << "$m += s/(?<!\\/\\/ CHECK: )($func)\\s*\\(\\s*([^,]+)\\s*,\\s*([^,\\)]+)\\s*(,\\s*|\\))\\s*/$func\\($2, HIP_SYMBOL\\($3\\)$4/g;\n";
      *perlStreamPtr.get() << space << "}\n";
    }
    *perlStreamPtr.get() << space << sReturn_m;
    *perlStreamPtr.get() << "}\n";
  }

  void generateDeviceFunctions(std::unique_ptr<std::ostream>& perlStreamPtr) {
    unsigned int countUnsupported = 0;
    unsigned int countSupported = 0;
    std::stringstream sSupported;
    std::stringstream sUnsupported;
    for (auto& ma : CUDA_DEVICE_FUNC_MAP) {
      bool isUnsupported = Statistics::isUnsupported(ma.second);
      (isUnsupported ? sUnsupported : sSupported) << ((isUnsupported && countUnsupported) || (!isUnsupported && countSupported) ? ",\n" : "") << double_space << "\"" << ma.first.str() << "\"";
      if (isUnsupported) {
        countUnsupported++;
      } else {
        countSupported++;
      }
    }
    std::stringstream subCountSupported;
    std::stringstream subWarnUnsupported;
    std::stringstream subCommon;
    std::string sCommon = space + sMy + space + sForeach;
    subCountSupported << "\n" << sSub << " countSupportedDeviceFunctions\n" << "{\n" << (countSupported ? sCommon : space + sReturn_0);
    subWarnUnsupported << "\n" << sSub << " warnUnsupportedDeviceFunctions\n" << "{\n" << (countUnsupported ? space + "my $line_num = shift;\n" + sCommon : space + sReturn_0);
    if (countSupported) {
      subCountSupported << sSupported.str() << "\n" << space << ")\n";
    }
    if (countUnsupported) {
      subWarnUnsupported << sUnsupported.str() << "\n" << space << ")\n";
    }
    if (countSupported || countUnsupported) {
      subCommon << space << "{\n";
      subCommon << double_space << "# match device function from the list, except those, which have a namespace prefix (aka somenamespace::umin(...));\n";
      subCommon << double_space << "# function with only global namespace qualifier '::' (aka ::umin(...)) should be treated as a device function (and warned as well as without such qualifier);\n";
      subCommon << double_space << "my $mt_namespace = m/(\\w+)::($func)\\s*\\(\\s*.*\\s*\\)/g;\n";
      subCommon << double_space << "my $mt = m/($func)\\s*\\(\\s*.*\\s*\\)/g;\n";
      subCommon << double_space << "if ($mt && !$mt_namespace) {\n";
      subCommon << triple_space << "$m += $mt;\n";
    }
    if (countSupported) {
      subCountSupported << subCommon.str();
    }
    if (countUnsupported) {
      subWarnUnsupported << subCommon.str();
      subWarnUnsupported << triple_space << "print STDERR \"  warning: $fileName:$line_num: unsupported device function \\\"$func\\\": $_\\n\";\n";
    }
    if (countSupported || countUnsupported) {
      sCommon = double_space + "}\n" + space + "}\n" + space + sReturn_m;
    }
    if (countSupported) {
      subCountSupported << sCommon;
    }
    if (countUnsupported) {
      subWarnUnsupported << sCommon;
    }
    subCountSupported << "}\n";
    subWarnUnsupported << "}\n";
    *perlStreamPtr.get() << subCountSupported.str();
    *perlStreamPtr.get() << subWarnUnsupported.str();
  }

  bool generate(bool Generate) {
    if (!Generate) {
      return true;
    }
    std::string dstPerlMap = OutputPerlMapFilename, dstPerlMapDir = OutputPerlMapDir;
    if (dstPerlMap.empty()) {
      dstPerlMap = "hipify-perl-map";
    }
    std::error_code EC;
    if (!dstPerlMapDir.empty()) {
      std::string sOutputPerlMapDirAbsPath = getAbsoluteDirectoryPath(OutputPerlMapDir, EC, "output hipify-perl map");
      if (EC) {
        return false;
      }
      dstPerlMap = sOutputPerlMapDirAbsPath + "/" + dstPerlMap;
    }
    SmallString<128> tmpFile;
    StringRef ext = "hipify-tmp";
    EC = sys::fs::createTemporaryFile(dstPerlMap, ext, tmpFile);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
      return false;
    }
    std::unique_ptr<std::ostream> perlStreamPtr = std::unique_ptr<std::ostream>(new std::ofstream(tmpFile.c_str(), std::ios_base::trunc));
    std::string sConv = "my $conversions = ";
    *perlStreamPtr.get() << "@statNames = (";
    for (int i = 0; i < NUM_CONV_TYPES - 1; i++) {
      *perlStreamPtr.get() << "\"" << counterNames[i] << "\", ";
      sConv += "$ft{'" + std::string(counterNames[i]) + "'} + ";
    }
    *perlStreamPtr.get() << "\"" << counterNames[NUM_CONV_TYPES - 1] << "\");\n\n";
    *perlStreamPtr.get() << sConv << "$ft{'" << counterNames[NUM_CONV_TYPES - 1] << "'};\n\n";
    for (int i = 0; i < NUM_CONV_TYPES; i++) {
      if (i == CONV_INCLUDE_CUDA_MAIN_H || i == CONV_INCLUDE) {
        for (auto& ma : CUDA_INCLUDE_MAP) {
          if (Statistics::isUnsupported(ma.second)) {
            continue;
          }
          if (i == ma.second.type) {
            std::string sCUDA = ma.first.str();
            std::string sHIP = ma.second.hipName.str();
            sCUDA = std::regex_replace(sCUDA, std::regex("/"), "\\/");
            sHIP = std::regex_replace(sHIP, std::regex("/"), "\\/");
            *perlStreamPtr.get() << "$ft{'" << counterNames[ma.second.type] << "'} += s/\\b" << sCUDA << "\\b/" << sHIP << "/g;\n";
          }
        }
      }
      else {
        for (auto& ma : CUDA_RENAMES_MAP()) {
          if (Statistics::isUnsupported(ma.second)) {
            continue;
          }
          if (i == ma.second.type) {
            *perlStreamPtr.get() << "$ft{'" << counterNames[ma.second.type] << "'} += s/\\b" << ma.first.str() << "\\b/" << ma.second.hipName.str() << "/g;\n";
          }
        }
      }
    }
    generateSymbolFunctions(perlStreamPtr);
    generateDeviceFunctions(perlStreamPtr);
    perlStreamPtr.get()->flush();
    bool ret = true;
    EC = sys::fs::copy_file(tmpFile, dstPerlMap);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << tmpFile << " to " << dstPerlMap << "\n";
      ret = false;
    }
    if (!SaveTemps) {
      sys::fs::remove(tmpFile);
    }
    return ret;
  }
}

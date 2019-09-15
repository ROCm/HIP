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

  void generateUnsupportedDeviceFunctions(std::unique_ptr<std::ostream>& perlStreamPtr) {
    unsigned int num = 0;
    std::stringstream sUnsupported;
    for (auto& ma : CUDA_DEVICE_FUNC_MAP) {
      if (Statistics::isUnsupported(ma.second)) {
        sUnsupported << (num ? ",\n" : "") << double_space << "\"" << ma.first.str() << "\"";
        num++;
      }
    }
    if (num) {
      *perlStreamPtr.get() << "\nsub warnUnsupportedDeviceFunctions\n" << "{\n" << space << "my $line_num = shift;\n" << space << "my $m = 0;\n" << space << "foreach $func (\n";
      *perlStreamPtr.get() << sUnsupported.str() << "\n" << space << ")\n";
      *perlStreamPtr.get() << space << "{\n";
      *perlStreamPtr.get() << double_space << "# match math at the beginning of a word, but not if it already has a namespace qualifier ('::') :\n";
      *perlStreamPtr.get() << double_space << "my $mt = m/[:]?[:]?\\b($func)\\b(\\w*\\()/g;\n";
      *perlStreamPtr.get() << double_space << "if ($mt) {\n";
      *perlStreamPtr.get() << triple_space << "$m += $mt;\n";
      *perlStreamPtr.get() << triple_space << "print STDERR \"  warning: $fileName:#$line_num : unsupported device function : $_\\n\";\n";
      *perlStreamPtr.get() << double_space << "}\n";
      *perlStreamPtr.get() << space << "}\n";
      *perlStreamPtr.get() << space << "return $m;\n";
      *perlStreamPtr.get() << "}\n";
    }
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
    generateUnsupportedDeviceFunctions(perlStreamPtr);
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

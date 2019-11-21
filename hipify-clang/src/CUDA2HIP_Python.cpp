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

namespace python {

  bool generate(bool Generate) {
    if (!Generate) {
      return true;
    }
    std::string dstPythonMap = "cuda_to_hip_mappings.py", dstPythonMapDir = OutputPythonMapDir;
    std::error_code EC;
    if (!dstPythonMapDir.empty()) {
      std::string sOutputPythonMapDirAbsPath = getAbsoluteDirectoryPath(OutputPythonMapDir, EC, "output hipify-python map");
      if (EC) {
        return false;
      }
      dstPythonMap = sOutputPythonMapDirAbsPath + "/" + dstPythonMap;
    }
    SmallString<128> tmpFile;
    StringRef ext = "hipify-tmp";
    EC = sys::fs::createTemporaryFile(dstPythonMap, ext, tmpFile);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
      return false;
    }
    std::unique_ptr<std::ostream> pythonStreamPtr = std::unique_ptr<std::ostream>(new std::ofstream(tmpFile.c_str(), std::ios_base::trunc));
    *pythonStreamPtr.get() << "import collections\n\n";
    *pythonStreamPtr.get() << "from pyHIPIFY.constants import *\n\n";
    *pythonStreamPtr.get() << "CUDA_RENAMES_MAP = collections.OrderedDict([\n";
    const std::string sHIP_UNS = ", HIP_UNSUPPORTED";
    for (int i = 0; i < NUM_CONV_TYPES; ++i) {
      if (i == CONV_INCLUDE_CUDA_MAIN_H || i == CONV_INCLUDE) {
        for (auto &ma : CUDA_INCLUDE_MAP) {
          if (i == ma.second.type) {
            std::string sUnsupported;
            if (Statistics::isUnsupported(ma.second)) {
              sUnsupported = sHIP_UNS;
            }
            StringRef repName = Statistics::isToRoc(ma.second) ? ma.second.rocName : ma.second.hipName;
            *pythonStreamPtr.get() << "    (\"" << ma.first.str() << "\", (\"" << repName.str() << "\", " << counterTypes[i] << ", " << apiTypes[ma.second.apiType] << sUnsupported << ")),\n";
          }
        }
      }
      else {
        for (auto &ma : CUDA_RENAMES_MAP()) {
          if (i == ma.second.type) {
            std::string sUnsupported;
            if (Statistics::isUnsupported(ma.second)) {
              sUnsupported = sHIP_UNS;
            }
            StringRef repName = Statistics::isToRoc(ma.second) ? ma.second.rocName : ma.second.hipName;
            *pythonStreamPtr.get() << "    (\"" << ma.first.str() << "\", (\"" << repName.str() << "\", " << counterTypes[i] << ", " << apiTypes[ma.second.apiType] << sUnsupported << ")),\n";
          }
        }
      }
    }
    *pythonStreamPtr.get() << "])\n\n";
    *pythonStreamPtr.get() << "CUDA_TO_HIP_MAPPINGS = [CUDA_RENAMES_MAP, C10_MAPPINGS, PYTORCH_SPECIFIC_MAPPINGS]\n";
    pythonStreamPtr.get()->flush();
    bool ret = true;
    EC = sys::fs::copy_file(tmpFile, dstPythonMap);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << tmpFile << " to " << dstPythonMap << "\n";
      ret = false;
    }
    if (!SaveTemps) {
      sys::fs::remove(tmpFile);
    }
    return true;
  }
}

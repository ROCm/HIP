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

#include <cstdio>
#include <fstream>
#include <set>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "CUDA2HIP.h"
#include "LLVMCompat.h"
#include "HipifyAction.h"
#include "ArgParse.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "cuda2hip"

std::string sHipify = "[HIPIFY] ", sConflict = "conflict: ", sError = "error: ";
namespace ct = clang::tooling;

std::string getAbsoluteFilePath(const std::string& sFile, std::error_code& EC) {
  if (sFile.empty()) {
    return sFile;
  }
  if (!sys::fs::exists(sFile)) {
    llvm::errs() << "\n" << sHipify << sError << "source file: " << sFile << " doesn't exist\n";
    EC = std::error_code(static_cast<int>(std::errc::no_such_file_or_directory), std::generic_category());
    return "";
  }
  SmallString<256> fileAbsPath;
  EC = llcompat::real_path(sFile, fileAbsPath, true);
  if (EC) {
    llvm::errs() << "\n" << sHipify << sError << EC.message() << ": source file: " << sFile << "\n";
    return "";
  }
  EC = std::error_code();
  return fileAbsPath.c_str();
}

std::string getAbsoluteDirectoryPath(const std::string& sDir, std::error_code& EC,
                                 const std::string& sDirType = "temporary",
                                 bool bCreateDir = true) {
  if (sDir.empty()) {
    return sDir;
  }
  EC = std::error_code();
  SmallString<256> dirAbsPath;
  if (sys::fs::exists(sDir)) {
    if (sys::fs::is_regular_file(sDir)) {
      llvm::errs() << "\n" << sHipify << sError << sDir << " is not a directory\n";
      EC = std::error_code(static_cast<int>(std::errc::not_a_directory), std::generic_category());
      return "";
    }
  } else {
    if (bCreateDir) {
      EC = sys::fs::create_directory(sDir);
      if (EC) {
        llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << sDirType << " directory: " << sDir << "\n";
        return "";
      }
    } else {
      llvm::errs() << "\n" << sHipify << sError << sDirType << " directory: " << sDir << " doesn't exist\n";
      EC = std::error_code(static_cast<int>(std::errc::no_such_file_or_directory), std::generic_category());
      return "";
    }
  }
  EC = llcompat::real_path(sDir, dirAbsPath, true);
  if (EC) {
    llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << sDirType << " directory: " << sDir << "\n";
    return "";
  }
  return dirAbsPath.c_str();
}

bool generatePerl(bool Generate = true) {
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
          *perlStreamPtr.get() << "$ft{'" << counterNames[ma.second.type] << "'} += s/\\b" << ma.first.str() << "\\b/" << ma.second.hipName.str() << "/g;\n";
        }
      }
    } else {
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

bool generatePython(bool Generate = true) {
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
  for (int i = 0; i < NUM_CONV_TYPES; i++) {
    if (i == CONV_INCLUDE_CUDA_MAIN_H || i == CONV_INCLUDE) {
      for (auto& ma : CUDA_INCLUDE_MAP) {
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
      for (auto& ma : CUDA_RENAMES_MAP()) {
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

int main(int argc, const char **argv) {
  std::vector<const char*> new_argv(argv, argv + argc);
  if (std::find(new_argv.begin(), new_argv.end(), std::string("--")) == new_argv.end()) {
    new_argv.push_back("--");
    new_argv.push_back(nullptr);
    argv = new_argv.data();
    argc++;
  }
  llcompat::PrintStackTraceOnErrorSignal();
  ct::CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory, llvm::cl::Optional);
  std::vector<std::string> fileSources = OptionsParser.getSourcePathList();
  if (fileSources.empty() && !GeneratePerl && !GeneratePython) {
    llvm::errs() << "\n" << sHipify << sError << "Must specify at least 1 positional argument for source file." << "\n";
    return 1;
  }
  if (!generatePerl(GeneratePerl)) {
    llvm::errs() << "\n" << sHipify << sError << "hipify-perl generating failed." << "\n";
    return 1;
  }
  bool bToRoc = TranslateToRoc;
  TranslateToRoc = true;
  bool bToPython = generatePython(GeneratePython);
  TranslateToRoc = bToRoc;
  if (!bToPython) {
    llvm::errs() << "\n" << sHipify << sError << "hipify-python generating failed." << "\n";
    return 1;
  }
  if (fileSources.empty()) {
    return 0;
  }
  std::string dst = OutputFilename, dstDir = OutputDir;
  std::error_code EC;
  std::string sOutputDirAbsPath = getAbsoluteDirectoryPath(OutputDir, EC, "output");
  if (EC) {
    return 1;
  }
  if (!dst.empty()) {
    if (fileSources.size() > 1) {
      llvm::errs() << sHipify << sConflict << "-o and multiple source files are specified.\n";
      return 1;
    }
    if (Inplace) {
      llvm::errs() << sHipify << sConflict << "both -o and -inplace options are specified.\n";
      return 1;
    }
    if (NoOutput) {
      llvm::errs() << sHipify << sConflict << "both -no-output and -o options are specified.\n";
      return 1;
    }
    if (!dstDir.empty()) {
      dst = sOutputDirAbsPath + "/" + dst;
    }
  }
  if (NoOutput && Inplace) {
    llvm::errs() << sHipify << sConflict << "both -no-output and -inplace options are specified.\n";
    return 1;
  }
  if (!dstDir.empty() && Inplace) {
    llvm::errs() << sHipify << sConflict << "both -o-dir and -inplace options are specified.\n";
    return 1;
  }
  if (Examine) {
    NoOutput = PrintStats = true;
  }
  int Result = 0;
  SmallString<128> tmpFile;
  StringRef sourceFileName, ext = "hip";
  std::string sTmpFileName, sSourceAbsPath;
  std::string sTmpDirAbsParh = getAbsoluteDirectoryPath(TemporaryDir, EC);
  if (EC) {
    return 1;
  }
  // Arguments for the Statistics print routines.
  std::unique_ptr<std::ostream> csv = nullptr;
  llvm::raw_ostream* statPrint = nullptr;
  if (!OutputStatsFilename.empty()) {
    csv = std::unique_ptr<std::ostream>(new std::ofstream(OutputStatsFilename, std::ios_base::trunc));
  }
  if (PrintStats) {
    statPrint = &llvm::errs();
  }
  for (const auto & src : fileSources) {
    // Create a copy of the file to work on. When we're done, we'll move this onto the
    // output (which may mean overwriting the input, if we're in-place).
    // Should we fail for some reason, we'll just leak this file and not corrupt the input.
    sSourceAbsPath = getAbsoluteFilePath(src, EC);
    if (EC) {
      continue;
    }
    sourceFileName = sys::path::filename(sSourceAbsPath);
    if (dst.empty()) {
      if (Inplace) {
        dst = src;
      } else {
        dst = src + "." + ext.str();
        if (!dstDir.empty()) {
          dst = sOutputDirAbsPath + "/" + sourceFileName.str() + "." + ext.str();
        }
      }
    }
    if (TemporaryDir.empty()) {
      EC = sys::fs::createTemporaryFile(sourceFileName, ext, tmpFile);
      if (EC) {
        llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
        Result = 1;
        continue;
      }
    } else {
      sTmpFileName = sTmpDirAbsParh + "/" + sourceFileName.str() + "." + ext.str();
      tmpFile = sTmpFileName;
    }
    EC = sys::fs::copy_file(src, tmpFile);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << src << " to " << tmpFile << "\n";
      Result = 1;
      continue;
    }
    // Initialise the statistics counters for this file.
    Statistics::setActive(src);
    // RefactoringTool operates on the file in-place. Giving it the output path is no good,
    // because that'll break relative includes, and we don't want to overwrite the input file.
    // So what we do is operate on a copy, which we then move to the output.
    ct::RefactoringTool Tool(OptionsParser.getCompilations(), std::string(tmpFile.c_str()));
    ct::Replacements& replacementsToUse = llcompat::getReplacements(Tool, tmpFile.c_str());
    ReplacementsFrontendActionFactory<HipifyAction> actionFactory(&replacementsToUse);
    std::string sInclude = "-I" + sys::path::parent_path(sSourceAbsPath).str();
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sInclude.c_str(), ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("cuda", ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-x", ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("--cuda-host-only", ct::ArgumentInsertPosition::BEGIN));
    if (!CudaPath.empty()) {
      std::string sCudaPath = "--cuda-path=" + CudaPath;
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sCudaPath.c_str(), ct::ArgumentInsertPosition::BEGIN));
    }
    // Includes for clang's CUDA wrappers for using by packaged hipify-clang
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("./include", ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-isystem", ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("./include/cuda_wrappers", ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-isystem", ct::ArgumentInsertPosition::BEGIN));
    // Ensure at least c++11 is used.
    std::string stdCpp = "-std=c++11";
#if defined(_MSC_VER)
    stdCpp = "-std=c++14";
#endif
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(stdCpp.c_str(), ct::ArgumentInsertPosition::BEGIN));
#if defined(HIPIFY_CLANG_RES)
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-resource-dir=" HIPIFY_CLANG_RES));
#endif
    if (llcompat::pragma_once_outside_header()) {
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-Wno-pragma-once-outside-header", ct::ArgumentInsertPosition::BEGIN));
    }
    if (!MacroNames.empty()) {
      for (std::string s : MacroNames) {
        Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-D", ct::ArgumentInsertPosition::END));
        Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(s.c_str(), ct::ArgumentInsertPosition::END));
      }
    }
    if (!IncludeDirs.empty()) {
      for (std::string s : IncludeDirs) {
        Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-I", ct::ArgumentInsertPosition::END));
        Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(s.c_str(), ct::ArgumentInsertPosition::END));
      }
    }
    if (Verbose) {
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-v", ct::ArgumentInsertPosition::END));
    }
    Tool.appendArgumentsAdjuster(ct::getClangSyntaxOnlyAdjuster());
    Statistics& currentStat = Statistics::current();
    // Hipify _all_ the things!
    if (Tool.runAndSave(&actionFactory)) {
      currentStat.hasErrors = true;
      LLVM_DEBUG(llvm::dbgs() << "Skipped some replacements.\n");
    }
    // Copy the tmpfile to the output
    if (!NoOutput && !currentStat.hasErrors) {
      EC = sys::fs::copy_file(tmpFile, dst);
      if (EC) {
        llvm::errs() << "\n" << sHipify << sError << EC.message() << ": while copying " << tmpFile << " to " << dst << "\n";
        Result = 1;
        continue;
      }
    }
    // Remove the tmp file without error check
    if (!SaveTemps) {
      sys::fs::remove(tmpFile);
    }
    Statistics::current().markCompletion();
    Statistics::current().print(csv.get(), statPrint);
    dst.clear();
  }
  if (fileSources.size() > 1) {
    Statistics::printAggregate(csv.get(), statPrint);
  }
  return Result;
}

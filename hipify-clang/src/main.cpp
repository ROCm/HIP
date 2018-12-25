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

namespace ct = clang::tooling;

int main(int argc, const char **argv) {
  llcompat::PrintStackTraceOnErrorSignal();
  ct::CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory, llvm::cl::OneOrMore);
  std::vector<std::string> fileSources = OptionsParser.getSourcePathList();
  std::string dst = OutputFilename, sHipify = "[HIPIFY] ", sConflict = "conflict: ", sError = "error: ";
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
  }
  if (NoOutput && Inplace) {
    llvm::errs() << sHipify << sConflict << "both -no-output and -inplace options are specified.\n";
    return 1;
  }
  if (Examine) {
    NoOutput = PrintStats = true;
  }
  int Result = 0;
  std::error_code EC;
  SmallString<128> tmpFile;
  SmallString<256> sourceAbsPath, tmpDirAbsPath;
  StringRef sourceFileName, sourceDir, ext = "hip";
  std::string sTmpDirAbsParh, sTmpFileName;
  if (!TemporaryDir.empty()) {
    EC = sys::fs::real_path(TemporaryDir, tmpDirAbsPath, true);
    if (!EC && sys::fs::is_regular_file(tmpDirAbsPath)) {
      llvm::errs() << "\n" << sHipify << sError << TemporaryDir << " is not a directory\n";
      return 1;
    }
    if (EC) {
      EC = sys::fs::create_directory(TemporaryDir);
      if (EC) {
        llvm::errs() << "\n" << sHipify << sError << EC.message() << ": temporary directory: " << TemporaryDir << "\n";
        return 1;
      }
      EC = sys::fs::real_path(TemporaryDir, tmpDirAbsPath, true);
      if (EC) {
        llvm::errs() << "\n" << sHipify << sError << EC.message() << ": temporary directory: " << TemporaryDir << "\n";
        return 1;
      }
    }
    sTmpDirAbsParh = tmpDirAbsPath.c_str();
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
  int FD;
  for (const auto & src : fileSources) {
    if (dst.empty()) {
      if (Inplace) {
        dst = src;
      } else {
        dst = src + "." + ext.str();
      }
    }
    // Create a copy of the file to work on. When we're done, we'll move this onto the
    // output (which may mean overwriting the input, if we're in-place).
    // Should we fail for some reason, we'll just leak this file and not corrupt the input.
    EC = sys::fs::real_path(src, sourceAbsPath, true);
    if (EC) {
      llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << src << "\n";
      Result = 1;
      continue;
    }
    if (TemporaryDir.empty()) {
      EC = sys::fs::createTemporaryFile(src, ext, FD, tmpFile);
      if (EC) {
        llvm::errs() << "\n" << sHipify << sError << EC.message() << ": " << tmpFile << "\n";
        Result = 1;
        continue;
      }
    } else {
      sourceFileName = sys::path::filename(sourceAbsPath);
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
    std::string sInclude = "-I" + sys::path::parent_path(sourceAbsPath).str();
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sInclude.c_str(), ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("--cuda-host-only", ct::ArgumentInsertPosition::BEGIN));
    // Ensure at least c++11 is used.
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-std=c++11", ct::ArgumentInsertPosition::BEGIN));
#if defined(HIPIFY_CLANG_RES)
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-resource-dir=" HIPIFY_CLANG_RES));
#endif
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

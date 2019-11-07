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
#include "CUDA2HIP_Scripting.h"
#include "LLVMCompat.h"
#include "HipifyAction.h"
#include "ArgParse.h"
#include "StringUtils.h"
#include "llvm/Support/Debug.h"
#if LLVM_VERSION_MAJOR < 8
#include "llvm/Support/Path.h"
#endif

constexpr auto DEBUG_TYPE = "cuda2hip";

namespace ct = clang::tooling;

int main(int argc, const char **argv) {
  std::vector<const char*> new_argv(argv, argv + argc);
  if (std::find(new_argv.begin(), new_argv.end(), std::string("--")) == new_argv.end()) {
    new_argv.push_back("--");
    new_argv.push_back(nullptr);
    argv = new_argv.data();
    argc++;
  }
  llcompat::PrintStackTraceOnErrorSignal();
  ct::CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory, llvm::cl::ZeroOrMore);
  if (!llcompat::CheckCompatibility()) {
    return 1;
  }
  std::vector<std::string> fileSources = OptionsParser.getSourcePathList();
  if (fileSources.empty() && !GeneratePerl && !GeneratePython) {
    llvm::errs() << "\n" << sHipify << sError << "Must specify at least 1 positional argument for source file" << "\n";
    return 1;
  }
  if (!perl::generate(GeneratePerl)) {
    llvm::errs() << "\n" << sHipify << sError << "hipify-perl generating failed" << "\n";
    return 1;
  }
  bool bToRoc = TranslateToRoc;
  TranslateToRoc = true;
  bool bToPython = python::generate(GeneratePython);
  TranslateToRoc = bToRoc;
  if (!bToPython) {
    llvm::errs() << "\n" << sHipify << sError << "hipify-python generating failed" << "\n";
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
      llvm::errs() << sHipify << sConflict << "-o and multiple source files are specified\n";
      return 1;
    }
    if (Inplace) {
      llvm::errs() << sHipify << sConflict << "both -o and -inplace options are specified\n";
      return 1;
    }
    if (NoOutput) {
      llvm::errs() << sHipify << sConflict << "both -no-output and -o options are specified\n";
      return 1;
    }
    if (!dstDir.empty()) {
      dst = sOutputDirAbsPath + "/" + dst;
    }
  }
  if (NoOutput && Inplace) {
    llvm::errs() << sHipify << sConflict << "both -no-output and -inplace options are specified\n";
    return 1;
  }
  if (!dstDir.empty() && Inplace) {
    llvm::errs() << sHipify << sConflict << "both -o-dir and -inplace options are specified\n";
    return 1;
  }
  if (Examine) {
    NoOutput = PrintStats = true;
  }
  int Result = 0;
  SmallString<128> tmpFile;
  StringRef sourceFileName, ext = "hip", csv_ext = "csv";
  std::string sTmpFileName, sSourceAbsPath;
  std::string sTmpDirAbsParh = getAbsoluteDirectoryPath(TemporaryDir, EC);
  if (EC) {
    return 1;
  }
  // Arguments for the Statistics print routines.
  std::unique_ptr<std::ostream> csv = nullptr;
  llvm::raw_ostream* statPrint = nullptr;
  bool create_csv = false;
  if (!OutputStatsFilename.empty()) {
    PrintStatsCSV = true;
    create_csv = true;
  } else {
    if (PrintStatsCSV && fileSources.size() > 1) {
      OutputStatsFilename = "sum_stat.csv";
      create_csv = true;
    }
  }
  if (create_csv) {
    if (!OutputDir.empty()) {
      OutputStatsFilename = sOutputDirAbsPath + "/" + OutputStatsFilename;
    }
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
    if (PrintStatsCSV) {
      if (OutputStatsFilename.empty()) {
        OutputStatsFilename = sourceFileName.str() + "." + csv_ext.str();
        if (!OutputDir.empty()) {
          OutputStatsFilename = sOutputDirAbsPath + "/" + OutputStatsFilename;
        }
      }
      if (!csv) {
        csv = std::unique_ptr<std::ostream>(new std::ofstream(OutputStatsFilename, std::ios_base::trunc));
      }
    }
    // Initialise the statistics counters for this file.
    Statistics::setActive(src);
    // RefactoringTool operates on the file in-place. Giving it the output path is no good,
    // because that'll break relative includes, and we don't want to overwrite the input file.
    // So what we do is operate on a copy, which we then move to the output.
    ct::RefactoringTool Tool(OptionsParser.getCompilations(), std::string(tmpFile.c_str()));
    ct::Replacements& replacementsToUse = llcompat::getReplacements(Tool, tmpFile.c_str());
    ReplacementsFrontendActionFactory<HipifyAction> actionFactory(&replacementsToUse);
    if (!IncludeDirs.empty()) {
      for (std::string s : IncludeDirs) {
        Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(s.c_str(), ct::ArgumentInsertPosition::BEGIN));
        Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-I", ct::ArgumentInsertPosition::BEGIN));
      }
    }
    if (!MacroNames.empty()) {
      for (std::string s : MacroNames) {
        Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(s.c_str(), ct::ArgumentInsertPosition::BEGIN));
        Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-D", ct::ArgumentInsertPosition::BEGIN));
      }
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
    std::string sInclude = "-I" + sys::path::parent_path(sSourceAbsPath).str();
#if defined(HIPIFY_CLANG_RES)
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-resource-dir=" HIPIFY_CLANG_RES, ct::ArgumentInsertPosition::BEGIN));
#endif
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sInclude.c_str(), ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-fno-delayed-template-parsing", ct::ArgumentInsertPosition::BEGIN));
    if (llcompat::pragma_once_outside_header()) {
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-Wno-pragma-once-outside-header", ct::ArgumentInsertPosition::BEGIN));
    }
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("--cuda-host-only", ct::ArgumentInsertPosition::BEGIN));
    if (!CudaGpuArch.empty()) {
      std::string sCudaGpuArch = "--cuda-gpu-arch=" + CudaGpuArch;
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sCudaGpuArch.c_str(), ct::ArgumentInsertPosition::BEGIN));
    }
    if (!CudaPath.empty()) {
      std::string sCudaPath = "--cuda-path=" + CudaPath;
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster(sCudaPath.c_str(), ct::ArgumentInsertPosition::BEGIN));
    }
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("cuda", ct::ArgumentInsertPosition::BEGIN));
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-x", ct::ArgumentInsertPosition::BEGIN));
    if (Verbose) {
      Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-v", ct::ArgumentInsertPosition::END));
    }
    Tool.appendArgumentsAdjuster(ct::getClangSyntaxOnlyAdjuster());
    Statistics& currentStat = Statistics::current();
    // Hipify _all_ the things!
    if (Tool.runAndSave(&actionFactory)) {
      currentStat.hasErrors = true;
      Result = 1;
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

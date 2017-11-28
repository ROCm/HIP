/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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
/**
 * @file Cuda2Hip.cpp
 *
 * This file is compiled and linked into clang based hipify tool.
 */
#include <cstdio>
#include <fstream>
#include <set>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "CUDA2HipMap.h"
#include "LLVMCompat.h"
#include "HipifyAction.h"
#include "ArgParse.h"

#define DEBUG_TYPE "cuda2hip"

namespace ct = clang::tooling;


namespace {

void copyFile(const std::string& src, const std::string& dst) {
  std::ifstream source(src, std::ios::binary);
  std::ofstream dest(dst, std::ios::binary);
  dest << source.rdbuf();
}

} // anonymous namespace

int main(int argc, const char **argv) {
  llcompat::PrintStackTraceOnErrorSignal();

  ct::CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory, llvm::cl::OneOrMore);
  std::vector<std::string> fileSources = OptionsParser.getSourcePathList();
  std::string dst = OutputFilename;
  if (!dst.empty() && fileSources.size() > 1) {
    llvm::errs() << "[HIPIFY] conflict: -o and multiple source files are specified.\n";
    return 1;
  }

  if (NoOutput) {
    if (Inplace) {
      llvm::errs() << "[HIPIFY] conflict: both -no-output and -inplace options are specified.\n";
      return 1;
    }
    if (!dst.empty()) {
      llvm::errs() << "[HIPIFY] conflict: both -no-output and -o options are specified.\n";
      return 1;
    }
  }

  if (Examine) {
    NoOutput = PrintStats = true;
  }

  int Result = 0;

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
    if (dst.empty()) {
      if (Inplace) {
        dst = src;
      } else {
        dst = src + ".hip";
      }
    } else if (Inplace) {
      llvm::errs() << "[HIPIFY] conflict: both -o and -inplace options are specified.\n";
      return 1;
    }

    std::string tmpFile = src + ".hipify-tmp";

    // Create a copy of the file to work on. When we're done, we'll move this onto the
    // output (which may mean overwriting the input, if we're in-place).
    // Should we fail for some reason, we'll just leak this file and not corrupt the input.
    copyFile(src, tmpFile);

    // Initialise the statistics counters for this file.
    Statistics::setActive(src);

    // RefactoringTool operates on the file in-place. Giving it the output path is no good,
    // because that'll break relative includes, and we don't want to overwrite the input file.
    // So what we do is operate on a copy, which we then move to the output.
    ct::RefactoringTool Tool(OptionsParser.getCompilations(), tmpFile);
    ct::Replacements& replacementsToUse = llcompat::getReplacements(Tool, tmpFile);

    ReplacementsFrontendActionFactory<HipifyAction> actionFactory(&replacementsToUse);

    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("--cuda-host-only", ct::ArgumentInsertPosition::BEGIN));

    // Ensure at least c++11 is used.
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-std=c++11", ct::ArgumentInsertPosition::BEGIN));
#if defined(HIPIFY_CLANG_RES)
    Tool.appendArgumentsAdjuster(ct::getInsertArgumentAdjuster("-resource-dir=" HIPIFY_CLANG_RES));
#endif
    Tool.appendArgumentsAdjuster(ct::getClangSyntaxOnlyAdjuster());

    // Hipify _all_ the things!
    if (Tool.runAndSave(&actionFactory)) {
      DEBUG(llvm::dbgs() << "Skipped some replacements.\n");
    }

    // Either move the tmpfile to the output, or remove it.
    if (!NoOutput) {
      rename(tmpFile.c_str(), dst.c_str());
    } else {
      remove(tmpFile.c_str());
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

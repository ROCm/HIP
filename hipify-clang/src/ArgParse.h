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

#pragma once

#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;
namespace ct = clang::tooling;

extern cl::OptionCategory ToolTemplateCategory;
extern cl::opt<std::string> OutputFilename;
extern cl::opt<std::string> OutputDir;
extern cl::opt<std::string> TemporaryDir;
extern cl::opt<std::string> CudaPath;
extern cl::list<std::string> IncludeDirs;
extern cl::list<std::string> MacroNames;
extern cl::opt<bool> Inplace;
extern cl::opt<bool> SaveTemps;
extern cl::opt<bool> NoBackup;
extern cl::opt<bool> NoOutput;
extern cl::opt<bool> PrintStats;
extern cl::opt<std::string> OutputStatsFilename;
extern cl::opt<bool> Examine;
extern cl::extrahelp CommonHelp;
extern cl::opt<bool> TranslateToRoc;
extern cl::opt<bool> DashDash;

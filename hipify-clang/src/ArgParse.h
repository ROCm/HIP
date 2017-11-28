#pragma once

#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;
namespace ct = clang::tooling;

extern cl::OptionCategory ToolTemplateCategory;

extern cl::opt<std::string> OutputFilename;
extern cl::opt<bool> Inplace;
extern cl::opt<bool> NoBackup;
extern cl::opt<bool> NoOutput;
extern cl::opt<bool> PrintStats;
extern cl::opt<std::string> OutputStatsFilename;
extern cl::opt<bool> Examine;

extern cl::extrahelp CommonHelp;

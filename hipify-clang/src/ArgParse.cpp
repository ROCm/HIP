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

#include "ArgParse.h"

cl::OptionCategory ToolTemplateCategory("CUDA to HIP source translator options");

cl::opt<std::string> OutputFilename("o",
  cl::desc("Output filename"),
  cl::value_desc("filename"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> OutputDir("o-dir",
  cl::desc("Output direcory"),
  cl::value_desc("directory"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> TemporaryDir("temp-dir",
  cl::desc("Temporary direcory"),
  cl::value_desc("directory"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> CudaPath("cuda-path",
  cl::desc("CUDA installation path"),
  cl::value_desc("directory"),
  cl::cat(ToolTemplateCategory));

cl::opt <bool> SaveTemps("save-temps",
  cl::desc("Save temporary files"),
  cl::value_desc("save-temps"),
  cl::cat(ToolTemplateCategory));

cl::opt <bool> TranslateToRoc("roc",
  cl::desc("Translate to roc instead of hip where it is possible"),
  cl::value_desc("roc"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> Inplace("inplace",
  cl::desc("Modify input file inplace, replacing input with hipified output, save backup in .prehip file"),
  cl::value_desc("inplace"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> NoBackup("no-backup",
  cl::desc("Don't create a backup file for the hipified source"),
  cl::value_desc("no-backup"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> NoOutput("no-output",
  cl::desc("Don't write any translated output to stdout"),
  cl::value_desc("no-output"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> PrintStats("print-stats",
  cl::desc("Print translation statistics"),
  cl::value_desc("print-stats"),
  cl::cat(ToolTemplateCategory));

cl::opt<std::string> OutputStatsFilename("o-stats",
  cl::desc("Output filename for statistics"),
  cl::value_desc("filename"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> Examine("examine",
  cl::desc("Combines -no-output and -print-stats options"),
  cl::value_desc("examine"),
  cl::cat(ToolTemplateCategory));

cl::opt<bool> DashDash("-",
  cl::desc("Separator between hipify-clang and clang options;\ndon't specify if there are no clang options"),
  cl::value_desc("--"),
  cl::cat(ToolTemplateCategory));

cl::list<std::string> IncludeDirs("I",
  cl::desc("Add directory to include search path"),
  cl::value_desc("directory"),
  cl::ZeroOrMore,
  cl::Prefix,
  cl::cat(ToolTemplateCategory));

cl::list<std::string> MacroNames("D",
  cl::desc("Define <macro> to <value> or 1 if <value> omitted"),
  cl::value_desc("macro>=<value"),
  cl::ZeroOrMore,
  cl::Prefix,
  cl::cat(ToolTemplateCategory));

cl::extrahelp CommonHelp(ct::CommonOptionsParser::HelpMessage);

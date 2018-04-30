#include "ArgParse.h"

cl::OptionCategory ToolTemplateCategory("CUDA to HIP source translator options");

cl::opt<std::string> OutputFilename("o",
  cl::desc("Output filename"),
  cl::value_desc("filename"),
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

cl::extrahelp CommonHelp(ct::CommonOptionsParser::HelpMessage);

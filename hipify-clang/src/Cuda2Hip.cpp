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
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/MacroArgs.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"

#include <cstdio>
#include <fstream>
#include <set>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "CUDA2HipMap.h"
#include "Types.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;
using namespace llvm;

#define DEBUG_TYPE "cuda2hip"

const char *counterNames[CONV_LAST] = {
    "version",      "init",     "device",  "mem",       "kern",        "coord_func", "math_func",
    "special_func", "stream",   "event",   "occupancy", "ctx",         "peer",       "module",
    "cache",        "exec",     "err",     "def",       "tex",         "gl",         "graphics",
    "surface",      "jit",      "d3d9",    "d3d10",     "d3d11",       "vdpau",      "egl",
    "thread",       "other",    "include", "include_cuda_main_header", "type",       "literal",
    "numeric_literal"};

const char *apiNames[API_LAST] = {
    "CUDA Driver API", "CUDA RT API", "CUBLAS API"};

// Set up the command line options
static cl::OptionCategory ToolTemplateCategory("CUDA to HIP source translator options");

static cl::opt<std::string> OutputFilename("o",
  cl::desc("Output filename"),
  cl::value_desc("filename"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> Inplace("inplace",
  cl::desc("Modify input file inplace, replacing input with hipified output, save backup in .prehip file"),
  cl::value_desc("inplace"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> NoBackup("no-backup",
  cl::desc("Don't create a backup file for the hipified source"),
  cl::value_desc("no-backup"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> NoOutput("no-output",
  cl::desc("Don't write any translated output to stdout"),
  cl::value_desc("no-output"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> PrintStats("print-stats",
  cl::desc("Print translation statistics"),
  cl::value_desc("print-stats"),
  cl::cat(ToolTemplateCategory));

static cl::opt<std::string> OutputStatsFilename("o-stats",
  cl::desc("Output filename for statistics"),
  cl::value_desc("filename"),
  cl::cat(ToolTemplateCategory));

static cl::opt<bool> Examine("examine",
  cl::desc("Combines -no-output and -print-stats options"),
  cl::value_desc("examine"),
  cl::cat(ToolTemplateCategory));

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

uint64_t countRepsTotal[CONV_LAST] = { 0 };
uint64_t countApiRepsTotal[API_LAST] = { 0 };
uint64_t countRepsTotalUnsupported[CONV_LAST] = { 0 };
uint64_t countApiRepsTotalUnsupported[API_LAST] = { 0 };
std::map<std::string, uint64_t> cuda2hipConvertedTotal;
std::map<std::string, uint64_t> cuda2hipUnconvertedTotal;

StringRef unquoteStr(StringRef s) {
  if (s.size() > 1 && s.front() == '"' && s.back() == '"')
    return s.substr(1, s.size() - 2);
  return s;
}

/**
 * If `s` starts with `prefix`, remove it. Otherwise, does nothing.
 */
void removePrefixIfPresent(std::string& s, std::string prefix) {
  if (s.find(prefix) != 0) {
    return;
  }

  s.erase(0, prefix.size());
}

class Cuda2Hip {
public:
  Cuda2Hip(Replacements *R, const std::string &srcFileName) :
    Replace(R), mainFileName(srcFileName) {}
  uint64_t countReps[CONV_LAST] = { 0 };
  uint64_t countApiReps[API_LAST] = { 0 };
  uint64_t countRepsUnsupported[CONV_LAST] = { 0 };
  uint64_t countApiRepsUnsupported[API_LAST] = { 0 };
  std::map<std::string, uint64_t> cuda2hipConverted;
  std::map<std::string, uint64_t> cuda2hipUnconverted;
  std::set<unsigned> LOCs;

  enum msgTypes {
    HIPIFY_ERROR = 0,
    HIPIFY_WARNING
  };

  std::string getMsgType(msgTypes type) {
    switch (type) {
      case HIPIFY_ERROR: return "error";
      default:
      case HIPIFY_WARNING: return "warning";
    }
  }

protected:
  Replacements *Replace;
  std::string mainFileName;

  virtual void insertReplacement(const Replacement &rep, const FullSourceLoc &fullSL) {
#if LLVM_VERSION_MAJOR > 3
    // New clang added error checking to Replacements, and *insists* that you explicitly check it.
    llvm::Error e = Replace->add(rep);
#else
    // In older versions, it's literally an std::set<Replacement>
    Replace->insert(rep);
#endif
    if (PrintStats) {
      LOCs.insert(fullSL.getExpansionLineNumber());
    }
  }
  void insertHipHeaders(Cuda2Hip *owner, const SourceManager &SM) {
    if (owner->countReps[CONV_INCLUDE_CUDA_MAIN_H] == 0 && countReps[CONV_INCLUDE_CUDA_MAIN_H] == 0 && Replace->size() > 0) {
      std::string repName = "#include <hip/hip_runtime.h>";
      hipCounter counter = { repName, CONV_INCLUDE_CUDA_MAIN_H, API_RUNTIME };
      updateCounters(counter, repName);
      SourceLocation sl = SM.getLocForStartOfFile(SM.getMainFileID());
      FullSourceLoc fullSL(sl, SM);
      Replacement Rep(SM, sl, 0, repName + "\n");
      insertReplacement(Rep, fullSL);
    }
  }

  void printHipifyMessage(const SourceManager &SM, const SourceLocation &sl, const std::string &message, msgTypes msgType = HIPIFY_WARNING) {
    FullSourceLoc fullSL(sl, SM);
    llvm::errs() << "[HIPIFY] " << getMsgType(msgType) << ": " << mainFileName << ":" << fullSL.getExpansionLineNumber() << ":" << fullSL.getExpansionColumnNumber() << ": " << message << "\n";
  }

  void updateCountersExt(const hipCounter &counter, const std::string &cudaName) {
    std::map<std::string, uint64_t> *map = &cuda2hipConverted;
    std::map<std::string, uint64_t> *mapTotal = &cuda2hipConvertedTotal;
    if (counter.unsupported) {
      map = &cuda2hipUnconverted;
      mapTotal = &cuda2hipUnconvertedTotal;
    }
    auto found = map->find(cudaName);
    if (found == map->end()) {
      map->insert(std::pair<std::string, uint64_t>(cudaName, 1));
    } else {
      found->second++;
    }
    auto foundT = mapTotal->find(cudaName);
    if (foundT == mapTotal->end()) {
      mapTotal->insert(std::pair<std::string, uint64_t>(cudaName, 1));
    } else {
      foundT->second++;
    }
  }

  virtual void updateCounters(const hipCounter &counter, const std::string &cudaName) {
    if (!PrintStats) {
      return;
    }
    updateCountersExt(counter, cudaName);
    if (counter.unsupported) {
      countRepsUnsupported[counter.countType]++;
      countRepsTotalUnsupported[counter.countType]++;
      countApiRepsUnsupported[counter.countApiType]++;
      countApiRepsTotalUnsupported[counter.countApiType]++;
    } else {
      countReps[counter.countType]++;
      countRepsTotal[counter.countType]++;
      countApiReps[counter.countApiType]++;
      countApiRepsTotal[counter.countApiType]++;
    }
  }

  void processString(StringRef s, SourceManager &SM, SourceLocation start) {
    size_t begin = 0;
    while ((begin = s.find("cu", begin)) != StringRef::npos) {
      const size_t end = s.find_first_of(" ", begin + 4);
      StringRef name = s.slice(begin, end);
      const auto found = CUDA_RENAMES_MAP().find(name);
      if (found != CUDA_RENAMES_MAP().end()) {
        StringRef repName = found->second.hipName;
        hipCounter counter = {"", CONV_LITERAL, API_RUNTIME, found->second.unsupported};
        updateCounters(counter, name.str());
        if (!counter.unsupported) {
          SourceLocation sl = start.getLocWithOffset(begin + 1);
          Replacement Rep(SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        // std::string msg = "the following reference is not handled: '" + name.str() + "' [string literal].";
        // printHipifyMessage(SM, start, msg);
      }
      if (end == StringRef::npos) {
        break;
      }
      begin = end + 1;
    }
  }
};

class Cuda2HipCallback;

class HipifyPPCallbacks : public PPCallbacks, public SourceFileCallbacks, public Cuda2Hip {
public:
  HipifyPPCallbacks(Replacements *R, const std::string &mainFileName)
    : Cuda2Hip(R, mainFileName) {}

  virtual bool handleBeginSource(CompilerInstance &CI
#if LLVM_VERSION_MAJOR <= 4
                                 , StringRef Filename
#endif
                                 ) override {
    Preprocessor &PP = CI.getPreprocessor();
    SourceManager &SM = CI.getSourceManager();
    setSourceManager(&SM);
    PP.addPPCallbacks(std::unique_ptr<HipifyPPCallbacks>(this));
    setPreprocessor(&PP);
    return true;
  }

  virtual void handleEndSource() override;

  virtual void InclusionDirective(SourceLocation hash_loc,
                                  const Token &include_token,
                                  StringRef file_name, bool is_angled,
                                  CharSourceRange filename_range,
                                  const FileEntry *file, StringRef search_path,
                                  StringRef relative_path,
                                  const clang::Module *imported) override {
    if (!_sm->isWrittenInMainFile(hash_loc) || !is_angled) {
      return; // We're looking to rewrite angle-includes in the main file to point to hip.
    }

    const auto found = CUDA_INCLUDE_MAP.find(file_name);
    if (found == CUDA_INCLUDE_MAP.end()) {
      // Not a CUDA include - don't touch it.
      return;
    }

    updateCounters(found->second, file_name.str());
    if (found->second.unsupported) {
      // An unsupported CUDA header? Oh dear. Print a warning.
      printHipifyMessage(*_sm, hash_loc, "Unsupported CUDA header used: " + file_name.str());
      return;
    }

    StringRef repName = found->second.hipName;
    DEBUG(dbgs() << "Include file found: " << file_name << "\n"
                 << "SourceLocation: "
                 << filename_range.getBegin().printToString(*_sm) << "\n"
                 << "Will be replaced with " << repName << "\n");
    SourceLocation sl = filename_range.getBegin();
    SourceLocation sle = filename_range.getEnd();
    const char *B = _sm->getCharacterData(sl);
    const char *E = _sm->getCharacterData(sle);
    SmallString<128> tmpData;
    Replacement Rep(*_sm, sl, E - B, Twine("<" + repName + ">").toStringRef(tmpData));
    FullSourceLoc fullSL(sl, *_sm);
    insertReplacement(Rep, fullSL);
  }

  virtual void MacroDefined(const Token &MacroNameTok,
                            const MacroDirective *MD) override {
    if (!_sm->isWrittenInMainFile(MD->getLocation()) ||
        MD->getKind() != MacroDirective::MD_Define) {
      return;
    }

    for (auto T : MD->getMacroInfo()->tokens()) {
      // We're looking for CUDA identifiers in the macro definition to rewrite...
      if (!T.isAnyIdentifier()) {
        continue;
      }

      StringRef name = T.getIdentifierInfo()->getName();
      const auto found = CUDA_RENAMES_MAP().find(name);
      if (found == CUDA_RENAMES_MAP().end()) {
        // So it's an identifier that isn't CUDA? Boring.
        continue;
      }

      updateCounters(found->second, name.str());
      SourceLocation sl = T.getLocation();
      if (found->second.unsupported) {
        // An unsupported identifier? Curses! Warn the user.
        printHipifyMessage(*_sm, sl, "Unsupported CUDA identifier used: " + name.str());
        continue;
      }

      StringRef repName = found->second.hipName;
      DEBUG(dbgs() << "Identifier " << name << " found in definition of macro "
                   << MacroNameTok.getIdentifierInfo()->getName() << "\n"
                   << "will be replaced with: " << repName << "\n"
                   << "SourceLocation: " << sl.printToString(*_sm) << "\n");
      Replacement Rep(*_sm, sl, name.size(), repName);
      FullSourceLoc fullSL(sl, *_sm);
      insertReplacement(Rep, fullSL);
    }
  }

  virtual void MacroExpands(const Token &MacroNameTok,
                            const MacroDefinition &MD, SourceRange Range,
                            const MacroArgs *Args) override {

    if (!_sm->isWrittenInMainFile(MacroNameTok.getLocation())) {
      return; // Macros in headers are not our concern.
    }

    // The getNumArgs function was rather unhelpfully renamed in clang 4.0. Its semantics
    // remain unchanged.
#if LLVM_VERSION_MAJOR > 4
    #define GET_NUM_ARGS() getNumParams()
#else
    #define GET_NUM_ARGS() getNumArgs()
#endif

    for (unsigned int i = 0; Args && i < MD.getMacroInfo()->GET_NUM_ARGS(); i++) {
      std::vector<Token> toks;
      // Code below is a kind of stolen from 'MacroArgs::getPreExpArgument'
      // to workaround the 'const' MacroArgs passed into this hook.
      const Token *start = Args->getUnexpArgument(i);
      size_t len = Args->getArgLength(start) + 1;
#if (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR == 8)
      _pp->EnterTokenStream(start, len, false, false);
#else
      _pp->EnterTokenStream(ArrayRef<Token>(start, len), false);
#endif
      do {
        toks.push_back(Token());
        Token &tk = toks.back();
        _pp->Lex(tk);
      } while (toks.back().isNot(tok::eof));

      _pp->RemoveTopOfLexerStack();
      // end of stolen code

      for (auto tok : toks) {
        if (tok.isAnyIdentifier()) {
          StringRef name = tok.getIdentifierInfo()->getName();
          SourceLocation sl = tok.getLocation();

          const auto found = CUDA_RENAMES_MAP().find(name);
          if (found == CUDA_RENAMES_MAP().end()) {
            // It's not a CUDA identifier. We have nothing to do.
            continue;
          }

          updateCounters(found->second, name.str());
          if (found->second.unsupported) {
            // We know about it, but it isn't supported. Warn the user.
            printHipifyMessage(*_sm, sl, "Unsupported CUDA identifier: " + name.str());
            continue;
          }

          StringRef repName = found->second.hipName;
          DEBUG(dbgs() << "Identifier " << name
                       << " found as an actual argument in expansion of macro "
                       << MacroNameTok.getIdentifierInfo()->getName() << "\n"
                       << "will be replaced with: " << repName << "\n");
          size_t length = name.size();
          if (_sm->isMacroBodyExpansion(sl)) {
            LangOptions DefaultLangOptions;
            SourceLocation sl_macro = _sm->getExpansionLoc(sl);
            SourceLocation sl_end = Lexer::getLocForEndOfToken(sl_macro, 0, *_sm, DefaultLangOptions);
            length = _sm->getCharacterData(sl_end) - _sm->getCharacterData(sl_macro);
            sl = sl_macro;
          }
          Replacement Rep(*_sm, sl, length, repName);
          FullSourceLoc fullSL(sl, *_sm);
          insertReplacement(Rep, fullSL);
        } else if (tok.isLiteral()) {
          SourceLocation sl = tok.getLocation();
          if (_sm->isMacroBodyExpansion(sl)) {
            LangOptions DefaultLangOptions;
            SourceLocation sl_macro = _sm->getExpansionLoc(sl);
            SourceLocation sl_end = Lexer::getLocForEndOfToken(sl_macro, 0, *_sm, DefaultLangOptions);
            size_t length = _sm->getCharacterData(sl_end) - _sm->getCharacterData(sl_macro);
            StringRef name = StringRef(_sm->getCharacterData(sl_macro), length);

            const auto found = CUDA_RENAMES_MAP().find(name);
            if (found == CUDA_RENAMES_MAP().end()) {
              continue; // Not CUDA, we don't care.
            }

            updateCounters(found->second, name.str());
            if (found->second.unsupported) {
              printHipifyMessage(*_sm, sl, "Unsupported CUDA identifier: " + name.str());
              continue;
            }

            StringRef repName = found->second.hipName;
            sl = sl_macro;
            Replacement Rep(*_sm, sl, length, repName);
            FullSourceLoc fullSL(sl, *_sm);
            insertReplacement(Rep, fullSL);
          } else if (tok.is(tok::string_literal)) {
            StringRef s(tok.getLiteralData(), tok.getLength());
            processString(unquoteStr(s), *_sm, tok.getLocation());
          }
        }
      }
    }
  }

  void EndOfMainFile() override {}

  bool SeenEnd = false;
  void setSourceManager(SourceManager *sm) { _sm = sm; }
  void setPreprocessor(Preprocessor *pp) { _pp = pp; }
  void setMatch(Cuda2HipCallback *match) { Match = match; }

private:
  SourceManager *_sm = nullptr;
  Preprocessor *_pp = nullptr;
  Cuda2HipCallback *Match = nullptr;
};

class Cuda2HipCallback : public MatchFinder::MatchCallback, public Cuda2Hip {
private:
  void convertKernelDecl(const FunctionDecl *kernelDecl, const MatchFinder::MatchResult &Result) {
    SourceManager *SM = Result.SourceManager;
    LangOptions DefaultLangOptions;
    SmallString<40> XStr;
    raw_svector_ostream OS(XStr);
    SourceLocation sl = kernelDecl->getNameInfo().getEndLoc();
    SourceLocation kernelArgListStart = Lexer::findLocationAfterToken(sl, tok::l_paren, *SM, DefaultLangOptions, true);
    DEBUG(dbgs() << kernelArgListStart.printToString(*SM));
    if (kernelDecl->getNumParams() > 0) {
      const ParmVarDecl *pvdFirst = kernelDecl->getParamDecl(0);
      const ParmVarDecl *pvdLast =  kernelDecl->getParamDecl(kernelDecl->getNumParams() - 1);
      SourceLocation kernelArgListStart(pvdFirst->getLocStart());
      SourceLocation kernelArgListEnd(pvdLast->getLocEnd());
      SourceLocation stop = Lexer::getLocForEndOfToken(kernelArgListEnd, 0, *SM, DefaultLangOptions);
      size_t repLength = SM->getCharacterData(stop) - SM->getCharacterData(kernelArgListStart);
      OS << StringRef(SM->getCharacterData(kernelArgListStart), repLength);
      Replacement Rep0(*(Result.SourceManager), kernelArgListStart, repLength, OS.str());
      FullSourceLoc fullSL(sl, *(Result.SourceManager));
      insertReplacement(Rep0, fullSL);
    }
  }

  bool cudaCall(const MatchFinder::MatchResult &Result) {
    const CallExpr *call = Result.Nodes.getNodeAs<CallExpr>("cudaCall");
    if (!call) {
      return false; // Another handler will do it.
    }

    const FunctionDecl *funcDcl = call->getDirectCallee();
    std::string name = funcDcl->getDeclName().getAsString();
    SourceManager *SM = Result.SourceManager;
    SourceLocation sl = call->getLocStart();

    // TODO: Make a lookup table just for functions to improve performance.
    const auto found = CUDA_IDENTIFIER_MAP.find(name);
    if (found == CUDA_IDENTIFIER_MAP.end()) {
      std::string msg = "the following reference is not handled: '" + name + "' [function call].";
      printHipifyMessage(*SM, sl, msg);
      return true;
    }

    const hipCounter& hipCtr = found->second;
    updateCounters(found->second, name);

    if (hipCtr.unsupported) {
      return true; // Silently fail when you find an unsupported member.
      // TODO: Print a warning with the diagnostics API?
    }

    size_t length = name.size();
    bool bReplace = true;
    if (SM->isMacroArgExpansion(sl)) {
      sl = SM->getImmediateSpellingLoc(sl);
    } else if (SM->isMacroBodyExpansion(sl)) {
      LangOptions DefaultLangOptions;
      SourceLocation sl_macro = SM->getExpansionLoc(sl);
      SourceLocation sl_end = Lexer::getLocForEndOfToken(sl_macro, 0, *SM, DefaultLangOptions);
      length = SM->getCharacterData(sl_end) - SM->getCharacterData(sl_macro);
      StringRef macroName = StringRef(SM->getCharacterData(sl_macro), length);
      if (CUDA_EXCLUDES.end() != CUDA_EXCLUDES.find(macroName)) {
        bReplace = false;
      } else {
        sl = sl_macro;
      }
    }

    if (bReplace) {
      updateCounters(found->second, name);
      Replacement Rep(*SM, sl, length, hipCtr.hipName);
      FullSourceLoc fullSL(sl, *SM);
      insertReplacement(Rep, fullSL);
    }

    return true;
  }

  bool cudaLaunchKernel(const MatchFinder::MatchResult &Result) {
    StringRef refName = "cudaLaunchKernel";
    if (const CUDAKernelCallExpr *launchKernel = Result.Nodes.getNodeAs<CUDAKernelCallExpr>(refName)) {
      SmallString<40> XStr;
      raw_svector_ostream OS(XStr);
      StringRef calleeName;
      const FunctionDecl *kernelDecl = launchKernel->getDirectCallee();
      if (kernelDecl) {
        calleeName = kernelDecl->getName();
        convertKernelDecl(kernelDecl, Result);
      } else {
        const Expr *e = launchKernel->getCallee();
        if (const UnresolvedLookupExpr *ule =
          dyn_cast<UnresolvedLookupExpr>(e)) {
          calleeName = ule->getName().getAsIdentifierInfo()->getName();
          owner->addMatcher(functionTemplateDecl(hasName(calleeName))
            .bind("unresolvedTemplateName"), this);
        }
      }
      XStr.clear();
      if (calleeName.find(',') != StringRef::npos) {
        SmallString<128> tmpData;
        calleeName = Twine("(" + calleeName + ")").toStringRef(tmpData);
      }
      OS << "hipLaunchKernelGGL(" << calleeName << ",";
      const CallExpr *config = launchKernel->getConfig();
      DEBUG(dbgs() << "Kernel config arguments:" << "\n");
      SourceManager *SM = Result.SourceManager;
      LangOptions DefaultLangOptions;
      for (unsigned argno = 0; argno < config->getNumArgs(); argno++) {
        const Expr *arg = config->getArg(argno);
        if (!isa<CXXDefaultArgExpr>(arg)) {
          const ParmVarDecl *pvd = config->getDirectCallee()->getParamDecl(argno);
          SourceLocation sl(arg->getLocStart());
          SourceLocation el(arg->getLocEnd());
          SourceLocation stop = Lexer::getLocForEndOfToken(el, 0, *SM, DefaultLangOptions);
          StringRef outs(SM->getCharacterData(sl), SM->getCharacterData(stop) - SM->getCharacterData(sl));
          DEBUG(dbgs() << "args[ " << argno << "]" << outs << " <" << pvd->getType().getAsString() << ">\n");
          if (pvd->getType().getAsString().compare("dim3") == 0) {
            OS << " dim3(" << outs << "),";
          } else {
            OS << " " << outs << ",";
          }
        } else {
          OS << " 0,";
        }
      }
      for (unsigned argno = 0; argno < launchKernel->getNumArgs(); argno++) {
        const Expr *arg = launchKernel->getArg(argno);
        SourceLocation sl(arg->getLocStart());
        if (SM->isMacroBodyExpansion(sl)) {
          sl = SM->getExpansionLoc(sl);
        } else if (SM->isMacroArgExpansion(sl)) {
          sl = SM->getImmediateSpellingLoc(sl);
        }
        SourceLocation el(arg->getLocEnd());
        if (SM->isMacroBodyExpansion(el)) {
          el = SM->getExpansionLoc(el);
        } else if (SM->isMacroArgExpansion(el)) {
          el = SM->getImmediateSpellingLoc(el);
        }
        SourceLocation stop = Lexer::getLocForEndOfToken(el, 0, *SM, DefaultLangOptions);
        std::string outs(SM->getCharacterData(sl), SM->getCharacterData(stop) - SM->getCharacterData(sl));
        DEBUG(dbgs() << outs << "\n");
        OS << " " << outs << ",";
      }
      XStr.pop_back();
      OS << ")";
      size_t length = SM->getCharacterData(Lexer::getLocForEndOfToken(
                        launchKernel->getLocEnd(), 0, *SM, DefaultLangOptions)) -
                        SM->getCharacterData(launchKernel->getLocStart());
      Replacement Rep(*SM, launchKernel->getLocStart(), length, OS.str());
      FullSourceLoc fullSL(launchKernel->getLocStart(), *SM);
      insertReplacement(Rep, fullSL);
      hipCounter counter = {"hipLaunchKernelGGL", CONV_KERN, API_RUNTIME};
      updateCounters(counter, refName.str());
      return true;
    }
    return false;
  }

  bool cudaBuiltin(const MatchFinder::MatchResult &Result) {
    if (const MemberExpr *threadIdx = Result.Nodes.getNodeAs<MemberExpr>("cudaBuiltin")) {
      if (const OpaqueValueExpr *refBase =
        dyn_cast<OpaqueValueExpr>(threadIdx->getBase())) {
        if (const DeclRefExpr *declRef =
          dyn_cast<DeclRefExpr>(refBase->getSourceExpr())) {
          SourceLocation sl = threadIdx->getLocStart();
          SourceManager *SM = Result.SourceManager;
          StringRef name = declRef->getDecl()->getName();
          StringRef memberName = threadIdx->getMemberDecl()->getName();
          size_t pos = memberName.find_first_not_of("__fetch_builtin_");
          memberName = memberName.slice(pos, memberName.size());
          SmallString<128> tmpData;
          name = Twine(name + "." + memberName).toStringRef(tmpData);

          // TODO: Make a lookup table just for builtins to improve performance.
          const auto found = CUDA_IDENTIFIER_MAP.find(name);
          if (found != CUDA_IDENTIFIER_MAP.end()) {
            updateCounters(found->second, name.str());
            if (!found->second.unsupported) {
              StringRef repName = found->second.hipName;
              Replacement Rep(*SM, sl, name.size(), repName);
              FullSourceLoc fullSL(sl, *SM);
              insertReplacement(Rep, fullSL);
            }
          } else {
            std::string msg = "the following reference is not handled: '" + name.str() + "' [builtin].";
            printHipifyMessage(*SM, sl, msg);
          }
        }
      }
      return true;
    }
    return false;
  }

  bool cudaEnumConstantRef(const MatchFinder::MatchResult &Result) {
    if (const DeclRefExpr *enumConstantRef = Result.Nodes.getNodeAs<DeclRefExpr>("cudaEnumConstantRef")) {
      StringRef name = enumConstantRef->getDecl()->getName();
      SourceLocation sl = enumConstantRef->getLocStart();
      SourceManager *SM = Result.SourceManager;

      // TODO: Make a lookup table just for enum values to improve performance.
      const auto found = CUDA_IDENTIFIER_MAP.find(name);
      if (found != CUDA_IDENTIFIER_MAP.end()) {
        updateCounters(found->second, name.str());
        if (!found->second.unsupported) {
          StringRef repName = found->second.hipName;
          Replacement Rep(*SM, sl, name.size(), repName);
          FullSourceLoc fullSL(sl, *SM);
          insertReplacement(Rep, fullSL);
        }
      } else {
        std::string msg = "the following reference is not handled: '" + name.str() + "' [enum constant ref].";
        printHipifyMessage(*SM, sl, msg);
      }
      return true;
    }
    return false;
  }

  bool cudaType(const MatchFinder::MatchResult& Result) {
      const clang::TypeLoc* ret = Result.Nodes.getNodeAs<TypeLoc>("cudaType");
      if (!ret) {
          return false;
      }

      // Ignore qualifiers - they don't alter our decision to rename.
      clang::UnqualTypeLoc tl = ret->getUnqualifiedLoc();
      const Type& typeObject = *(tl.getTypePtr());

      std::string typeName = tl.getType().getAsString();

      // Irritatingly, enum/struct types are identified as `enum/struct <something>`, and unlike most compound
      // types (such as pointers or references), there isn't another type node inside. So we have
      // to make do with what we've got. There's probably a better way of doing this...
      if (typeObject.isEnumeralType()) {
        removePrefixIfPresent(typeName, "enum ");
      }
      if (typeObject.isStructureType()) {
        removePrefixIfPresent(typeName, "struct ");
      }

      // Do we have a replacement for this type?
      const auto found = CUDA_TYPE_NAME_MAP.find(typeName);
      if (found == CUDA_TYPE_NAME_MAP.end()) {
          return false;
      }

      SourceManager &SM = *(Result.SourceManager);

      // Start of the type expression to replace.
      SourceLocation sl = tl.getBeginLoc();

      const hipCounter& hipCtr = found->second;
      if (hipCtr.unsupported) {
          printHipifyMessage(SM, sl, "Unsupported CUDA '" + typeName);
          return false;
      }

      // Apply the rename!
      Replacement Rep(SM, sl, typeName.size(), hipCtr.hipName);
      FullSourceLoc fullSL(sl, SM);
      insertReplacement(Rep, fullSL);

      return true;
  }

  bool cudaSharedIncompleteArrayVar(const MatchFinder::MatchResult &Result) {
    StringRef refName = "cudaSharedIncompleteArrayVar";
    if (const VarDecl *sharedVar = Result.Nodes.getNodeAs<VarDecl>(refName)) {
      // Example: extern __shared__ uint sRadix1[];
      if (sharedVar->hasExternalFormalLinkage()) {
        QualType QT = sharedVar->getType();
        std::string typeName;
        if (QT->isIncompleteArrayType()) {
          const ArrayType *AT = QT.getTypePtr()->getAsArrayTypeUnsafe();
          QT = AT->getElementType();
          if (QT.getTypePtr()->isBuiltinType()) {
            QT = QT.getCanonicalType();
            const BuiltinType *BT = dyn_cast<BuiltinType>(QT);
            if (BT) {
              LangOptions LO;
              LO.CUDA = true;
              PrintingPolicy policy(LO);
              typeName = BT->getName(policy);
            }
          } else {
            typeName = QT.getAsString();
          }
        }
        if (!typeName.empty()) {
          SourceLocation slStart = sharedVar->getLocStart();
          SourceLocation slEnd = sharedVar->getLocEnd();
          SourceManager *SM = Result.SourceManager;
          size_t repLength = SM->getCharacterData(slEnd) - SM->getCharacterData(slStart) + 1;
          std::string varName = sharedVar->getNameAsString();
          std::string repName = "HIP_DYNAMIC_SHARED(" + typeName + ", " + varName + ")";
          Replacement Rep(*SM, slStart, repLength, repName);
          FullSourceLoc fullSL(slStart, *SM);
          insertReplacement(Rep, fullSL);
          hipCounter counter = { "HIP_DYNAMIC_SHARED", CONV_MEM, API_RUNTIME };
          updateCounters(counter, refName.str());
        }
      }
      return true;
    }
    return false;
  }

  bool unresolvedTemplateName(const MatchFinder::MatchResult &Result) {
    if (const FunctionTemplateDecl *templateDecl = Result.Nodes.getNodeAs<FunctionTemplateDecl>("unresolvedTemplateName")) {
      FunctionDecl *kernelDecl = templateDecl->getTemplatedDecl();
      convertKernelDecl(kernelDecl, Result);
      return true;
    }
    return false;
  }

  bool stringLiteral(const MatchFinder::MatchResult &Result) {
    if (const clang::StringLiteral *sLiteral = Result.Nodes.getNodeAs<clang::StringLiteral>("stringLiteral")) {
      if (sLiteral->getCharByteWidth() == 1) {
        StringRef s = sLiteral->getString();
        SourceManager *SM = Result.SourceManager;
        processString(s, *SM, sLiteral->getLocStart());
      }
      return true;
    }
    return false;
  }

public:
  Cuda2HipCallback(Replacements *Replace, ast_matchers::MatchFinder *parent, HipifyPPCallbacks *PPCallbacks, const std::string &mainFileName)
    : Cuda2Hip(Replace, mainFileName), owner(parent), PP(PPCallbacks) {
    PP->setMatch(this);
  }

  void run(const MatchFinder::MatchResult &Result) override {
    if (cudaType(Result)) return;
    if (cudaCall(Result)) return;
    if (cudaBuiltin(Result)) return;
    if (cudaEnumConstantRef(Result)) return;
    if (cudaLaunchKernel(Result)) return;
    if (cudaSharedIncompleteArrayVar(Result)) return;
    if (stringLiteral(Result)) return;
    if (unresolvedTemplateName(Result)) return;
  }

private:
  ast_matchers::MatchFinder *owner;
  HipifyPPCallbacks *PP;
};

void HipifyPPCallbacks::handleEndSource() {
  insertHipHeaders(Match, *_sm);
}

void addAllMatchers(ast_matchers::MatchFinder &Finder, Cuda2HipCallback *Callback) {
  // Rewrite CUDA api calls to hip ones.
  Finder.addMatcher(
      callExpr(
          isExpansionInMainFile(),
          callee(
              functionDecl(
                  matchesName("cu.*")
              )
          )
      ).bind("cudaCall"),
      Callback
  );

  // Rewrite all references to CUDA types to their corresponding hip types.
  Finder.addMatcher(
      typeLoc(
          isExpansionInMainFile()
      ).bind("cudaType"),
      Callback
  );

  // Replace references to CUDA names in string literals with the equivalent hip names.
  Finder.addMatcher(stringLiteral(isExpansionInMainFile()).bind("stringLiteral"), Callback);

  // Replace the <<<...>>> language extension with a hip kernel launch
  Finder.addMatcher(cudaKernelCallExpr(isExpansionInMainFile()).bind("cudaLaunchKernel"), Callback);

  // Replace cuda builtins.
  Finder.addMatcher(
      memberExpr(
          isExpansionInMainFile(),
          hasObjectExpression(
              hasType(
                  cxxRecordDecl(
                      matchesName("__cuda_builtin_")
                  )
              )
          )
      ).bind("cudaBuiltin"),
      Callback
  );

  // Map CUDA enum _values_ to their hip equivalents.
  Finder.addMatcher(
      declRefExpr(
          isExpansionInMainFile(),
          to(
              enumConstantDecl(
                  matchesName("cu.*|CU.*")
              )
          )
      ).bind("cudaEnumConstantRef"),
      Callback
  );

  Finder.addMatcher(
      varDecl(
          isExpansionInMainFile(),
          allOf(
              hasAttr(attr::CUDAShared),
              hasType(incompleteArrayType())
          )
      ).bind("cudaSharedIncompleteArrayVar"),
      Callback
  );
}

int64_t printStats(const std::string &csvFile, const std::string &srcFile,
                   HipifyPPCallbacks &PPCallbacks, Cuda2HipCallback &Callback,
                   uint64_t replacedBytes, uint64_t totalBytes, unsigned totalLines,
                   const std::chrono::steady_clock::time_point &start) {
  std::ofstream csv(csvFile, std::ios::app);
  int64_t sum = 0, sum_interm = 0;
  std::string str;
  const std::string hipify_info = "[HIPIFY] info: ", separator = ";";
  for (int i = 0; i < CONV_LAST; i++) {
    sum += Callback.countReps[i] + PPCallbacks.countReps[i];
  }
  int64_t sum_unsupported = 0;
  for (int i = 0; i < CONV_LAST; i++) {
    sum_unsupported += Callback.countRepsUnsupported[i] + PPCallbacks.countRepsUnsupported[i];
  }
  if (sum > 0 || sum_unsupported > 0) {
    str = "file \'" + srcFile + "\' statistics:\n";
    llvm::outs() << "\n" << hipify_info << str;
    csv << "\n" << str;
    str = "CONVERTED refs count";
    llvm::outs() << "  " << str << ": " << sum << "\n";
    csv << "\n" << str << separator << sum << "\n";
    str = "UNCONVERTED refs count";
    llvm::outs() << "  " << str << ": " << sum_unsupported << "\n";
    csv << str << separator << sum_unsupported << "\n";
    str = "CONVERSION %";
    long conv = 100 - std::lround(double(sum_unsupported*100)/double(sum + sum_unsupported));
    llvm::outs() << "  " << str << ": " << conv << "%\n";
    csv << str << separator << conv << "%\n";
    str = "REPLACED bytes";
    llvm::outs() << "  " << str << ": " << replacedBytes << "\n";
    csv << str << separator << replacedBytes << "\n";
    str = "TOTAL bytes";
    llvm::outs() << "  " << str << ": " << totalBytes << "\n";
    csv << str << separator << totalBytes << "\n";
    str = "CHANGED lines of code";
    unsigned changedLines = Callback.LOCs.size() + PPCallbacks.LOCs.size();
    llvm::outs() << "  " << str << ": " << changedLines << "\n";
    csv << str << separator << changedLines << "\n";
    str = "TOTAL lines of code";
    llvm::outs() << "  " << str << ": " << totalLines << "\n";
    csv << str << separator << totalLines << "\n";
    if (totalBytes > 0) {
      str = "CODE CHANGED (in bytes) %";
      conv = std::lround(double(replacedBytes * 100) / double(totalBytes));
      llvm::outs() << "  " << str << ": " << conv << "%\n";
      csv << str << separator << conv << "%\n";
    }
    if (totalLines > 0) {
      str = "CODE CHANGED (in lines) %";
      conv = std::lround(double(changedLines * 100) / double(totalLines));
      llvm::outs() << "  " << str << ": " << conv << "%\n";
      csv << str << separator << conv << "%\n";
    }
    typedef std::chrono::duration<double, std::milli> duration;
    duration elapsed = std::chrono::steady_clock::now() - start;
    str = "TIME ELAPSED s";
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << elapsed.count() / 1000;
    llvm::outs() << "  " << str << ": " << stream.str() << "\n";
    csv << str << separator << stream.str() << "\n";
  }
  if (sum > 0) {
    llvm::outs() << hipify_info << "CONVERTED refs by type:\n";
    csv << "\nCUDA ref type" << separator << "Count\n";
    for (int i = 0; i < CONV_LAST; i++) {
      sum_interm = Callback.countReps[i] + PPCallbacks.countReps[i];
      if (0 == sum_interm) {
        continue;
      }
      llvm::outs() << "  " << counterNames[i] << ": " << sum_interm << "\n";
      csv << counterNames[i] << separator << sum_interm << "\n";
    }
    llvm::outs() << hipify_info << "CONVERTED refs by API:\n";
    csv << "\nCUDA API" << separator << "Count\n";
    for (int i = 0; i < API_LAST; i++) {
      llvm::outs() << "  " << apiNames[i] << ": " << Callback.countApiReps[i] + PPCallbacks.countApiReps[i] << "\n";
      csv << apiNames[i] << separator << Callback.countApiReps[i] + PPCallbacks.countApiReps[i] << "\n";
    }
    for (const auto & it : PPCallbacks.cuda2hipConverted) {
      const auto found = Callback.cuda2hipConverted.find(it.first);
      if (found == Callback.cuda2hipConverted.end()) {
        Callback.cuda2hipConverted.insert(std::pair<std::string, uint64_t>(it.first, 1));
      } else {
        found->second += it.second;
      }
    }
    llvm::outs() << hipify_info << "CONVERTED refs by names:\n";
    csv << "\nCUDA ref name" << separator << "Count\n";
    for (const auto & it : Callback.cuda2hipConverted) {
      llvm::outs() << "  " << it.first << ": " << it.second << "\n";
      csv << it.first << separator << it.second << "\n";
    }
  }
  if (sum_unsupported > 0) {
    str = "UNCONVERTED refs by type:";
    llvm::outs() << hipify_info << str << "\n";
    csv << "\nUNCONVERTED CUDA ref type" << separator << "Count\n";
    for (int i = 0; i < CONV_LAST; i++) {
      sum_interm = Callback.countRepsUnsupported[i] + PPCallbacks.countRepsUnsupported[i];
      if (0 == sum_interm) {
        continue;
      }
      llvm::outs() << "  " << counterNames[i] << ": " << sum_interm << "\n";
      csv << counterNames[i] << separator << sum_interm << "\n";
    }
    llvm::outs() << hipify_info << "UNCONVERTED refs by API:\n";
    csv << "\nUNCONVERTED CUDA API" << separator << "Count\n";
    for (int i = 0; i < API_LAST; i++) {
      llvm::outs() << "  " << apiNames[i] << ": " << Callback.countApiRepsUnsupported[i] + PPCallbacks.countApiRepsUnsupported[i] << "\n";
      csv << apiNames[i] << separator << Callback.countApiRepsUnsupported[i] + PPCallbacks.countApiRepsUnsupported[i] << "\n";
    }
    for (const auto & it : PPCallbacks.cuda2hipUnconverted) {
      const auto found = Callback.cuda2hipUnconverted.find(it.first);
      if (found == Callback.cuda2hipUnconverted.end()) {
        Callback.cuda2hipUnconverted.insert(std::pair<std::string, uint64_t>(it.first, 1));
      } else {
        found->second += it.second;
      }
    }
    llvm::outs() << hipify_info << "UNCONVERTED refs by names:\n";
    csv << "\nUNCONVERTED CUDA ref name" << separator << "Count\n";
    for (const auto & it : Callback.cuda2hipUnconverted) {
      llvm::outs() << "  " << it.first << ": " << it.second << "\n";
      csv << it.first << separator << it.second << "\n";
    }
  }
  csv.close();
  return sum;
}

void printAllStats(const std::string &csvFile, int64_t totalFiles, int64_t convertedFiles,
                   uint64_t replacedBytes, uint64_t totalBytes, unsigned changedLines, unsigned totalLines,
                   const std::chrono::steady_clock::time_point &start) {
  std::ofstream csv(csvFile, std::ios::app);
  int64_t sum = 0, sum_interm = 0;
  std::string str;
  const std::string hipify_info = "[HIPIFY] info: ", separator = ";";
  for (int i = 0; i < CONV_LAST; i++) {
    sum += countRepsTotal[i];
  }
  int64_t sum_unsupported = 0;
  for (int i = 0; i < CONV_LAST; i++) {
    sum_unsupported += countRepsTotalUnsupported[i];
  }
  if (sum > 0 || sum_unsupported > 0) {
    str = "TOTAL statistics:\n";
    llvm::outs() << "\n" << hipify_info << str;
    csv << "\n" << str;
    str = "CONVERTED files";
    llvm::outs() << "  " << str << ": " << convertedFiles << "\n";
    csv << "\n" << str << separator << convertedFiles << "\n";
    str = "PROCESSED files";
    llvm::outs() << "  " << str << ": " << totalFiles << "\n";
    csv << str << separator << totalFiles << "\n";
    str = "CONVERTED refs count";
    llvm::outs() << "  " << str << ": " << sum << "\n";
    csv << str << separator << sum << "\n";
    str = "UNCONVERTED refs count";
    llvm::outs() << "  " << str << ": " << sum_unsupported << "\n";
    csv << str << separator << sum_unsupported << "\n";
    str = "CONVERSION %";
    long conv = 100 - std::lround(double(sum_unsupported * 100) / double(sum + sum_unsupported));
    llvm::outs() << "  " << str << ": " << conv << "%\n";
    csv << str << separator << conv << "%\n";
    str = "REPLACED bytes";
    llvm::outs() << "  " << str << ": " << replacedBytes << "\n";
    csv << str << separator << replacedBytes << "\n";
    str = "TOTAL bytes";
    llvm::outs() << "  " << str << ": " << totalBytes << "\n";
    csv << str << separator << totalBytes << "\n";
    str = "CHANGED lines of code";
    llvm::outs() << "  " << str << ": " << changedLines << "\n";
    csv << str << separator << changedLines << "\n";
    str = "TOTAL lines of code";
    llvm::outs() << "  " << str << ": " << totalLines << "\n";
    csv << str << separator << totalLines << "\n";
    if (totalBytes > 0) {
      str = "CODE CHANGED (in bytes) %";
      conv = std::lround(double(replacedBytes * 100) / double(totalBytes));
      llvm::outs() << "  " << str << ": " << conv << "%\n";
      csv << str << separator << conv << "%\n";
    }
    if (totalLines > 0) {
      str = "CODE CHANGED (in lines) %";
      conv = std::lround(double(changedLines * 100) / double(totalLines));
      llvm::outs() << "  " << str << ": " << conv << "%\n";
      csv << str << separator << conv << "%\n";
    }
    typedef std::chrono::duration<double, std::milli> duration;
    duration elapsed = std::chrono::steady_clock::now() - start;
    str = "TIME ELAPSED s";
    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << elapsed.count() / 1000;
    llvm::outs() << "  " << str << ": " << stream.str() << "\n";
    csv << str << separator << stream.str() << "\n";
  }
  if (sum > 0) {
    llvm::outs() << hipify_info << "CONVERTED refs by type:\n";
    csv << "\nCUDA ref type" << separator << "Count\n";
    for (int i = 0; i < CONV_LAST; i++) {
      sum_interm = countRepsTotal[i];
      if (0 == sum_interm) {
        continue;
      }
      llvm::outs() << "  " << counterNames[i] << ": " << sum_interm << "\n";
      csv << counterNames[i] << separator << sum_interm << "\n";
    }
    llvm::outs() << hipify_info << "CONVERTED refs by API:\n";
    csv << "\nCUDA API" << separator << "Count\n";
    for (int i = 0; i < API_LAST; i++) {
      llvm::outs() << "  " << apiNames[i] << ": " << countApiRepsTotal[i] << "\n";
      csv << apiNames[i] << separator << countApiRepsTotal[i] << "\n";
    }
    llvm::outs() << hipify_info << "CONVERTED refs by names:\n";
    csv << "\nCUDA ref name" << separator << "Count\n";
    for (const auto & it : cuda2hipConvertedTotal) {
      llvm::outs() << "  " << it.first << ": " << it.second << "\n";
      csv << it.first << separator << it.second << "\n";
    }
  }
  if (sum_unsupported > 0) {
    str = "UNCONVERTED refs by type:";
    llvm::outs() << hipify_info << str << "\n";
    csv << "\nUNCONVERTED CUDA ref type" << separator << "Count\n";
    for (int i = 0; i < CONV_LAST; i++) {
      sum_interm = countRepsTotalUnsupported[i];
      if (0 == sum_interm) {
        continue;
      }
      llvm::outs() << "  " << counterNames[i] << ": " << sum_interm << "\n";
      csv << counterNames[i] << separator << sum_interm << "\n";
    }
    llvm::outs() << hipify_info << "UNCONVERTED refs by API:\n";
    csv << "\nUNCONVERTED CUDA API" << separator << "Count\n";
    for (int i = 0; i < API_LAST; i++) {
      llvm::outs() << "  " << apiNames[i] << ": " << countApiRepsTotalUnsupported[i] << "\n";
      csv << apiNames[i] << separator << countApiRepsTotalUnsupported[i] << "\n";
    }
    llvm::outs() << hipify_info << "UNCONVERTED refs by names:\n";
    csv << "\nUNCONVERTED CUDA ref name" << separator << "Count\n";
    for (const auto & it : cuda2hipUnconvertedTotal) {
      llvm::outs() << "  " << it.first << ": " << it.second << "\n";
      csv << it.first << separator << it.second << "\n";
    }
  }
  csv.close();
}

void copyFile(const std::string& src, const std::string& dst) {
  std::ifstream source(src, std::ios::binary);
  std::ofstream dest(dst, std::ios::binary);
  dest << source.rdbuf();
}

int main(int argc, const char **argv) {
  auto start = std::chrono::steady_clock::now();
  auto begin = start;

  // The signature of PrintStackTraceOnErrorSignal changed in llvm 3.9. We don't support
  // anything older than 3.8, so let's specifically detect the one old version we support.
#if (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR == 8)
  llvm::sys::PrintStackTraceOnErrorSignal();
#else
  llvm::sys::PrintStackTraceOnErrorSignal(StringRef());
#endif

  CommonOptionsParser OptionsParser(argc, argv, ToolTemplateCategory, llvm::cl::OneOrMore);
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
  std::string csv;
  if (!OutputStatsFilename.empty()) {
    csv = OutputStatsFilename;
  } else {
    csv = "hipify_stats.csv";
  }
  size_t filesTranslated = fileSources.size();
  uint64_t repBytesTotal = 0;
  uint64_t bytesTotal = 0;
  unsigned changedLinesTotal = 0;
  unsigned linesTotal = 0;
  if (PrintStats && filesTranslated > 1) {
    std::remove(csv.c_str());
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

    // RefactoringTool operates on the file in-place. Giving it the output path is no good,
    // because that'll break relative includes, and we don't want to overwrite the input file.
    // So what we do is operate on a copy, which we then move to the output.
    RefactoringTool Tool(OptionsParser.getCompilations(), tmpFile);
    ast_matchers::MatchFinder Finder;

    // The Replacements to apply to the file `src`.
    Replacements* replacementsToUse;
#if LLVM_VERSION_MAJOR > 3
    // getReplacements() now returns a map from filename to Replacements - so create an entry
    // for this source file and return a pointer to it.
    replacementsToUse = &(Tool.getReplacements()[tmpFile]);
#else
    replacementsToUse = &Tool.getReplacements();
#endif

    HipifyPPCallbacks* PPCallbacks = new HipifyPPCallbacks(replacementsToUse, tmpFile);
    Cuda2HipCallback Callback(replacementsToUse, &Finder, PPCallbacks, tmpFile);

    addAllMatchers(Finder, &Callback);

    auto action = newFrontendActionFactory(&Finder, PPCallbacks);

    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("--cuda-host-only", ArgumentInsertPosition::BEGIN));

    // Ensure at least c++11 is used.
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-std=c++11", ArgumentInsertPosition::BEGIN));
#if defined(HIPIFY_CLANG_RES)
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-resource-dir=" HIPIFY_CLANG_RES));
#endif
    Tool.appendArgumentsAdjuster(getClangSyntaxOnlyAdjuster());
    Result += Tool.run(action.get());
    Tool.clearArgumentsAdjusters();

    LangOptions DefaultLangOptions;
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
    TextDiagnosticPrinter DiagnosticPrinter(llvm::errs(), &*DiagOpts);
    DiagnosticsEngine Diagnostics(IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts, &DiagnosticPrinter, false);

    uint64_t repBytes = 0;
    uint64_t bytes = 0;
    unsigned lines = 0;
    SourceManager SM(Diagnostics, Tool.getFiles());
    if (PrintStats) {
      DEBUG(dbgs() << "Replacements collected by the tool:\n");
#if LLVM_VERSION_MAJOR > 3
        Replacements& replacements = Tool.getReplacements().begin()->second;
#else
        Replacements& replacements = Tool.getReplacements();
#endif
      for (const auto &replacement : replacements) {
        DEBUG(dbgs() << replacement.toString() << "\n");
        repBytes += replacement.getLength();
      }
      std::ifstream src_file(dst, std::ios::binary | std::ios::ate);
      src_file.clear();
      src_file.seekg(0);
      lines = std::count(std::istreambuf_iterator<char>(src_file), std::istreambuf_iterator<char>(), '\n');
      bytes = src_file.tellg();
    }
    Rewriter Rewrite(SM, DefaultLangOptions);
    if (!Tool.applyAllReplacements(Rewrite)) {
      DEBUG(dbgs() << "Skipped some replacements.\n");
    }

    // Either move the tmpfile to the output, or remove it.
    if (!NoOutput) {
      Result += Rewrite.overwriteChangedFiles();
      rename(tmpFile.c_str(), dst.c_str());
    } else {
      remove(tmpFile.c_str());
    }
    if (PrintStats) {
      if (fileSources.size() == 1) {
        if (OutputStatsFilename.empty()) {
          csv = dst + ".csv";
        }
        std::remove(csv.c_str());
      }
      if (0 == printStats(csv, src, *PPCallbacks, Callback, repBytes, bytes, lines, start)) {
        filesTranslated--;
      }
      start = std::chrono::steady_clock::now();
      repBytesTotal += repBytes;
      bytesTotal += bytes;
      changedLinesTotal += PPCallbacks->LOCs.size() + Callback.LOCs.size();
      linesTotal += lines;
    }
    dst.clear();
  }
  if (PrintStats && fileSources.size() > 1) {
    printAllStats(csv, fileSources.size(), filesTranslated, repBytesTotal, bytesTotal, changedLinesTotal, linesTotal, begin);
  }
  return Result;
}

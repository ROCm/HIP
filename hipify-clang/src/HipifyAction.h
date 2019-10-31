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

#include "clang/Lex/PPCallbacks.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "ReplacementsFrontendActionFactory.h"
#include "Statistics.h"

namespace ct = clang::tooling;
namespace mat = clang::ast_matchers;
using namespace llvm;

/**
  * A FrontendAction that hipifies CUDA programs.
  */
class HipifyAction : public clang::ASTFrontendAction,
                     public mat::MatchFinder::MatchCallback {
private:
  ct::Replacements *replacements;
  std::map<std::string, clang::SourceLocation> Ifndefs;
  std::unique_ptr<mat::MatchFinder> Finder;
  // CUDA implicitly adds its runtime header. We rewrite explicitly-provided CUDA includes with equivalent
  // ones, and track - using this flag - if the result led to us including the hip runtime header. If it did
  // not, we insert it at the top of the file when we finish processing it.
  // This approach means we do the best it's possible to do w.r.t preserving the user's include order.
  bool insertedRuntimeHeader = false;
  bool insertedBLASHeader = false;
  bool insertedRANDHeader = false;
  bool insertedRAND_kernelHeader = false;
  bool insertedDNNHeader = false;
  bool insertedFFTHeader = false;
  bool insertedSPARSEHeader = false;
  bool insertedComplexHeader = false;
  bool firstHeader = false;
  bool pragmaOnce = false;
  clang::SourceLocation firstHeaderLoc;
  clang::SourceLocation pragmaOnceLoc;
  // Rewrite a string literal to refer to hip, not CUDA.
  void RewriteString(StringRef s, clang::SourceLocation start);
  // Replace a CUDA identifier with the corresponding hip identifier, if applicable.
  void RewriteToken(const clang::Token &t);
  // Calculate str's SourceLocation in SourceRange sr
  clang::SourceLocation GetSubstrLocation(const std::string &str, const clang::SourceRange &sr);

public:
  explicit HipifyAction(ct::Replacements *replacements): clang::ASTFrontendAction(),
    replacements(replacements) {}
  // MatchCallback listeners
  bool cudaLaunchKernel(const mat::MatchFinder::MatchResult &Result);
  bool cudaSharedIncompleteArrayVar(const mat::MatchFinder::MatchResult &Result);
  bool cudaDeviceFuncCall(const mat::MatchFinder::MatchResult &Result);
  bool cudaHostFuncCall(const mat::MatchFinder::MatchResult &Result);
  bool cubNamespacePrefix(const mat::MatchFinder::MatchResult &Result);
  bool cubFunctionTemplateDecl(const mat::MatchFinder::MatchResult &Result);
  bool cubUsingNamespaceDecl(const mat::MatchFinder::MatchResult &Result);
  // Called by the preprocessor for each include directive during the non-raw lexing pass.
  void InclusionDirective(clang::SourceLocation hash_loc,
                          const clang::Token &include_token,
                          StringRef file_name,
                          bool is_angled,
                          clang::CharSourceRange filename_range,
                          const clang::FileEntry *file,
                          StringRef search_path,
                          StringRef relative_path,
                          const clang::Module *imported);
  // Called by the preprocessor for each pragma directive during the non-raw lexing pass.
  void PragmaDirective(clang::SourceLocation Loc, clang::PragmaIntroducerKind Introducer);
  // Called by the preprocessor for each ifndef directive during the non-raw lexing pass.
  // Found ifndef will be used in EndSourceFileAction() for catching include guard controlling macro.
  void Ifndef(clang::SourceLocation Loc, const clang::Token &MacroNameTok, const clang::MacroDefinition &MD);

protected:
  // Add a Replacement for the current file. These will all be applied after executing the FrontendAction.
  void insertReplacement(const ct::Replacement &rep, const clang::FullSourceLoc &fullSL);
  // FrontendAction entry point.
  void ExecuteAction() override;
  // Callback before starting processing a single input; used by hipify-clang for setting Preprocessor options.
  bool BeginInvocation(clang::CompilerInstance &CI) override;
  // Called at the start of each new file to process.
  void EndSourceFileAction() override;
  // MatchCallback API entry point. Called by the AST visitor while searching the AST for things we registered an interest for.
  void run(const mat::MatchFinder::MatchResult &Result) override;
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &CI, StringRef InFile) override;
  bool Exclude(const hipCounter &hipToken);
  void FindAndReplace(StringRef name, clang::SourceLocation sl, const std::map<StringRef, hipCounter> &repMap, bool bReplace = true);
};

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

#include <clang/Tooling/Core/Replacement.h>
#include <clang/Tooling/Refactoring.h>
#include <llvm/Support/Signals.h>
#include <clang/Lex/Token.h>
#include <clang/Lex/Preprocessor.h>

namespace ct = clang::tooling;

// Things for papering over the differences between different LLVM versions.

namespace llcompat {
/**
  * The getNumArgs function on macros was rather unhelpfully renamed in clang 4.0. Its semantics
  * remain unchanged, so let's be slightly ugly about it here. :D
  */
#if LLVM_VERSION_MAJOR > 4
  #define GET_NUM_ARGS() getNumParams()
#else
  #define GET_NUM_ARGS() getNumArgs()
#endif

#if LLVM_VERSION_MAJOR < 7
  #define LLVM_DEBUG(X) DEBUG(X)
#endif

clang::SourceLocation getBeginLoc(const clang::Stmt* stmt);
clang::SourceLocation getBeginLoc(const clang::TypeLoc& typeLoc);

clang::SourceLocation getEndLoc(const clang::Stmt* stmt);
clang::SourceLocation getEndLoc(const clang::TypeLoc& typeLoc);

void PrintStackTraceOnErrorSignal();

using namespace llvm;

/**
  * Get the replacement map for a given filename in a RefactoringTool.
  *
  * Older LLVM versions don't actually support multiple filenames, so everything all gets
  * smushed together. It is the caller's responsibility to cope with this.
  */
ct::Replacements& getReplacements(ct::RefactoringTool& Tool, StringRef file);

/**
  * Add a Replacement to a Replacements.
  */
void insertReplacement(ct::Replacements& replacements, const ct::Replacement& rep);

/**
  * Version-agnostic version of Preprocessor::EnterTokenStream().
  */
void EnterPreprocessorTokenStream(clang::Preprocessor& _pp,
                                  const clang::Token *start,
                                  size_t len,
                                  bool DisableMacroExpansion);

std::error_code real_path(const Twine &path, SmallVectorImpl<char> &output,
                          bool expand_tilde = false);

} // namespace llcompat

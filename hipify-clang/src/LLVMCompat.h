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

} // namespace llcompat

#include "LLVMCompat.h"

namespace llcompat {

void PrintStackTraceOnErrorSignal() {
  // The signature of PrintStackTraceOnErrorSignal changed in llvm 3.9. We don't support
  // anything older than 3.8, so let's specifically detect the one old version we support.
#if (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR == 8)
  llvm::sys::PrintStackTraceOnErrorSignal();
#else
  llvm::sys::PrintStackTraceOnErrorSignal(clang::StringRef());
#endif
}

ct::Replacements& getReplacements(ct::RefactoringTool& Tool, clang::StringRef file) {
#if LLVM_VERSION_MAJOR > 3
  // getReplacements() now returns a map from filename to Replacements - so create an entry
  // for this source file and return a reference to it.
  return Tool.getReplacements()[file];
#else
  return Tool.getReplacements();
#endif
}

void insertReplacement(ct::Replacements& replacements, const ct::Replacement& rep) {
#if LLVM_VERSION_MAJOR > 3
  // New clang added error checking to Replacements, and *insists* that you explicitly check it.
  llvm::consumeError(replacements.add(rep));
#else
  // In older versions, it's literally an std::set<Replacement>
  replacements.insert(rep);
#endif
}

void EnterPreprocessorTokenStream(clang::Preprocessor& _pp, const clang::Token *start, size_t len, bool DisableMacroExpansion) {
#if (LLVM_VERSION_MAJOR == 3) && (LLVM_VERSION_MINOR == 8)
  _pp.EnterTokenStream(start, len, false, DisableMacroExpansion);
#else
  _pp.EnterTokenStream(clang::ArrayRef<clang::Token>{start, len}, DisableMacroExpansion);
#endif
}

} // namespace llcompat

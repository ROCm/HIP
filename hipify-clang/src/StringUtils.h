#pragma once

#include <string>
#include "llvm/ADT/StringRef.h"

/**
  * Remove double-quotes from the start/end of a string, if present.
  */
llvm::StringRef unquoteStr(llvm::StringRef s);

/**
  * If `s` starts with `prefix`, remove it. Otherwise, does nothing.
  */
void removePrefixIfPresent(std::string &s, std::string prefix);

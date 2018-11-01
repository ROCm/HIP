#include "StringUtils.h"

llvm::StringRef unquoteStr(llvm::StringRef s) {
  if (s.size() > 1 && s.front() == '"' && s.back() == '"') {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

void removePrefixIfPresent(std::string &s, const std::string& prefix) {
  if (s.find(prefix) != 0) {
    return;
  }
  s.erase(0, prefix.size());
}

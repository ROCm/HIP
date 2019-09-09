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

#include <hip/hip_runtime.h>
#include "cl_common.hpp"
#include <hip/hiprtc.h>
#include <boost/core/demangle.hpp>

const char* hiprtcGetErrorString(hiprtcResult x) {
  switch (x) {
    case HIPRTC_SUCCESS:
      return "HIPRTC_SUCCESS";
    case HIPRTC_ERROR_OUT_OF_MEMORY:
      return "HIPRTC_ERROR_OUT_OF_MEMORY";
    case HIPRTC_ERROR_PROGRAM_CREATION_FAILURE:
      return "HIPRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case HIPRTC_ERROR_INVALID_INPUT:
      return "HIPRTC_ERROR_INVALID_INPUT";
    case HIPRTC_ERROR_INVALID_PROGRAM:
      return "HIPRTC_ERROR_INVALID_PROGRAM";
    case HIPRTC_ERROR_INVALID_OPTION:
      return "HIPRTC_ERROR_INVALID_OPTION";
    case HIPRTC_ERROR_COMPILATION:
      return "HIPRTC_ERROR_COMPILATION";
    case HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE:
      return "HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
      return "HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
      return "HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
      return "HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case HIPRTC_ERROR_INTERNAL_ERROR:
      return "HIPRTC_ERROR_INTERNAL_ERROR";
    default:
      throw std::logic_error{"Invalid HIPRTC result."};
  };
}

namespace hip_impl {
inline std::string demangle(const char* x) {
  if (!x) {
    return {};
  }
  return boost::core::demangle(x);
}
}  // Namespace hip_impl

struct _hiprtcProgram {
  static amd::Monitor lock_;
  static std::vector<std::unique_ptr<_hiprtcProgram>> programs_;

  std::vector<std::pair<std::string, std::string>> headers;
  std::vector<std::pair<std::string, std::string>> names;
  std::vector<std::string> loweredNames;
  std::vector<char> elf;
  std::string source;
  std::string name;
  std::string log;
  bool compiled;

  static _hiprtcProgram* build(std::string s, std::string n,
                              std::vector<std::pair<std::string, std::string>> h) {
    std::unique_ptr<_hiprtcProgram> tmp{
        new _hiprtcProgram{std::move(h), {}, {}, {}, std::move(s), std::move(n), {}, false}};

    amd::ScopedLock lock(_hiprtcProgram::lock_);

    programs_.push_back(move(tmp));

    return programs_.back().get();
  }

  static hiprtcResult destroy(_hiprtcProgram* p) {
    amd::ScopedLock lock(_hiprtcProgram::lock_);

    const auto it{ std::find_if(programs_.cbegin(), programs_.cend(),
                     [=](const std::unique_ptr<_hiprtcProgram>& x)
                     { return x.get() == p; }) };

    if (it == programs_.cend()) {
      return HIPRTC_ERROR_INVALID_PROGRAM;
    }

    return HIPRTC_SUCCESS;
  }

  static std::string handleMangledName(std::string name) {
    name = hip_impl::demangle(name.c_str());

    if (name.empty()) {
      return name;
    }

    if (name.find("void ") == 0) {
      name.erase(0, strlen("void "));
    }

    auto dx {name.find_first_of("(<")};

    if (dx == std::string::npos) {
      return name;
    }

    if (name[dx] == '<') {
      auto cnt{1u};
      do {
      ++dx;
      cnt += (name[dx] == '<') ? 1 : ((name[dx] == '>') ? -1 : 0);
      } while (cnt);

      name.erase(++dx);
    } else {
      name.erase(dx);
    }

    return name;
  }

  static bool isValid(_hiprtcProgram* p) {
    return std::find_if(programs_.cbegin(), programs_.cend(),
                        [=](const std::unique_ptr<_hiprtcProgram>& x) {
                            return x.get() == p; }) != programs_.cend();
  }
};

// Init
std::vector<std::unique_ptr<_hiprtcProgram>> _hiprtcProgram::programs_{};
amd::Monitor _hiprtcProgram::lock_("hiprtcProgram lock");

inline bool isValidProgram(const hiprtcProgram p) {
  if (p == nullptr) {
    return false;
  }

  amd::ScopedLock lock(_hiprtcProgram::lock_);

  return _hiprtcProgram::isValid(p);
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram* p, const char* src, const char* name, int n,
                                 const char** hdrs, const char** incs) {
  if (p == nullptr) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }
  if (n < 0) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }
  if (n && (hdrs == nullptr || incs == nullptr)) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  std::vector<std::pair<std::string, std::string>> h;

  for (auto i = 0; i != n; ++i) {
    h.emplace_back(incs[i], hdrs[i]);
  }
  *p = _hiprtcProgram::build(src, name ? name : "default_name", std::move(h));

  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcAddNameExpression(hiprtcProgram p, const char* n) {
  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram p, int n, const char** o) {
  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram* p) {
  if (p == nullptr) {
    return HIPRTC_SUCCESS;
  }
  return _hiprtcProgram::destroy(*p);
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram p, const char* n, const char** loweredNames) {
  if (n == nullptr || loweredNames == nullptr) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  if (!isValidProgram(p)) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  if (!p->compiled) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  const auto it{ std::find_if(p->names.cbegin(), p->names.cend(),
                             [=](const pair<string, string>& x)
                                { return x.first == n; })};

  if (it == p->names.cend()) {
    return HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
  }

  *loweredNames = p->loweredNames[distance(p->names.cbegin(), it)].c_str();

  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram p, char* log) {
  if (log == nullptr) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  if (!isValidProgram(p)) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  if (!p->compiled) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  log = std::copy_n(p->log.data(), p->log.size(), log);
  *log = '\0';

  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram p, std::size_t* sz) {
  if (sz == nullptr) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  if (!isValidProgram(p)) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  if (!p->compiled) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  *sz = p->log.empty() ? 0 : p->log.size() + 1;
  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetCode(hiprtcProgram p, char* c) {
  if (c == nullptr) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  if (!isValidProgram(p)) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  if (!p->compiled) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  std::copy_n(p->elf.data(), p->elf.size(), c);

  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram p, std::size_t* sz) {
  if (sz == nullptr) {
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  if (!isValidProgram(p)) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  if (!p->compiled) {
    return HIPRTC_ERROR_INVALID_PROGRAM;
  }

  *sz = p->elf.size();

  return HIPRTC_SUCCESS;
}
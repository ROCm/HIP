/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <vector>

#include <hip_test_common.hh>

class ModuleGuard {
 public:
  ~ModuleGuard() { static_cast<void>(hipModuleUnload(module_)); }

  ModuleGuard(const ModuleGuard&) = delete;
  ModuleGuard(ModuleGuard&&) = delete;

  static ModuleGuard LoadModule(const char* fname);

  static ModuleGuard LoadModuleDataFile(const char* fname);

  static ModuleGuard LoadModuleDataRTC(const char* code);

  hipModule_t module() const { return module_; }

 private:
  ModuleGuard(const hipModule_t module) : module_{module} {}
  hipModule_t module_ = nullptr;
};

// Load module into buffer instead of mapping file to avoid platform specific mechanisms
std::vector<char> LoadModuleIntoBuffer(const char* path_string);

std::vector<char> CreateRTCCharArray(const char* src);

inline hipFunction_t GetKernel(const hipModule_t module, const char* kname) {
  hipFunction_t kernel = nullptr;
  HIP_CHECK(hipModuleGetFunction(&kernel, module, kname));
  return kernel;
}
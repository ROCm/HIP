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

#include "hip_module_common.hh"

#include <experimental/filesystem>
#include <fstream>

#include <hip_test_common.hh>
#include <hip/hiprtc.h>

ModuleGuard ModuleGuard::LoadModule(const char* fname) {
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, fname));
  return ModuleGuard{module};
}

ModuleGuard ModuleGuard::LoadModuleDataFile(const char* fname) {
  const auto loaded_module = LoadModuleIntoBuffer(fname);
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, loaded_module.data()));
  return ModuleGuard{module};
}

ModuleGuard ModuleGuard::LoadModuleDataRTC(const char* code) {
  const auto rtc = CreateRTCCharArray(code);
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoadData(&module, rtc.data()));
  return ModuleGuard{module};
}

// Load module into buffer instead of mapping file to avoid platform specific mechanisms
std::vector<char> LoadModuleIntoBuffer(const char* path_string) {
  std::experimental::filesystem::path p(path_string);
  const auto file_size = std::experimental::filesystem::file_size(p);
  std::ifstream f(p, std::ios::binary | std::ios::in);
  REQUIRE(f);
  std::vector<char> empty_module(file_size);
  REQUIRE(f.read(empty_module.data(), file_size));
  return empty_module;
}

std::vector<char> CreateRTCCharArray(const char* src) {
  hiprtcProgram prog;
  HIPRTC_CHECK(hiprtcCreateProgram(&prog, src, "prog", 0, nullptr, nullptr));
  HIPRTC_CHECK(hiprtcCompileProgram(prog, 0, nullptr));
  size_t code_size = 0;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
  std::vector<char> code(code_size, '\0');
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));
  return code;
}
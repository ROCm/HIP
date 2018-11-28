/*
Copyright (c) 2018 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include <string>
#include <fstream>

#include "hip_fatbin.h"
#include "hip/hip_runtime.h"
#include "hip_hcc_internal.h"
#include "trace_helper.h"

void __hipDumpCodeObject(const std::string& image) {
    char fname[30];
    static std::atomic<int> index;
    sprintf(fname, "__hip_dump_code_object%04d.o", index++);
    tprintf(DB_FB, "Dump code object %s\n", fname);
    std::ofstream ofs;
    ofs.open(fname, std::ios::binary);
    ofs << image;
    ofs.close();
}

// Returns a pointer to the code object in the fatbin. The pointer should not
// be freed.
const void* __hipExtractCodeObjectFromFatBinary(const void* data,
    const char* agent_name)
{
  HIP_INIT();

  tprintf(DB_FB, "Enter __hipExtractCodeObjectFromFatBinary(%p, \"%s\")\n",
      data, agent_name);

  const __ClangOffloadBundleHeader* header
      = reinterpret_cast<const __ClangOffloadBundleHeader*>(data);
  std::string magic(reinterpret_cast<const char*>(header),
      sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC)) {
    return nullptr;
  }

  const __ClangOffloadBundleDesc* desc = &header->desc[0];
  for (uint64_t i = 0; i < header->numBundles; ++i,
       desc = reinterpret_cast<const __ClangOffloadBundleDesc*>(
           reinterpret_cast<uintptr_t>(&desc->triple[0]) + desc->tripleSize)) {

    std::string triple{&desc->triple[0], sizeof(AMDGCN_AMDHSA_TRIPLE) - 1};
    if (triple.compare(AMDGCN_AMDHSA_TRIPLE))
      continue;

    std::string target{&desc->triple[sizeof(AMDGCN_AMDHSA_TRIPLE)],
      desc->tripleSize - sizeof(AMDGCN_AMDHSA_TRIPLE)};
    tprintf(DB_FB, "Found hip-clang bundle for %s\n", target.c_str());
    if (target.compare(agent_name)) {
       continue;
    }

    auto *codeobj = reinterpret_cast<const char*>(
        reinterpret_cast<uintptr_t>(header) + desc->offset);
    if (HIP_DUMP_CODE_OBJECT)
      __hipDumpCodeObject(std::string{codeobj, desc->size});

    tprintf(DB_FB, "__hipExtractCodeObjectFromFatBinary succeeds and returns %p\n",
        codeobj);
    return codeobj;
  }

  // hipcc --genco for HCC generates fat binaries with different triple strings.
  // It will reach here and return a null pointer. The fat binary itself will
  // be handled in a different place.
  tprintf(DB_FB, "No hip-clang device code bundle for %s\n", agent_name);
  return nullptr;
}


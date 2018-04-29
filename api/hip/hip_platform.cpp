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

#include "hip_internal.hpp"
#include "platform/program.hpp"
#include "platform/runtime.hpp"

#include <unordered_map>
#include "elfio.hpp"

constexpr unsigned __cudaFatMAGIC2 = 0x466243b1;

struct __CudaFatBinaryWrapper {
  unsigned int magic;
  unsigned int version;
  void*        binary;
  void*        dummy1;
};

#define CLANG_OFFLOAD_BUNDLER_MAGIC_STR "__CLANG_OFFLOAD_BUNDLE__"
#define OPENMP_AMDGCN_AMDHSA_TRIPLE "openmp-amdgcn--amdhsa"
#define HCC_AMDGCN_AMDHSA_TRIPLE "hcc-amdgcn--amdhsa"

struct __ClangOffloadBundleDesc {
  uint64_t offset;
  uint64_t size;
  uint64_t tripleSize;
  const char triple[1];
};

struct __ClangOffloadBundleHeader {
  const char magic[sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC_STR) - 1];
  uint64_t numBundles;
  __ClangOffloadBundleDesc desc[1];
};

extern "C" hipModule_t __hipRegisterFatBinary(const void* data)
{
  HIP_INIT();

  const __CudaFatBinaryWrapper* fbwrapper = reinterpret_cast<const __CudaFatBinaryWrapper*>(data);
  if (fbwrapper->magic != __cudaFatMAGIC2 || fbwrapper->version != 1) {
    return nullptr;
  }
  std::string magic((char*)fbwrapper->binary, sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC_STR) - 1);
  if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC_STR)) {
    return nullptr;
  }

  amd::Program* program = new amd::Program(*g_context);
  if (!program)
    return nullptr;

  const auto obheader = reinterpret_cast<const __ClangOffloadBundleHeader*>(fbwrapper->binary);
  const auto* desc = &obheader->desc[0];
  for (uint64_t i = 0; i < obheader->numBundles; ++i,
       desc = reinterpret_cast<const __ClangOffloadBundleDesc*>(
           reinterpret_cast<uintptr_t>(&desc->triple[0]) + desc->tripleSize)) {

    std::string triple(desc->triple, sizeof(OPENMP_AMDGCN_AMDHSA_TRIPLE) - 1);
    if (triple.compare(OPENMP_AMDGCN_AMDHSA_TRIPLE))
      continue;

    std::string target(desc->triple + sizeof(OPENMP_AMDGCN_AMDHSA_TRIPLE),
                       desc->tripleSize - sizeof(OPENMP_AMDGCN_AMDHSA_TRIPLE));
    if (target.compare(g_context->devices()[0]->info().name_))
      continue;

    const void *image = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(obheader) + desc->offset);
    size_t size = desc->size;

    if (CL_SUCCESS == program->addDeviceProgram(*g_context->devices()[0], image, size) &&
        CL_SUCCESS == program->build(g_context->devices(), nullptr, nullptr, nullptr))
      break;
  }

  return reinterpret_cast<hipModule_t>(as_cl(program));
}

std::map<const void*, hipFunction_t> g_functions;


extern "C" void __hipRegisterFunction(
  hipModule_t  module,
  const void*  hostFunction,
  char*        deviceFunction,
  const char*  deviceName,
  unsigned int threadLimit,
  uint3*       tid,
  uint3*       bid,
  dim3*        blockDim,
  dim3*        gridDim,
  int*         wSize)
{
  HIP_INIT();

  amd::Program* program = as_amd(reinterpret_cast<cl_program>(module));

  const amd::Symbol* symbol = program->findSymbol(deviceName);
  if (!symbol) return;

  amd::Kernel* kernel = new amd::Kernel(*program, *symbol, deviceName);
  if (!kernel) return;

  // FIXME: not thread safe
  g_functions.insert(std::make_pair(hostFunction, reinterpret_cast<hipFunction_t>(as_cl(kernel))));
}

extern "C" void __hipRegisterVar(
  hipModule_t module,
  char*       hostVar,
  char*       deviceVar,
  const char* deviceName,
  int         ext,
  int         size,
  int         constant,
  int         global)
{
  HIP_INIT();
}

extern "C" void __hipUnregisterFatBinary(
  hipModule_t module
)
{
  HIP_INIT();
}

dim3 g_gridDim; // FIXME: place in execution stack
dim3 g_blockDim; // FIXME: place in execution stack
size_t g_sharedMem; // FIXME: place in execution stack
hipStream_t g_stream; // FIXME: place in execution stack

extern "C" hipError_t hipConfigureCall(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem,
  hipStream_t stream)
{
  HIP_INIT_API(gridDim, blockDim, sharedMem, stream);

  // FIXME: should push and new entry on the execution stack

  g_gridDim = gridDim;
  g_blockDim = blockDim;
  g_sharedMem = sharedMem;
  g_stream = stream;

  return hipSuccess;
}

char* g_arguments[1024]; // FIXME: needs to grow

extern "C" hipError_t hipSetupArgument(
  const void *arg,
  size_t size,
  size_t offset)
{
  HIP_INIT_API(arg, size, offset);

  // FIXME: should modify the top of the execution stack

  ::memcpy(g_arguments + offset, arg, size);
  return hipSuccess;
}

extern "C" hipError_t hipLaunchByPtr(const void *hostFunction)
{
  HIP_INIT_API(hostFunction);

  const auto it = g_functions.find(hostFunction);
  if (it == g_functions.cend())
    return hipErrorUnknown;

  // FIXME: should pop an entry from the execution stack

  void *extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, g_arguments,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, 0 /* FIXME: not needed, but should be correct*/,
      HIP_LAUNCH_PARAM_END
    };

  return hipModuleLaunchKernel(it->second,
    g_gridDim.x, g_gridDim.y, g_gridDim.z,
    g_blockDim.x, g_blockDim.y, g_blockDim.z,
    g_sharedMem, g_stream, nullptr, extra);
}

#if defined(ATI_OS_LINUX)

namespace hip_impl {

struct dl_phdr_info {
  ELFIO::Elf64_Addr        dlpi_addr;
  const char       *dlpi_name;
  const ELFIO::Elf64_Phdr *dlpi_phdr;
  ELFIO::Elf64_Half        dlpi_phnum;
};

extern "C" int dl_iterate_phdr(
  int (*callback) (struct dl_phdr_info *info, size_t size, void *data), void *data
);

struct Symbol {
  std::string name;
  ELFIO::Elf64_Addr value = 0;
  ELFIO::Elf_Xword size = 0;
  ELFIO::Elf_Half sect_idx = 0;
  uint8_t bind = 0;
  uint8_t type = 0;
  uint8_t other = 0;
};

inline Symbol read_symbol(const ELFIO::symbol_section_accessor& section, unsigned int idx) {
  assert(idx < section.get_symbols_num());

  Symbol r;
  section.get_symbol(idx, r.name, r.value, r.size, r.bind, r.type, r.sect_idx, r.other);

  return r;
}

template <typename P>
inline ELFIO::section* find_section_if(ELFIO::elfio& reader, P p) {
    const auto it = find_if(reader.sections.begin(), reader.sections.end(), std::move(p));

    return it != reader.sections.end() ? *it : nullptr;
}

std::vector<std::pair<uintptr_t, std::string>> function_names_for(const ELFIO::elfio& reader,
                                                                  ELFIO::section* symtab) {
  std::vector<std::pair<uintptr_t, std::string>> r;
  ELFIO::symbol_section_accessor symbols{reader, symtab};

  for (auto i = 0u; i != symbols.get_symbols_num(); ++i) {
    auto tmp = read_symbol(symbols, i);

    if (tmp.type == STT_FUNC && tmp.sect_idx != SHN_UNDEF && !tmp.name.empty()) {
      r.emplace_back(tmp.value, tmp.name);
    }
  }

  return r;
}

const std::vector<std::pair<uintptr_t, std::string>>& function_names_for_process() {
  static constexpr const char self[] = "/proc/self/exe";

  static std::vector<std::pair<uintptr_t, std::string>> r;
  static std::once_flag f;

  std::call_once(f, []() {
    ELFIO::elfio reader;

    if (reader.load(self)) {
      const auto it = find_section_if(
          reader, [](const ELFIO::section* x) { return x->get_type() == SHT_SYMTAB; });

      if (it) r = function_names_for(reader, it);
    }
  });

  return r;
}


const std::unordered_map<uintptr_t, std::string>& function_names()
{
  static std::unordered_map<uintptr_t, std::string> r{
    function_names_for_process().cbegin(),
    function_names_for_process().cend()};
  static std::once_flag f;

  std::call_once(f, []() {
    dl_iterate_phdr([](dl_phdr_info* info, size_t, void*) {
      ELFIO::elfio reader;

      if (reader.load(info->dlpi_name)) {
        const auto it = find_section_if(
            reader, [](const ELFIO::section* x) { return x->get_type() == SHT_SYMTAB; });

        if (it) {
          auto n = function_names_for(reader, it);

          for (auto&& f : n) f.first += info->dlpi_addr;

          r.insert(make_move_iterator(n.begin()), make_move_iterator(n.end()));
        }
      }
      return 0;
    },
    nullptr);
  });

  return r;
}

std::vector<char> bundles_for_process() {
  static constexpr const char self[] = "/proc/self/exe";
  static constexpr const char kernel_section[] = ".kernel";
  std::vector<char> r;

  ELFIO::elfio reader;

  if (reader.load(self)) {
    auto it = find_section_if(
        reader, [](const ELFIO::section* x) { return x->get_name() == kernel_section; });

    if (it) r.insert(r.end(), it->get_data(), it->get_data() + it->get_size());
  }

  return r;
}

const std::vector<hipModule_t>& modules() {
    static std::vector<hipModule_t> r;
    static std::once_flag f;

    std::call_once(f, []() {
      static std::vector<std::vector<char>> bundles{bundles_for_process()};

      dl_iterate_phdr(
          [](dl_phdr_info* info, std::size_t, void*) {
        ELFIO::elfio tmp;
        if (tmp.load(info->dlpi_name)) {
          const auto it = find_section_if(
              tmp, [](const ELFIO::section* x) { return x->get_name() == ".kernel"; });

          if (it) bundles.emplace_back(it->get_data(), it->get_data() + it->get_size());
        }
        return 0;
      },
      nullptr);

      for (auto&& bundle : bundles) {
        std::string magic(&bundle[0], sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC_STR) - 1);
        if (magic.compare(CLANG_OFFLOAD_BUNDLER_MAGIC_STR))
          continue;

        const auto obheader = reinterpret_cast<const __ClangOffloadBundleHeader*>(&bundle[0]);
        const auto* desc = &obheader->desc[0];
        for (uint64_t i = 0; i < obheader->numBundles; ++i,
             desc = reinterpret_cast<const __ClangOffloadBundleDesc*>(
                 reinterpret_cast<uintptr_t>(&desc->triple[0]) + desc->tripleSize)) {

          std::string triple(desc->triple, sizeof(HCC_AMDGCN_AMDHSA_TRIPLE) - 1);
          if (triple.compare(HCC_AMDGCN_AMDHSA_TRIPLE))
            continue;

          std::string target(desc->triple + sizeof(HCC_AMDGCN_AMDHSA_TRIPLE),
                             desc->tripleSize - sizeof(HCC_AMDGCN_AMDHSA_TRIPLE));

          if (!target.compare(g_context->devices()[0]->info().name_)) {
            hipModule_t module;
            if (hipSuccess == hipModuleLoadData(&module, reinterpret_cast<const void*>(
                reinterpret_cast<uintptr_t>(obheader) + desc->offset)))
              r.push_back(module);
              break;
          }
        }
      }
    });

    return r;
}

const std::unordered_map<uintptr_t, hipFunction_t>& functions()
{
  static std::unordered_map<uintptr_t, hipFunction_t> r;
  static std::once_flag f;

  std::call_once(f, []() {
    for (auto&& function : function_names()) {
      for (auto&& module : modules()) {
        hipFunction_t f;
        if (hipSuccess == hipModuleGetFunction(&f, module, function.second.c_str()))
          r[function.first] = f;
      }
    }
  });

  return r;
}


void hipLaunchKernelGGLImpl(
  uintptr_t function_address,
  const dim3& numBlocks,
  const dim3& dimBlocks,
  uint32_t sharedMemBytes,
  hipStream_t stream,
  void** kernarg)
{
  HIP_INIT();

  const auto it = functions().find(function_address);
  if (it == functions().cend())
    assert(0);

  hipModuleLaunchKernel(it->second,
    numBlocks.x, numBlocks.y, numBlocks.z,
    dimBlocks.x, dimBlocks.y, dimBlocks.z,
    sharedMemBytes, stream, nullptr, kernarg);
}

}

#endif // defined(ATI_OS_LINUX)

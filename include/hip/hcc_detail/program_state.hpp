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

#include "code_object_bundle.hpp"
#include "hsa_helpers.hpp"

#if !defined(__cpp_exceptions)
    #define try if (true)
    #define catch(...) if (false)
#endif
#include "elfio/elfio.hpp"
#if !defined(__cpp_exceptions)
    #undef try
    #undef catch
#endif

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <link.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ihipModuleSymbol_t;
using hipFunction_t = ihipModuleSymbol_t*;

namespace std {
template<>
struct hash<hsa_agent_t> {
    size_t operator()(hsa_agent_t x) const {
        return hash<decltype(x.handle)>{}(x.handle);
    }
};

template<>
struct hash<hsa_isa_t> {
    size_t operator()(hsa_isa_t x) const {
        return hash<decltype(x.handle)>{}(x.handle);
    }
};
}  // namespace std

inline constexpr bool operator==(hsa_agent_t x, hsa_agent_t y) {
    return x.handle == y.handle;
}
inline constexpr bool operator==(hsa_isa_t x, hsa_isa_t y) {
    return x.handle == y.handle;
}

namespace hip_impl {

[[noreturn]]
void hip_throw(const std::exception&);

class Kernel_descriptor {
    std::uint64_t kernel_object_{};
    amd_kernel_code_t const* kernel_header_{nullptr};
    std::string name_{};
public:
    Kernel_descriptor() = default;
    Kernel_descriptor(std::uint64_t kernel_object, const std::string& name)
        : kernel_object_{kernel_object}, name_{name}
    {
        bool supported{false};
        std::uint16_t min_v{UINT16_MAX};
        auto r = hsa_system_major_extension_supported(
            HSA_EXTENSION_AMD_LOADER, 1, &min_v, &supported);

        if (r != HSA_STATUS_SUCCESS || !supported) return;

        hsa_ven_amd_loader_1_01_pfn_t tbl{};

        r = hsa_system_get_major_extension_table(
            HSA_EXTENSION_AMD_LOADER,
            1,
            sizeof(tbl),
            reinterpret_cast<void*>(&tbl));

        if (r != HSA_STATUS_SUCCESS) return;
        if (!tbl.hsa_ven_amd_loader_query_host_address) return;

        r = tbl.hsa_ven_amd_loader_query_host_address(
            reinterpret_cast<void*>(kernel_object_),
            reinterpret_cast<const void**>(&kernel_header_));

        if (r != HSA_STATUS_SUCCESS) return;
    }
    Kernel_descriptor(const Kernel_descriptor&) = default;
    Kernel_descriptor(Kernel_descriptor&&) = default;
    ~Kernel_descriptor() = default;

    Kernel_descriptor& operator=(const Kernel_descriptor&) = default;
    Kernel_descriptor& operator=(Kernel_descriptor&&) = default;

    operator hipFunction_t() const {  // TODO: this is awful and only meant for illustration.
        return reinterpret_cast<hipFunction_t>(const_cast<Kernel_descriptor*>(this));
    }
};


class kernargs_size_align;
class program_state_impl;
class program_state {
public:
    program_state();
    ~program_state();

    const Kernel_descriptor& kernel_descriptor(std::uintptr_t,
                                               hsa_agent_t);
    
    kernargs_size_align get_kernargs_size_align(std::uintptr_t);
    hsa_executable_t load_executable(const char*,
                                     hsa_executable_t,
                                     hsa_agent_t);

    void* global_addr_by_name(const char* name);

    // to fix later
    const std::unordered_map<
         hsa_agent_t, std::vector<hsa_executable_t>>& executables();


    program_state_impl& impl;
};

class kernargs_size_align {
public:
    std::size_t size(std::size_t n) const;
    std::size_t alignment(std::size_t n) const;
private:
    const void* handle;
    friend kernargs_size_align program_state::get_kernargs_size_align(std::uintptr_t);
};

inline
__attribute__((visibility("hidden")))
program_state& get_program_state() {
    static program_state ps;
    return ps;
}
}  // Namespace hip_impl.

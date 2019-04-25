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

#include <hsa/hsa.h>

#include <cstdint>
#include <functional>
#include <string>

namespace hip_impl {
inline void* address(hsa_executable_symbol_t x) {
    void* r = nullptr;
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &r);

    return r;
}

inline hsa_agent_t agent(hsa_executable_symbol_t x) {
    hsa_agent_t r = {};
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &r);

    return r;
}

inline std::uint32_t group_size(hsa_executable_symbol_t x) {
    std::uint32_t r = 0u;
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &r);

    return r;
}

inline hsa_isa_t isa(hsa_agent_t x) {
    hsa_isa_t r = {};
    hsa_agent_iterate_isas(x,
                           [](hsa_isa_t i, void* o) {
                               *static_cast<hsa_isa_t*>(o) = i;  // Pick the first.

                               return HSA_STATUS_INFO_BREAK;
                           },
                           &r);

    return r;
}

inline std::uint64_t kernel_object(hsa_executable_symbol_t x) {
    std::uint64_t r = 0u;
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &r);

    return r;
}

inline std::string name(hsa_executable_symbol_t x) {
    std::uint32_t sz = 0u;
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &sz);

    std::string r(sz, '\0');
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &r.front());

    return r;
}

inline std::uint32_t private_size(hsa_executable_symbol_t x) {
    std::uint32_t r = 0u;
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &r);

    return r;
}

inline std::uint32_t size(hsa_executable_symbol_t x) {
    std::uint32_t r = 0;
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &r);

    return r;
}

inline hsa_symbol_kind_t type(hsa_executable_symbol_t x) {
    hsa_symbol_kind_t r = {};
    hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &r);

    return r;
}

inline std::string transmogrify_triple(const std::string& triple) {
    static constexpr const char old_prefix[]{"hcc-amdgcn--amdhsa-gfx"};
    static constexpr const char new_prefix[]{"hcc-amdgcn-amd-amdhsa--gfx"};

    if (triple.find(old_prefix) == 0) {
        return new_prefix + triple.substr(sizeof(old_prefix) - 1);
    }

    return (triple.find(new_prefix) == 0) ? triple : "";
}

inline std::string isa_name(std::string triple) {
    static constexpr const char offload_prefix[]{"hcc-"};

    triple = transmogrify_triple(triple);
    if (triple.empty()) return {};

    triple.erase(0, sizeof(offload_prefix) - 1);

    return triple;
}

inline hsa_isa_t triple_to_hsa_isa(const std::string& triple) {
    const std::string isa{isa_name(std::move(triple))};
    if (isa.empty()) return hsa_isa_t({});
    hsa_isa_t r{};
    if(HSA_STATUS_SUCCESS != hsa_isa_from_name(isa.c_str(), &r)) {
        r.handle = 0;
    }
    return r;
}

}  // namespace hip_impl

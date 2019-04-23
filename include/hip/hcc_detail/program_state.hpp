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
#include <amd_comgr.h>

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
#include <iostream>
#include <fstream>

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

std::vector<hsa_agent_t> all_hsa_agents();

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

template<typename P>
inline
ELFIO::section* find_section_if(ELFIO::elfio& reader, P p) {
    const auto it = std::find_if(
        reader.sections.begin(), reader.sections.end(), std::move(p));

    return it != reader.sections.end() ? *it : nullptr;
}

inline
__attribute__((visibility("hidden")))
const std::unordered_map<
    hsa_isa_t, std::vector<std::vector<char>>>& code_object_blobs() {
    static std::unordered_map<hsa_isa_t, std::vector<std::vector<char>>> r;
    static std::once_flag f;

    std::call_once(f, []() {
        static std::vector<std::vector<char>> blobs{};

        dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void*) {
            ELFIO::elfio tmp;

            const auto elf =
                info->dlpi_addr ? info->dlpi_name : "/proc/self/exe";

            if (!tmp.load(elf)) return 0;

            const auto it = find_section_if(tmp, [](const ELFIO::section* x) {
                return x->get_name() == ".kernel";
            });

            if (!it) return 0;

            blobs.emplace_back(it->get_data(), it->get_data() + it->get_size());

            return 0;
        }, nullptr);

        for (auto&& multi_arch_blob : blobs) {
            auto it = multi_arch_blob.begin();
            while (it != multi_arch_blob.end()) {
                Bundled_code_header tmp{it, multi_arch_blob.end()};

                if (!valid(tmp)) break;

                for (auto&& bundle : bundles(tmp)) {
                    r[triple_to_hsa_isa(bundle.triple)].push_back(bundle.blob);
                }

                it += tmp.bundled_code_size;
            };
        }
    });

    return r;
}

struct Symbol {
    std::string name;
    ELFIO::Elf64_Addr value = 0;
    ELFIO::Elf_Xword size = 0;
    ELFIO::Elf_Half sect_idx = 0;
    std::uint8_t bind = 0;
    std::uint8_t type = 0;
    std::uint8_t other = 0;
};

inline
Symbol read_symbol(const ELFIO::symbol_section_accessor& section,
                   unsigned int idx) {
    assert(idx < section.get_symbols_num());

    Symbol r;
    section.get_symbol(
        idx, r.name, r.value, r.size, r.bind, r.type, r.sect_idx, r.other);

    return r;
}

inline
__attribute__((visibility("hidden")))
const std::unordered_map<
    std::string,
    std::pair<ELFIO::Elf64_Addr, ELFIO::Elf_Xword>>& symbol_addresses() {
    static std::unordered_map<
        std::string, std::pair<ELFIO::Elf64_Addr, ELFIO::Elf_Xword>> r;
    static std::once_flag f;

    std::call_once(f, []() {
        dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void*) {
            ELFIO::elfio tmp;
            const auto elf =
                info->dlpi_addr ? info->dlpi_name : "/proc/self/exe";

            if (!tmp.load(elf)) return 0;

            auto it = find_section_if(tmp, [](const ELFIO::section* x) {
                return x->get_type() == SHT_SYMTAB;
            });

            if (!it) return 0;

            const ELFIO::symbol_section_accessor symtab{tmp, it};

            for (auto i = 0u; i != symtab.get_symbols_num(); ++i) {
                auto s = read_symbol(symtab, i);

                if (s.type != STT_OBJECT || s.sect_idx == SHN_UNDEF) continue;

                const auto addr = s.value + info->dlpi_addr;
                r.emplace(std::move(s.name), std::make_pair(addr, s.size));
            }

            return 0;
        }, nullptr);
    });

    return r;
}

inline
__attribute__((visibility("hidden")))
std::unordered_map<std::string, void*>& globals() {
    static std::unordered_map<std::string, void*> r;
    static std::once_flag f;

    std::call_once(f, []() { r.reserve(symbol_addresses().size()); });

    return r;
}

inline
std::vector<std::string> copy_names_of_undefined_symbols(
    const ELFIO::symbol_section_accessor& section) {
    std::vector<std::string> r;

    for (auto i = 0u; i != section.get_symbols_num(); ++i) {
        // TODO: this is boyscout code, caching the temporaries
        //       may be of worth.
        auto tmp = read_symbol(section, i);
        if (tmp.sect_idx != SHN_UNDEF || tmp.name.empty()) continue;

        r.push_back(std::move(tmp.name));
    }

    return r;
}

[[noreturn]]
void hip_throw(const std::exception&);

inline
void associate_code_object_symbols_with_host_allocation(
    const ELFIO::elfio& reader,
    ELFIO::section* code_object_dynsym,
    hsa_agent_t agent,
    hsa_executable_t executable) {
    if (!code_object_dynsym) return;

    const auto undefined_symbols = copy_names_of_undefined_symbols(
        ELFIO::symbol_section_accessor{reader, code_object_dynsym});

    for (auto&& x : undefined_symbols) {
        if (globals().find(x) != globals().cend()) return;

        const auto it1 = symbol_addresses().find(x);

        if (it1 == symbol_addresses().cend()) {
            hip_throw(std::runtime_error{
                "Global symbol: " + x + " is undefined."});
        }

        static std::mutex mtx;
        std::lock_guard<std::mutex> lck{mtx};

        if (globals().find(x) != globals().cend()) return;

        globals().emplace(x, (void*)(it1->second.first));
        void* p = nullptr;
        hsa_amd_memory_lock(
            reinterpret_cast<void*>(it1->second.first),
            it1->second.second,
            nullptr,  // All agents.
            0,
            &p);

        hsa_executable_agent_global_variable_define(
            executable, agent, x.c_str(), p);
    }
}

inline
void load_code_object_and_freeze_executable(
    const std::string& file, hsa_agent_t agent, hsa_executable_t executable) {
    // TODO: the following sequence is inefficient, should be refactored
    //       into a single load of the file and subsequent ELFIO
    //       processing.
    static const auto cor_deleter = [](hsa_code_object_reader_t* p) {
        if (!p) return;

        hsa_code_object_reader_destroy(*p);
        delete p;
    };

    using RAII_code_reader =
        std::unique_ptr<hsa_code_object_reader_t, decltype(cor_deleter)>;

    if (file.empty()) return;

    RAII_code_reader tmp{new hsa_code_object_reader_t, cor_deleter};
    hsa_code_object_reader_create_from_memory(
        file.data(), file.size(), tmp.get());

    hsa_executable_load_agent_code_object(
        executable, agent, *tmp, nullptr, nullptr);

    hsa_executable_freeze(executable, nullptr);

    static std::vector<RAII_code_reader> code_readers;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lck{mtx};
    code_readers.push_back(move(tmp));
}

inline
hsa_executable_t load_executable(const std::string& file,
                                 hsa_executable_t executable,
                                 hsa_agent_t agent) {
    ELFIO::elfio reader;
    std::stringstream tmp{file};

    if (!reader.load(tmp)) return hsa_executable_t{};

    const auto code_object_dynsym = find_section_if(
        reader, [](const ELFIO::section* x) {
            return x->get_type() == SHT_DYNSYM;
    });

    associate_code_object_symbols_with_host_allocation(reader,
                                                       code_object_dynsym,
                                                       agent, executable);

    load_code_object_and_freeze_executable(file, agent, executable);

    return executable;
}

inline
__attribute__((visibility("hidden")))
const std::vector<hsa_executable_t>& executables(hsa_agent_t agent) {
    static std::unordered_map<hsa_agent_t, std::pair<std::once_flag,
        std::vector<hsa_executable_t>>> r;
    static std::once_flag f;

    // Create placeholder for each agent in the map.
    std::call_once(f, []() {
        for (auto&& x : hip_impl::all_hsa_agents()) {
            (void)r[x];
        }
    });

    if (r.find(agent) == r.cend()) {
        hip_throw(std::runtime_error{"invalid agent"});
    }

    std::call_once(r[agent].first, [](hsa_agent_t aa) {
        hsa_agent_iterate_isas(aa, [](hsa_isa_t x, void* pa) {
            const auto it = code_object_blobs().find(x);

            if (it == code_object_blobs().cend()) return HSA_STATUS_SUCCESS;

            hsa_agent_t a = *static_cast<hsa_agent_t*>(pa);

            for (auto&& blob : it->second) {
                hsa_executable_t tmp = {};

                hsa_executable_create_alt(
                    HSA_PROFILE_FULL,
                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                    nullptr,
                    &tmp);

                // TODO: this is massively inefficient and only meant for
                // illustration.
                std::string blob_to_str{blob.cbegin(), blob.cend()};
                tmp = load_executable(blob_to_str, tmp, a);

                if (tmp.handle) r[a].second.push_back(tmp);
            }

            return HSA_STATUS_SUCCESS;
        }, &aa);
    }, agent);

    return r[agent].second;
}

inline
std::vector<std::pair<std::uintptr_t, std::string>> function_names_for(
    const ELFIO::elfio& reader, ELFIO::section* symtab) {
    std::vector<std::pair<std::uintptr_t, std::string>> r;
    ELFIO::symbol_section_accessor symbols{reader, symtab};

    for (auto i = 0u; i != symbols.get_symbols_num(); ++i) {
        // TODO: this is boyscout code, caching the temporaries
        //       may be of worth.
        auto tmp = read_symbol(symbols, i);

        if (tmp.type != STT_FUNC) continue;
        if (tmp.type == SHN_UNDEF) continue;
        if (tmp.name.empty()) continue;

        r.emplace_back(tmp.value, tmp.name);
    }

    return r;
}

inline
__attribute__((visibility("hidden")))
const std::unordered_map<std::uintptr_t, std::string>& function_names() {
    static std::unordered_map<std::uintptr_t, std::string> r;
    static std::once_flag f;

    std::call_once(f, []() {
        dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void*) {
            ELFIO::elfio tmp;
            const auto elf =
                info->dlpi_addr ? info->dlpi_name : "/proc/self/exe";

            if (!tmp.load(elf)) return 0;

            const auto it = find_section_if(tmp, [](const ELFIO::section* x) {
                return x->get_type() == SHT_SYMTAB;
            });

            if (!it) return 0;

            auto names = function_names_for(tmp, it);
            for (auto&& x : names) x.first += info->dlpi_addr;

            r.insert(
                std::make_move_iterator(names.begin()),
                std::make_move_iterator(names.end()));

            return 0;
        }, nullptr);
    });

    return r;
}

inline
__attribute__((visibility("hidden")))
const std::unordered_map<
    std::string, std::vector<hsa_executable_symbol_t>>& kernels(hsa_agent_t agent) {
    static std::unordered_map<hsa_agent_t, std::pair<std::once_flag,
        std::unordered_map<std::string, std::vector<hsa_executable_symbol_t>>>> r;
    static std::once_flag f;

    // Create placeholder for each agent in the map.
    std::call_once(f, []() {
        for (auto&& x : hip_impl::all_hsa_agents()) {
            (void)r[x];
        }
    });

    if (r.find(agent) == r.cend()) {
        hip_throw(std::runtime_error{"invalid agent"});
    }

    std::call_once(r[agent].first, [](hsa_agent_t aa) {
        static const auto copy_kernels = [](
            hsa_executable_t, hsa_agent_t a, hsa_executable_symbol_t x, void*) {
            if (type(x) == HSA_SYMBOL_KIND_KERNEL) r[a].second[name(x)].push_back(x);

            return HSA_STATUS_SUCCESS;
        };

        for (auto&& executable : executables(aa)) {
            hsa_executable_iterate_agent_symbols(
                executable, aa, copy_kernels, nullptr);
        }
    }, agent);

    return r[agent].second;
}

inline
__attribute__((visibility("hidden")))
const std::unordered_map<
    std::uintptr_t,
    Kernel_descriptor>& functions(hsa_agent_t agent) {
    static std::unordered_map<hsa_agent_t, std::pair<std::once_flag,
        std::unordered_map<std::uintptr_t, Kernel_descriptor>>> r;
    static std::once_flag f;

    // Create placeholder for each agent in the map.
    std::call_once(f, []() {
        for (auto&& x : hip_impl::all_hsa_agents()) {
            (void)r[x];
        }
    });

    if (r.find(agent) == r.cend()) {
        hip_throw(std::runtime_error{"invalid agent"});
    }

    std::call_once(r[agent].first, [](hsa_agent_t aa) {
        for (auto&& function : function_names()) {
            const auto it = kernels(aa).find(function.second);

            if (it == kernels(aa).cend()) continue;

            for (auto&& kernel_symbol : it->second) {
                r[aa].second.emplace(
                    function.first,
                    Kernel_descriptor{kernel_object(kernel_symbol), it->first});
            }
        }
    }, agent);

    return r[agent].second;
}

inline
void checkError(
    amd_comgr_status_t status,
    char const *str) {
    if (status != AMD_COMGR_STATUS_SUCCESS) {
        const char *status_str;
        status = amd_comgr_status_string(status, &status_str);
        if (status == AMD_COMGR_STATUS_SUCCESS)
            std::cerr << "FAILED: " << str << "\n  REASON: " <<  status_str << std::endl;
        hip_throw(std::runtime_error{"Metadata parsing failed."});
    }
}

class comgr_metadata_node {
 public:
    amd_comgr_metadata_node_t node;
    bool active;
    comgr_metadata_node() : active(false) {}
    ~comgr_metadata_node() {
        if(active)
            checkError(amd_comgr_destroy_metadata(node), "amd_comgr_destroy_metadata");
    }
    bool is_active() { return active; }
    void set_active(bool value) { active = value; }
    comgr_metadata_node(const comgr_metadata_node&) = delete;
    comgr_metadata_node(comgr_metadata_node&&) = delete;
    comgr_metadata_node& operator=(const comgr_metadata_node&) = delete;
    comgr_metadata_node& operator=(comgr_metadata_node&&) = delete;

};

class comgr_data {
 public:
    amd_comgr_data_t data;
    bool active;
    comgr_data() : active(false) {}
    ~comgr_data() {
        if(active)
            checkError(amd_comgr_release_data(data), "amd_comgr_release_data");
    }
    bool is_active() { return active; }
    void set_active(bool value) { active = value; }
    comgr_data(const comgr_data&) = delete;
    comgr_data(comgr_data&&) = delete;
    comgr_data& operator=(const comgr_data&) = delete;
    comgr_data& operator=(comgr_data&&) = delete;

};

inline
std::string lookup_keyword_value(
    amd_comgr_metadata_node_t& in_node,
    std::string keyword) {
    amd_comgr_status_t status;
    size_t value_size;
    comgr_metadata_node value_meta;

    status = amd_comgr_metadata_lookup(in_node, keyword.c_str(), &value_meta.node);
    checkError(status, "amd_comgr_metadata_lookup");
    value_meta.set_active(true);
    status = amd_comgr_get_metadata_string(value_meta.node, &value_size, NULL);
    checkError(status, "amd_comgr_get_metadata_string");
    // Since value_size returns size with null terminator, we don't include for C++ string size
    value_size--;
    std::string value(value_size, '\0');
    status = amd_comgr_get_metadata_string(value_meta.node, &value_size, &value[0]);
    checkError(status, "amd_comgr_get_metadata_string");

    return value;
}

inline
void process_kernarg_metadata(
    amd_comgr_metadata_node_t blob_meta,
    std::unordered_map<
        std::string,
        std::vector<std::pair<std::size_t, std::size_t>>>& kernargs) {
    amd_comgr_status_t status;
    amd_comgr_metadata_kind_t mkindLookup;
    comgr_metadata_node kernelList;
    std::string kernel_name;
    size_t num_kernels = 0;
    size_t num_kern_args = 0;

    // Kernels is a list of MAPS!!
    status = amd_comgr_metadata_lookup(blob_meta, "Kernels", &kernelList.node);
    // FIXME - few hip memset kernels are missing Kernels node
    if(status != AMD_COMGR_STATUS_SUCCESS)
        return;
    kernelList.set_active(true);

    status = amd_comgr_get_metadata_kind(kernelList.node, &mkindLookup);
    if (mkindLookup != AMD_COMGR_METADATA_KIND_LIST) {
        hip_throw(std::runtime_error{"Lookup of Kernels didn't return a list\n"});
    }

    status = amd_comgr_get_metadata_list_size(kernelList.node, &num_kernels);
    checkError(status, "amd_comgr_get_metadata_list_size");
    for (int i = 0; i < num_kernels; i++) {
        comgr_metadata_node kernelMap;
        status = amd_comgr_index_list_metadata(kernelList.node, i, &kernelMap.node);
        checkError(status, "amd_comgr_index_list_metadata");
        kernelMap.set_active(true);

        kernel_name = std::move(lookup_keyword_value(kernelMap.node, "Name"));

        // Check if this kernel was already processed
        if(!kernargs[kernel_name].empty()) {
            continue;
        }

        comgr_metadata_node kernArgList;
        status = amd_comgr_metadata_lookup(kernelMap.node, "Args", &kernArgList.node);
        if (status == AMD_COMGR_STATUS_SUCCESS ) {
            kernArgList.set_active(true);
            status = amd_comgr_get_metadata_kind(kernArgList.node, &mkindLookup);
            if (mkindLookup != AMD_COMGR_METADATA_KIND_LIST) {
                 hip_throw(std::runtime_error{"Lookup of Args didn't return a list\n"});
            }

            if (status == AMD_COMGR_STATUS_SUCCESS ) {
                status = amd_comgr_get_metadata_list_size(kernArgList.node, &num_kern_args);
                checkError(status, "amd_comgr_get_metadata_list_size");
                for (int k_ar = 0; k_ar < num_kern_args; k_ar++) {
                    comgr_metadata_node kernArgMap;
                    status = amd_comgr_index_list_metadata(kernArgList.node, k_ar, &kernArgMap.node);
                    checkError(status, "amd_comgr_index_list_metadata");
                    kernArgMap.set_active(true);
                    size_t k_arg_size, k_arg_align;

                    k_arg_size = std::stoul(lookup_keyword_value(kernArgMap.node, "Size"));
                    k_arg_align = std::stoul(lookup_keyword_value(kernArgMap.node, "Align"));

                    // Save it into our kernargs
                    kernargs[kernel_name].emplace_back(k_arg_size, k_arg_align);

                } // end kernArgMap
            }
        } // end kernArgList
    } // end kernelMap
} // end kernelList

inline
void read_kernarg_metadata(
    std::string blob,
    std::unordered_map<
        std::string,
        std::vector<std::pair<std::size_t, std::size_t>>>& kernargs) {

    const char *blob_buf = blob.data();
    long blob_size = blob.size();

    amd_comgr_status_t status;
    comgr_data blob_data;
    status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_RELOCATABLE, &blob_data.data);
    checkError(status, "amd_comgr_create_data");
    blob_data.set_active(true);

    status = amd_comgr_set_data(blob_data.data, blob_size, blob_buf);
    if(status != AMD_COMGR_STATUS_SUCCESS)
        return;

    // We have a valid code object now
    status = amd_comgr_set_data_name(blob_data.data, "HIP Code Object");
    checkError(status, "amd_comgr_set_data_name");

    comgr_metadata_node blob_meta;
    status = amd_comgr_get_data_metadata(blob_data.data, &blob_meta.node);
    checkError(status, "amd_comgr_get_data_metadata");
    blob_meta.set_active(true);

    // Root is a map
    amd_comgr_metadata_kind_t blob_mkind;
    status = amd_comgr_get_metadata_kind(blob_meta.node, &blob_mkind);
    checkError(status, "amd_comgr_get_metadata_kind");
    if (blob_mkind != AMD_COMGR_METADATA_KIND_MAP) {
        hip_throw(std::runtime_error{"Root is not map\n"});
    }

    process_kernarg_metadata(blob_meta.node, kernargs);
}

inline
__attribute__((visibility("hidden")))
const std::unordered_map<
    std::string, std::vector<std::pair<std::size_t, std::size_t>>>& kernargs() {
    static std::unordered_map<
        std::string, std::vector<std::pair<std::size_t, std::size_t>>> r;
    static std::once_flag f;

    std::call_once(f, []() {
        for (auto&& isa_blobs : code_object_blobs()) {
            for (auto&& blob : isa_blobs.second) {
                read_kernarg_metadata(std::string{blob.cbegin(), blob.cend()}, r);
            }
        }
    });

    return r;
}
}  // Namespace hip_impl.

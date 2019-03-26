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

std::vector<hsa_agent_t> all_hsa_agents();

inline
__attribute__((visibility("hidden")))
const std::unordered_map<
    hsa_agent_t, std::vector<hsa_executable_t>>& executables() {
    static std::unordered_map<hsa_agent_t, std::vector<hsa_executable_t>> r;
    static std::once_flag f;

    std::call_once(f, []() {
        for (auto&& agent : hip_impl::all_hsa_agents()) {
            hsa_agent_iterate_isas(agent, [](hsa_isa_t x, void* pa) {
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

                    if (tmp.handle) r[a].push_back(tmp);
                }

                return HSA_STATUS_SUCCESS;
            }, &agent);
        }
    });

    return r;
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
    std::string, std::vector<hsa_executable_symbol_t>>& kernels() {
    static std::unordered_map<
        std::string, std::vector<hsa_executable_symbol_t>> r;
    static std::once_flag f;

    std::call_once(f, []() {
        static const auto copy_kernels = [](
            hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t x, void*) {
            if (type(x) == HSA_SYMBOL_KIND_KERNEL) r[name(x)].push_back(x);

            return HSA_STATUS_SUCCESS;
        };

        for (auto&& agent_executables : executables()) {
            for (auto&& executable : agent_executables.second) {
                hsa_executable_iterate_agent_symbols(
                    executable, agent_executables.first, copy_kernels, nullptr);
            }
        }
    });

    return r;
}

inline
__attribute__((visibility("hidden")))
const std::unordered_map<
    std::uintptr_t,
    std::vector<std::pair<hsa_agent_t, Kernel_descriptor>>>& functions() {
    static std::unordered_map<
        std::uintptr_t,
        std::vector<std::pair<hsa_agent_t, Kernel_descriptor>>> r;
    static std::once_flag f;

    std::call_once(f, []() {
        for (auto&& function : function_names()) {
            const auto it = kernels().find(function.second);

            if (it == kernels().cend()) continue;

            for (auto&& kernel_symbol : it->second) {
                r[function.first].emplace_back(
                    agent(kernel_symbol),
                    Kernel_descriptor{kernel_object(kernel_symbol), it->first});
            }
        }
    });

    return r;
}

inline
std::size_t parse_args(
    const std::string& metadata,
    std::size_t f,
    std::size_t l,
    std::vector<std::pair<std::size_t, std::size_t>>& size_align) {
    if (f == l) return f;
    if (!size_align.empty()) return l;

    do {
        static constexpr size_t size_sz{5};
        f = metadata.find("Size:", f) + size_sz;

        if (l <= f) return f;

        auto size = std::strtoul(&metadata[f], nullptr, 10);

        static constexpr size_t align_sz{6};
        f = metadata.find("Align:", f) + align_sz;

        char* l{};
        auto align = std::strtoul(&metadata[f], &l, 10);

        f += (l - &metadata[f]) + 1;

        size_align.emplace_back(size, align);
    } while (true);
}

inline
void read_kernarg_metadata(
    ELFIO::elfio& reader,
    std::unordered_map<
        std::string,
        std::vector<std::pair<std::size_t, std::size_t>>>& kernargs) {
    // TODO: this is inefficient.
    auto it = find_section_if(reader, [](const ELFIO::section* x) {
        return x->get_type() == SHT_NOTE;
    });

    if (!it) return;

    const ELFIO::note_section_accessor acc{reader, it};
    for (decltype(acc.get_notes_num()) i = 0; i != acc.get_notes_num(); ++i) {
        ELFIO::Elf_Word type{};
        std::string name{};
        void* desc{};
        ELFIO::Elf_Word desc_size{};

        acc.get_note(i, type, name, desc, desc_size);

        if (name != "AMD") continue; // TODO: switch to using NT_AMD_AMDGPU_HSA_METADATA.

        std::string tmp{
            static_cast<char*>(desc), static_cast<char*>(desc) + desc_size};

        auto dx = tmp.find("Kernels:");

        if (dx == std::string::npos) continue;

        static constexpr decltype(tmp.size()) kernels_sz{8};
        dx += kernels_sz;

        do {
            dx = tmp.find("Name:", dx);

            if (dx == std::string::npos) break;

            static constexpr decltype(tmp.size()) name_sz{5};
            dx = tmp.find_first_not_of(" '", dx + name_sz);

            auto fn = tmp.substr(dx, tmp.find_first_of("'\n", dx) - dx);
            dx += fn.size();

            auto dx1 = tmp.find("CodeProps", dx);
            dx = tmp.find("Args:", dx);

            if (dx1 < dx) {
                dx = dx1;
                continue;
            }
            if (dx == std::string::npos) break;

            static constexpr decltype(tmp.size()) args_sz{5};
            dx = parse_args(tmp, dx + args_sz, dx1, kernargs[fn]);
        } while (true);
    }
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
                std::stringstream tmp{std::string{blob.cbegin(), blob.cend()}};

                ELFIO::elfio reader;

                if (!reader.load(tmp)) continue;

                read_kernarg_metadata(reader, r);
            }
        }
    });

    return r;
}
}  // Namespace hip_impl.

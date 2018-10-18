#include "../include/hip/hcc_detail/program_state.hpp"

#include "../include/hip/hcc_detail/code_object_bundle.hpp"

#include "hip_hcc_internal.h"
#include "hsa_helpers.hpp"
#include "trace_helper.h"

#include "elfio/elfio.hpp"

#include <link.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace ELFIO;
using namespace hip_impl;
using namespace std;

namespace {
struct Symbol {
    string name;
    ELFIO::Elf64_Addr value = 0;
    Elf_Xword size = 0;
    Elf_Half sect_idx = 0;
    uint8_t bind = 0;
    uint8_t type = 0;
    uint8_t other = 0;
};

inline Symbol read_symbol(const symbol_section_accessor& section, unsigned int idx) {
    assert(idx < section.get_symbols_num());

    Symbol r;
    section.get_symbol(idx, r.name, r.value, r.size, r.bind, r.type, r.sect_idx, r.other);

    return r;
}

template <typename P>
inline section* find_section_if(elfio& reader, P p) {
    const auto it = find_if(reader.sections.begin(), reader.sections.end(), move(p));

    return it != reader.sections.end() ? *it : nullptr;
}

vector<string> copy_names_of_undefined_symbols(const symbol_section_accessor& section) {
    vector<string> r;

    for (auto i = 0u; i != section.get_symbols_num(); ++i) {
        // TODO: this is boyscout code, caching the temporaries
        //       may be of worth.

        auto tmp = read_symbol(section, i);
        if (tmp.sect_idx == SHN_UNDEF && !tmp.name.empty()) {
            r.push_back(std::move(tmp.name));
        }
    }

    return r;
}

const std::unordered_map<std::string, std::pair<ELFIO::Elf64_Addr, ELFIO::Elf_Xword>>&
symbol_addresses() {
    static unordered_map<string, pair<Elf64_Addr, Elf_Xword>> r;
    static once_flag f;

    call_once(f, []() {
        dl_iterate_phdr(
            [](dl_phdr_info* info, size_t, void*) {
                static constexpr const char self[] = "/proc/self/exe";
                elfio reader;

                static unsigned int iter = 0u;
                if (reader.load(!iter ? self : info->dlpi_name)) {
                    auto it = find_section_if(
                        reader, [](const class section* x) { return x->get_type() == SHT_SYMTAB; });

                    if (it) {
                        const symbol_section_accessor symtab{reader, it};

                        for (auto i = 0u; i != symtab.get_symbols_num(); ++i) {
                            auto tmp = read_symbol(symtab, i);

                            if (tmp.type == STT_OBJECT && tmp.sect_idx != SHN_UNDEF) {
                                const auto addr = tmp.value + (iter ? info->dlpi_addr : 0);
                                r.emplace(move(tmp.name), make_pair(addr, tmp.size));
                            }
                        }
                    }

                    ++iter;
                }

                return 0;
            },
            nullptr);
    });

    return r;
}

void associate_code_object_symbols_with_host_allocation(const elfio& reader,
                                                        section* code_object_dynsym,
                                                        hsa_agent_t agent,
                                                        hsa_executable_t executable) {
    if (!code_object_dynsym) return;

    const auto undefined_symbols =
        copy_names_of_undefined_symbols(symbol_section_accessor{reader, code_object_dynsym});

    for (auto&& x : undefined_symbols) {
        if (globals().find(x) != globals().cend()) return;

        const auto it1 = symbol_addresses().find(x);

        if (it1 == symbol_addresses().cend()) {
            throw runtime_error{"Global symbol: " + x + " is undefined."};
        }

        static mutex mtx;
        lock_guard<mutex> lck{mtx};

        if (globals().find(x) != globals().cend()) return;
        globals().emplace(x, (void*)(it1->second.first));
        void* p = nullptr;
        hsa_amd_memory_lock(reinterpret_cast<void*>(it1->second.first), it1->second.second,
                            nullptr,  // All agents.
                            0, &p);

        hsa_executable_agent_global_variable_define(executable, agent, x.c_str(), p);
    }
}

vector<char> code_object_blob_for_process() {
    static constexpr const char self[] = "/proc/self/exe";
    static constexpr const char kernel_section[] = ".kernel";

    elfio reader;

    if (!reader.load(self)) {
        throw runtime_error{"Failed to load ELF file for current process."};
    }

    auto kernels =
        find_section_if(reader, [](const section* x) { return x->get_name() == kernel_section; });

    vector<char> r;
    if (kernels) {
        r.insert(r.end(), kernels->get_data(), kernels->get_data() + kernels->get_size());
    }

    return r;
}

const unordered_map<hsa_isa_t, vector<vector<char>>>& code_object_blobs() {
    static unordered_map<hsa_isa_t, vector<vector<char>>> r;
    static once_flag f;

    call_once(f, []() {
        static vector<vector<char>> blobs{code_object_blob_for_process()};

        dl_iterate_phdr(
            [](dl_phdr_info* info, std::size_t, void*) {
                elfio tmp;
                if (tmp.load(info->dlpi_name)) {
                    const auto it = find_section_if(
                        tmp, [](const section* x) { return x->get_name() == ".kernel"; });

                    if (it) blobs.emplace_back(it->get_data(), it->get_data() + it->get_size());
                }
                return 0;
            },
            nullptr);

        for (auto&& blob : blobs) {
            for (auto sub_blob = blob.begin(); sub_blob != blob.end(); ) {
                Bundled_code_header tmp(sub_blob, blob.end());
                if (valid(tmp)) {
                    for (auto&& bundle : bundles(tmp)) {
                        r[triple_to_hsa_isa(bundle.triple)].push_back(bundle.blob);
                    }
                    sub_blob+=tmp.bundled_code_size;
                }
                else {
                    break;
                }
            }
        }
    });

    return r;
}

vector<pair<uintptr_t, string>> function_names_for(const elfio& reader, section* symtab) {
    vector<pair<uintptr_t, string>> r;
    symbol_section_accessor symbols{reader, symtab};

    for (auto i = 0u; i != symbols.get_symbols_num(); ++i) {
        // TODO: this is boyscout code, caching the temporaries
        //       may be of worth.
        auto tmp = read_symbol(symbols, i);

        if (tmp.type == STT_FUNC && tmp.sect_idx != SHN_UNDEF && !tmp.name.empty()) {
            r.emplace_back(tmp.value, tmp.name);
        }
    }

    return r;
}

const vector<pair<uintptr_t, string>>& function_names_for_process() {
    static constexpr const char self[] = "/proc/self/exe";

    static vector<pair<uintptr_t, string>> r;
    static once_flag f;

    call_once(f, []() {
        elfio reader;

        if (!reader.load(self)) {
            throw runtime_error{"Failed to load the ELF file for the current process."};
        }

        auto symtab =
            find_section_if(reader, [](const section* x) { return x->get_type() == SHT_SYMTAB; });

        if (symtab) r = function_names_for(reader, symtab);
    });

    return r;
}

const unordered_map<string, vector<hsa_executable_symbol_t>>& kernels() {
    static unordered_map<string, vector<hsa_executable_symbol_t>> r;
    static once_flag f;

    call_once(f, []() {
        static const auto copy_kernels = [](hsa_executable_t, hsa_agent_t,
                                            hsa_executable_symbol_t s, void*) {
            if (type(s) == HSA_SYMBOL_KIND_KERNEL) r[name(s)].push_back(s);

            return HSA_STATUS_SUCCESS;
        };

        for (auto&& agent_executables : executables()) {
            for (auto&& executable : agent_executables.second) {
                hsa_executable_iterate_agent_symbols(executable, agent_executables.first,
                                                     copy_kernels, nullptr);
            }
        }
    });

    return r;
}

void load_code_object_and_freeze_executable(
    const string& file, hsa_agent_t agent,
    hsa_executable_t
        executable) {  // TODO: the following sequence is inefficient, should be refactored
    //       into a single load of the file and subsequent ELFIO
    //       processing.
    static const auto cor_deleter = [](hsa_code_object_reader_t* p) {
        if (p) {
            hsa_code_object_reader_destroy(*p);
            delete p;
        }
    };

    using RAII_code_reader = unique_ptr<hsa_code_object_reader_t, decltype(cor_deleter)>;

    if (!file.empty()) {
        RAII_code_reader tmp{new hsa_code_object_reader_t, cor_deleter};
        hsa_code_object_reader_create_from_memory(file.data(), file.size(), tmp.get());

        hsa_executable_load_agent_code_object(executable, agent, *tmp, nullptr, nullptr);

        hsa_executable_freeze(executable, nullptr);

        static vector<RAII_code_reader> code_readers;
        static mutex mtx;

        lock_guard<mutex> lck{mtx};
        code_readers.push_back(move(tmp));
    }
}
}  // namespace

namespace hip_impl {
const unordered_map<hsa_agent_t, vector<hsa_executable_t>>&
executables() {  // TODO: This leaks the hsa_executable_ts, it should use RAII.
    static unordered_map<hsa_agent_t, vector<hsa_executable_t>> r;
    static once_flag f;

    call_once(f, []() {
        static const auto accelerators = hc::accelerator::get_all();

        for (auto&& acc : accelerators) {
            auto agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

            if (!agent || !acc.is_hsa_accelerator()) continue;

            hsa_agent_iterate_isas(*agent,
                                   [](hsa_isa_t x, void* pa) {
                                       const auto it = code_object_blobs().find(x);

                                       if (it != code_object_blobs().cend()) {
                                           hsa_agent_t a = *static_cast<hsa_agent_t*>(pa);

                                           for (auto&& blob : it->second) {
                                               hsa_executable_t tmp = {};

                                               hsa_executable_create_alt(
                                                   HSA_PROFILE_FULL,
                                                   HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                                   &tmp);

                                               // TODO: this is massively inefficient and only
                                               //       meant for illustration.
                                               string blob_to_str{blob.cbegin(), blob.cend()};
                                               tmp = load_executable(blob_to_str, tmp, a);

                                               if (tmp.handle) r[a].push_back(tmp);
                                           }
                                       }

                                       return HSA_STATUS_SUCCESS;
                                   },
                                   agent);
        }
    });

    return r;
}

const unordered_map<uintptr_t, string>& function_names() {
    static unordered_map<uintptr_t, string> r{function_names_for_process().cbegin(),
                                              function_names_for_process().cend()};
    static once_flag f;

    call_once(f, []() {
        dl_iterate_phdr(
            [](dl_phdr_info* info, size_t, void*) {
                elfio tmp;
                if (tmp.load(info->dlpi_name)) {
                    const auto it = find_section_if(
                        tmp, [](const section* x) { return x->get_type() == SHT_SYMTAB; });

                    if (it) {
                        auto n = function_names_for(tmp, it);

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

const unordered_map<uintptr_t, vector<pair<hsa_agent_t, Kernel_descriptor>>>& functions() {
    static unordered_map<uintptr_t, vector<pair<hsa_agent_t, Kernel_descriptor>>> r;
    static once_flag f;

    call_once(f, []() {
        for (auto&& function : function_names()) {
            const auto it = kernels().find(function.second);

            if (it != kernels().cend()) {
                for (auto&& kernel_symbol : it->second) {
                    r[function.first].emplace_back(
                        agent(kernel_symbol),
                        Kernel_descriptor{kernel_object(kernel_symbol), it->first});
                }
            }
        }
    });

    return r;
}

unordered_map<string, void*>& globals() {
    static unordered_map<string, void*> r;
    static once_flag f;
    call_once(f, []() { r.reserve(symbol_addresses().size()); });

    return r;
}

hsa_executable_t load_executable(const string& file, hsa_executable_t executable,
                                 hsa_agent_t agent) {
    elfio reader;
    stringstream tmp{file};

    if (!reader.load(tmp)) return hsa_executable_t{};

    const auto code_object_dynsym = find_section_if(
        reader, [](const ELFIO::section* x) { return x->get_type() == SHT_DYNSYM; });

    associate_code_object_symbols_with_host_allocation(reader, code_object_dynsym, agent,
                                                       executable);

    load_code_object_and_freeze_executable(file, agent, executable);

    return executable;
}

// HIP startup kernel loader logic
// When enabled HIP_STARTUP_LOADER, HIP will load the kernels and setup
// the function symbol map on program startup
extern "C" void __attribute__((constructor)) __startup_kernel_loader_init() {
    int hip_startup_loader=0;
    if (std::getenv("HIP_STARTUP_LOADER"))
        hip_startup_loader = atoi(std::getenv("HIP_STARTUP_LOADER"));
    if (hip_startup_loader) functions();
}

extern "C" void __attribute__((destructor)) __startup_kernel_loader_fini() {
}

}  // Namespace hip_impl.

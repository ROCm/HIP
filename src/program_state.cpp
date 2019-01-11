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
symbol_addresses(bool rebuild = false) {
    static unordered_map<string, pair<Elf64_Addr, Elf_Xword>> r;
    static once_flag f;

    auto cons = [rebuild]() {
        if (rebuild) {
            r.clear();
        }

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
    };

    call_once(f, cons);
    if (rebuild) {
        cons();
    }

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

const unordered_map<hsa_isa_t, vector<vector<char>>>& code_object_blobs(bool rebuild = false) {
    static unordered_map<hsa_isa_t, vector<vector<char>>> r;
    static once_flag f;

    auto cons = [rebuild]() {
        // names of shared libraries who .kernel sections already loaded
        static unordered_set<string> lib_names;
        static vector<vector<char>> blobs{code_object_blob_for_process()};

        if (rebuild) {
            r.clear();
            blobs.clear();
        }

        dl_iterate_phdr(
            [](dl_phdr_info* info, std::size_t, void*) {
                elfio tmp;
                if ((lib_names.find(info->dlpi_name) == lib_names.end()) &&
                    (tmp.load(info->dlpi_name))) {
                    const auto it = find_section_if(
                        tmp, [](const section* x) { return x->get_name() == ".kernel"; });

                    if (it) {
                        blobs.emplace_back(
                            it->get_data(), it->get_data() + it->get_size());
                        // register the shared library as already loaded
                        lib_names.emplace(info->dlpi_name);
                    }
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
    };


    call_once(f, cons);
    if (rebuild) {
        cons();
    }

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

const vector<pair<uintptr_t, string>>& function_names_for_process(bool rebuild = false) {
    static constexpr const char self[] = "/proc/self/exe";

    static vector<pair<uintptr_t, string>> r;
    static once_flag f;

    auto cons = [rebuild]() {
        elfio reader;

        if (!reader.load(self)) {
            throw runtime_error{"Failed to load the ELF file for the current process."};
        }

        auto symtab =
            find_section_if(reader, [](const section* x) { return x->get_type() == SHT_SYMTAB; });

        if (symtab) r = function_names_for(reader, symtab);
    };

    call_once(f, cons);
    if (rebuild) {
        cons();
    }

    return r;
}

const unordered_map<string, vector<hsa_executable_symbol_t>>& kernels(bool rebuild = false) {
    static unordered_map<string, vector<hsa_executable_symbol_t>> r;
    static once_flag f;

    auto cons = [rebuild]() {
        if (rebuild) {
            r.clear();
            executables(rebuild);
        }

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
    };

    call_once(f, cons);
    if (rebuild) {
        cons();
    }

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

size_t parse_args(
    const string& metadata,
    size_t f,
    size_t l,
    vector<pair<size_t, size_t>>& size_align) {
    if (f == l) return f;
    if (!size_align.empty()) return l;

    do {
        static constexpr size_t size_sz{5};
        f = metadata.find("Size:", f) + size_sz;

        if (l <= f) return f;

        auto size = strtoul(&metadata[f], nullptr, 10);

        static constexpr size_t align_sz{6};
        f = metadata.find("Align:", f) + align_sz;

        char* l{};
        auto align = strtoul(&metadata[f], &l, 10);

        f += (l - &metadata[f]) + 1;

        size_align.emplace_back(size, align);
    } while (true);
}

void read_kernarg_metadata(
    elfio& reader,
    unordered_map<string, vector<pair<size_t, size_t>>>& kernargs)
{   // TODO: this is inefficient.
    auto it = find_section_if(
        reader, [](const section* x) { return x->get_type() == SHT_NOTE; });

    if (!it) return;

    const note_section_accessor acc{reader, it};
    for (decltype(acc.get_notes_num()) i = 0; i != acc.get_notes_num(); ++i) {
        ELFIO::Elf_Word type{};
        string name{};
        void* desc{};
        Elf_Word desc_size{};

        acc.get_note(i, type, name, desc, desc_size);

        if (name != "AMD") continue; // TODO: switch to using NT_AMD_AMDGPU_HSA_METADATA.

        string tmp{
            static_cast<char*>(desc), static_cast<char*>(desc) + desc_size};

        auto dx = tmp.find("Kernels:");

        if (dx == string::npos) continue;

        static constexpr decltype(tmp.size()) kernels_sz{8};
        dx += kernels_sz;

        do {
            dx = tmp.find("Name:", dx);

            if (dx == string::npos) break;

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
            if (dx == string::npos) break;

            static constexpr decltype(tmp.size()) args_sz{5};
            dx = parse_args(tmp, dx + args_sz, dx1, kernargs[fn]);
        } while (true);
    }
}
}  // namespace

namespace hip_impl {
const unordered_map<hsa_agent_t, vector<hsa_executable_t>>&
executables(bool rebuild) {  // TODO: This leaks the hsa_executable_ts, it should use RAII.
    static unordered_map<hsa_agent_t, vector<hsa_executable_t>> r;
    static once_flag f;

    auto cons = [rebuild]() {
        static const auto accelerators = hc::accelerator::get_all();

        if (rebuild) {
            // do NOT clear r so we reuse instances of hsa_executable_t
            // created previously
            code_object_blobs(rebuild);
        }

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
    };

    call_once(f, cons);
    if (rebuild) {
        cons();
    }

    return r;
}

const unordered_map<uintptr_t, string>& function_names(bool rebuild) {
    static unordered_map<uintptr_t, string> r{function_names_for_process().cbegin(),
                                              function_names_for_process().cend()};
    static once_flag f;

    auto cons = [rebuild]() {
        if (rebuild) {
            r.clear();
            function_names_for_process(rebuild);
            r.insert(function_names_for_process().cbegin(),
                     function_names_for_process().cend());
        }

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
    };

    call_once(f, cons);
    if (rebuild) {
        static mutex mtx;
        lock_guard<mutex> lck{mtx};
        cons();
    }

    return r;
}

const unordered_map<uintptr_t, vector<pair<hsa_agent_t, Kernel_descriptor>>>& functions(bool rebuild) {
    static unordered_map<uintptr_t, vector<pair<hsa_agent_t, Kernel_descriptor>>> r;
    static once_flag f;

    auto cons = [rebuild]() {
        if (rebuild) {
            // do NOT clear r so we reuse instances of pair<hsa_agent_t, Kernel_descriptor>
            // created previously

            function_names(rebuild);
            kernargs(rebuild);
            kernels(rebuild);
            globals(rebuild);
        }

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
    };

    call_once(f, cons);
    if (rebuild) {
        static mutex mtx;
        lock_guard<mutex> lck{mtx};
        cons();
    }

    return r;
}

unordered_map<string, void*>& globals(bool rebuild) {
    static unordered_map<string, void*> r;
    static once_flag f;
    auto cons =[rebuild]() {
        if (rebuild) {
            r.clear();
            symbol_addresses(rebuild);
        }

        r.reserve(symbol_addresses().size());
    };

    call_once(f, cons);
    if (rebuild) {
        cons();
    }

    return r;
}

const unordered_map<string, vector<pair<size_t, size_t>>>& kernargs(
    bool rebuild) {
    static unordered_map<string, vector<pair<size_t, size_t>>> r;
    static once_flag f;

    static const auto build_map = [](decltype(r)& x) {
        for (auto&& isa_blobs : code_object_blobs()) {
            for (auto&& blob : isa_blobs.second) {
                stringstream tmp{std::string{blob.cbegin(), blob.cend()}};

                elfio reader;
                if (!reader.load(tmp)) continue;

                read_kernarg_metadata(reader, x);
            }
        }
    };
    call_once(f, []() { r.reserve(function_names().size()); build_map(r); });

    if (rebuild) {
        static mutex mtx;
        thread_local static decltype(r) tmp;

        {
            lock_guard<mutex> lck{mtx};

            tmp.insert(r.cbegin(), r.cend()); // Should use merge in C++17.
        }

        build_map(tmp);

        lock_guard<mutex> lck{mtx};

        r.insert(tmp.cbegin(), tmp.cend());
    }

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
    if (hip_startup_loader) functions(true);
}

extern "C" void __attribute__((destructor)) __startup_kernel_loader_fini() {
}

}  // Namespace hip_impl.

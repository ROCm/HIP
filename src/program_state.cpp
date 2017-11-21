#include "../include/hip/hcc_detail/program_state.hpp"

#include "../include/hip/hcc_detail/code_object_bundle.hpp"

#include "hip_hcc_internal.h"
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

namespace std
{
    template<>
    struct hash<hsa_isa_t> {
        size_t operator()(hsa_isa_t x) const
        {
            return hash<decltype(x.handle)>{}(x.handle);
        }
    };
}

inline
constexpr
bool operator==(hsa_isa_t x, hsa_isa_t y)
{
    return x.handle == y.handle;
}

namespace
{
    vector<string> copy_names_of_undefined_symbols(
        const symbol_section_accessor& section)
    {
        vector<string> r;

        for (auto i = 0u; i != section.get_symbols_num(); ++i) {
            // TODO: this is boyscout code, caching the temporaries
            //       may be of worth.
            string name;
            Elf64_Addr value = 0;
            Elf_Xword size = 0;
            Elf_Half sect_idx = 0;
            uint8_t bind = 0;
            uint8_t type = 0;
            uint8_t other = 0;

            section.get_symbol(
                i, name, value, size, bind, type, sect_idx, other);

            if (sect_idx == SHN_UNDEF && !name.empty()) {
                r.push_back(std::move(name));
            }
        }

        return r;
    }

    pair<Elf64_Addr, Elf_Xword> find_symbol_address(
        const symbol_section_accessor& section,
        const string& symbol_name)
    {
        static constexpr pair<Elf64_Addr, Elf_Xword> r{0, 0};

        for (auto i = 0u; i != section.get_symbols_num(); ++i) {
            // TODO: this is boyscout code, caching the temporaries
            //       may be of worth.
            string name;
            Elf64_Addr value = 0;
            Elf_Xword size = 0;
            Elf_Half sect_idx = 0;
            uint8_t bind = 0;
            uint8_t type = 0;
            uint8_t other = 0;

            section.get_symbol(
                i, name, value, size, bind, type, sect_idx, other);

            if (name == symbol_name) return make_pair(value, size);
        }

        return r;
    }

    void associate_code_object_symbols_with_host_allocation(
        const elfio& reader,
        const elfio& self_reader,
        section* code_object_dynsym,
        section* process_symtab,
        hsa_agent_t agent,
        hsa_executable_t executable)
    {
        if (!code_object_dynsym || !process_symtab) return;

        const auto undefined_symbols = copy_names_of_undefined_symbols(
            symbol_section_accessor{reader, code_object_dynsym});

        for (auto&& x : undefined_symbols) {
            const auto tmp = find_symbol_address(
                symbol_section_accessor{self_reader, process_symtab}, x);

            if (!tmp.first) {
                throw runtime_error{
                    "The global variable: " + x + ", could not be found."};
            }

            static unordered_map<
                Elf64_Addr,
                unique_ptr<void, decltype(hsa_amd_memory_unlock)*>> globals;

            if (globals.count(tmp.first) == 0) {
                void* p = nullptr;
                hsa_amd_memory_lock(
                    reinterpret_cast<void*>(tmp.first),
                    tmp.second,
                    &agent,
                    1,
                    &p);

                static mutex mtx;

                lock_guard<std::mutex> lck{mtx};
                globals.emplace(
                    piecewise_construct,
                    make_tuple(tmp.first),
                    make_tuple(p, hsa_amd_memory_unlock));
            }

            const auto it = globals.find(tmp.first);

            assert(it != globals.cend());

            hsa_executable_agent_global_variable_define(
                executable, agent, x.c_str(), it->second.get());
        }
    }

    template<typename P>
    inline
    section* find_section_if(elfio& reader, P p)
    {
        const auto it = find_if(
            reader.sections.begin(), reader.sections.end(), std::move(p));

        return it != reader.sections.end() ? *it : nullptr;
    }

    vector<uint8_t> code_object_blob_for_process()
    {
        static constexpr const char self[] = "/proc/self/exe";
        static constexpr const char kernel_section[] = ".kernel";

        elfio reader;

        if (!reader.load(self)) {
            throw runtime_error{"Failed to load ELF file for current process."};
        }

        auto kernels = find_section_if(reader, [](const section* x) {
            return x->get_name() == kernel_section;
        });

        vector<uint8_t> r;
        if (kernels) {
            r.insert(
                r.end(),
                kernels->get_data(),
                kernels->get_data() + kernels->get_size());
        }

        return r;
    }

    const unordered_map<hsa_isa_t, vector<vector<uint8_t>>>& code_object_blobs()
    {
        static unordered_map<hsa_isa_t, vector<vector<uint8_t>>> r;
        static once_flag f;

        call_once(f, []() {
            static vector<vector<uint8_t>> blobs{
                code_object_blob_for_process()};

            dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void*) {
                elfio tmp;
                if (tmp.load(info->dlpi_name)) {
                    const auto it = find_section_if(tmp, [](const section* x) {
                        return x->get_name() == ".kernel";
                    });

                    if (it) blobs.emplace_back(
                        it->get_data(), it->get_data() + it->get_size());
                }
                return 0;
            }, nullptr);

            for (auto&& blob : blobs) {
                Bundled_code_header tmp{blob};
                if (valid(tmp)) {
                    for (auto&& bundle : bundles(tmp)) {
                        r[triple_to_hsa_isa(bundle.triple)]
                            .push_back(bundle.blob);
                    }
                }
            }
        });

        return r;
    }

    vector<pair<uintptr_t, string>> function_names_for(
        const elfio& reader, section* symtab)
    {
        vector<pair<uintptr_t, string>> r;
        symbol_section_accessor symbols{reader, symtab};

        auto foo = reader.get_entry();

        for (auto i = 0u; i != symbols.get_symbols_num(); ++i) {
            // TODO: this is boyscout code, caching the temporaries
            //       may be of worth.
            string name;
            Elf64_Addr value = 0;
            Elf_Xword size = 0;
            Elf_Half sect_idx = 0;
            uint8_t bind = 0;
            uint8_t type = 0;
            uint8_t other = 0;

            symbols.get_symbol(
                i, name, value, size, bind, type, sect_idx, other);

            if (type == STT_FUNC && sect_idx != SHN_UNDEF && !name.empty()) {
                r.emplace_back(value, name);
            }
        }

        return r;
    }

    const vector<pair<uintptr_t, string>>& function_names_for_process()
    {
        static constexpr const char self[] = "/proc/self/exe";

        static vector<pair<uintptr_t, string>> r;
        static once_flag f;

        call_once(f, []() {
            elfio reader;

            if (!reader.load(self)) {
                throw runtime_error{
                    "Failed to load the ELF file for the current process."};
            }

            auto symtab = find_section_if(reader, [](const section* x) {
                return x->get_type() == SHT_SYMTAB;
            });

            r = function_names_for(reader, symtab);
        });

        return r;
    }

    inline
    hsa_agent_t agent(hsa_executable_symbol_t x)
    {
        hsa_agent_t r = {};
        hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &r);

        return r;
    }

    inline
    uint32_t group_size(hsa_executable_symbol_t x)
    {
        uint32_t r = 0u;
        hsa_executable_symbol_get_info(
            x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &r);

        return r;
    }

    inline
    uint64_t kernel_object(hsa_executable_symbol_t x)
    {
        uint64_t r = 0u;
        hsa_executable_symbol_get_info(
            x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &r);

        return r;
    }

    inline
    string name(hsa_executable_symbol_t x)
    {
        uint32_t sz = 0u;
        hsa_executable_symbol_get_info(
            x, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &sz);

        string r(sz, '\0');
        hsa_executable_symbol_get_info(
            x, HSA_EXECUTABLE_SYMBOL_INFO_NAME, &r.front());

        return r;
    }

    inline
    uint32_t private_size(hsa_executable_symbol_t x)
    {
        uint32_t r = 0u;
        hsa_executable_symbol_get_info(
            x, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &r);

        return r;
    }

    inline
    hsa_symbol_kind_t type(hsa_executable_symbol_t x)
    {
        hsa_symbol_kind_t r = {};
        hsa_executable_symbol_get_info(x, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &r);

        return r;
    }

    const unordered_map<string, vector<hsa_executable_symbol_t>>& kernels()
    {
        static unordered_map<string, vector<hsa_executable_symbol_t>> r;
        static once_flag f;

        call_once(f, []() {
            static const auto copy_kernels = [](
                hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t s, void*) {
                if (type(s) == HSA_SYMBOL_KIND_KERNEL) r[name(s)].push_back(s);

                return HSA_STATUS_SUCCESS;
            };

            for (auto&& agent_executables : executables()) {
                for (auto&& executable : agent_executables.second) {
                    hsa_executable_iterate_agent_symbols(
                        executable,
                        agent_executables.first,
                        copy_kernels,
                        nullptr);
                }
            }
        });

        return r;
    }

    void load_code_object_and_freeze_executable(
        istream& file, hsa_agent_t agent, hsa_executable_t executable)
    {   // TODO: the following sequence is inefficient, should be refactored
        //       into a single load of the file and subsequent ELFIO
        //       processing.
        static const auto cor_deleter = [](hsa_code_object_reader_t* p) {
            hsa_code_object_reader_destroy(*p);
        };

        using RAII_code_reader = unique_ptr<
            hsa_code_object_reader_t, decltype(cor_deleter)>;

        file.seekg(0);

        vector<uint8_t> blob{
            istreambuf_iterator<char>{file}, istreambuf_iterator<char>{}};
        RAII_code_reader tmp{new hsa_code_object_reader_t, cor_deleter};
        hsa_code_object_reader_create_from_memory(
            blob.data(), blob.size(), tmp.get());

        hsa_executable_load_agent_code_object(
            executable, agent, *tmp, nullptr, nullptr);

        hsa_executable_freeze(executable, nullptr);

        static vector<RAII_code_reader> code_readers;
        static mutex mtx;

        lock_guard<mutex> lck{mtx};
        code_readers.push_back(move(tmp));
    }
}

namespace hip_impl
{
    const unordered_map<hsa_agent_t, vector<hsa_executable_t>>& executables()
    {
        static unordered_map<hsa_agent_t, vector<hsa_executable_t>> r;
        static once_flag f;

        call_once(f, []() {
            static const auto accelerators = hc::accelerator::get_all();

            for (auto&& acc : accelerators) {
                auto agent = static_cast<hsa_agent_t*>(acc.get_hsa_agent());

                if (!agent) continue;

                hsa_agent_iterate_isas(*agent, [](hsa_isa_t x, void* pa) {
                    const auto it = code_object_blobs().find(x);

                    if (it != code_object_blobs().cend()) {
                        hsa_agent_t a = *static_cast<hsa_agent_t*>(pa);

                        for (auto&& blob : it->second) {
                            hsa_executable_t tmp = {};

                            hsa_executable_create_alt(
                                HSA_PROFILE_FULL,
                                HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                nullptr,
                                &tmp);

                            // TODO: this is massively inefficient and only
                            //       meant for illustration.
                            string blob_to_str{blob.cbegin(), blob.cend()};
                            stringstream istr{blob_to_str};
                            tmp = load_executable(tmp, a, istr);

                            if (tmp.handle) r[a].push_back(tmp);
                        }
                    }

                    return HSA_STATUS_SUCCESS;
                }, agent);
            }
        });

        return r;
    }

    const unordered_map<uintptr_t, string>& function_names()
    {
        static unordered_map<uintptr_t, string> r{
            function_names_for_process().cbegin(),
            function_names_for_process().cend()};
        static once_flag f;

        call_once(f, []() {
            dl_iterate_phdr([](dl_phdr_info* info, size_t, void*) {
                elfio tmp;
                if (tmp.load(info->dlpi_name)) {
                    const auto it = find_section_if(tmp, [](const section* x) {
                        return x->get_type() == SHT_SYMTAB;
                    });

                    if (it) {
                        auto n = function_names_for(tmp, it);

                        for (auto&& f : n) f.first += info->dlpi_addr;

                        r.insert(
                            make_move_iterator(n.begin()),
                            make_move_iterator(n.end()));
                    }
                }

                return 0;
            }, nullptr);
        });

        return r;
    }

    const unordered_map<
        uintptr_t, vector<pair<hsa_agent_t, Kernel_descriptor>>>& functions()
    {
        static unordered_map<
            uintptr_t, vector<pair<hsa_agent_t, Kernel_descriptor>>> r;
        static once_flag f;

        call_once(f, []() {
            for (auto&& function : function_names()) {
                const auto it = kernels().find(function.second);

                if (it != kernels().cend()) {
                    for (auto&& kernel_symbol : it->second) {
                        r[function.first].emplace_back(
                            agent(kernel_symbol),
                            Kernel_descriptor{
                                kernel_object(kernel_symbol),
                                group_size(kernel_symbol),
                                private_size(kernel_symbol),
                                it->first});
                    }
                }
            }
        });

        return r;
    }

    hsa_executable_t load_executable(
        hsa_executable_t executable, hsa_agent_t agent, istream& file)
    {
        elfio reader;
        if (!reader.load(file)) {
            return hsa_executable_t{};
        }
        else {
            // TODO: this may benefit from caching as well.
            elfio self_reader;
            self_reader.load("/proc/self/exe");

            const auto symtab =
                find_section_if(self_reader, [](const ELFIO::section* x) {
                    return x->get_type() == SHT_SYMTAB;
                });

            const auto code_object_dynsym =
                find_section_if(reader, [](const ELFIO::section* x) {
                    return x->get_type() == SHT_DYNSYM;
                });

            associate_code_object_symbols_with_host_allocation(
                reader, self_reader, code_object_dynsym, symtab, agent, executable);

            load_code_object_and_freeze_executable(file, agent, executable);

            return executable;
        }
    }
} // Namespace hip_impl.
#include "../include/hip/hcc_detail/program_state.hpp"

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hip_impl {

[[noreturn]]
void hip_throw(const std::exception&);

std::vector<hsa_agent_t> all_hsa_agents();

template<typename P>
inline
ELFIO::section* find_section_if(ELFIO::elfio& reader, P p) {
    const auto it = std::find_if(
        reader.sections.begin(), reader.sections.end(), std::move(p));

    return it != reader.sections.end() ? *it : nullptr;
}

std::size_t kernargs_size_align::kernargs_size_align::size(std::size_t n) const{
    return (*reinterpret_cast<const std::vector<std::pair<std::size_t, std::size_t>>*>(handle))[n].first;
}

std::size_t kernargs_size_align::alignment(std::size_t n) const{
    return (*reinterpret_cast<const std::vector<std::pair<std::size_t, std::size_t>>*>(handle))[n].second;
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




class program_state_impl {

public:

    std::pair<
        std::once_flag,
        std::unordered_map<
            hsa_isa_t,
            std::vector<std::vector<char>>>> code_object_blobs;

    std::pair<
        std::once_flag,
        std::unordered_map<
            std::string, 
            std::pair<ELFIO::Elf64_Addr, ELFIO::Elf_Xword>>> symbol_addresses;

    std::pair<
        std::once_flag,
        std::unordered_map<hsa_agent_t, std::vector<hsa_executable_t>>> executables;        

    std::pair<
        std::once_flag,
        std::unordered_map<
            std::string, std::vector<hsa_executable_symbol_t>>> kernels;        

    std::pair<
        std::once_flag,
        std::unordered_map<
            std::string, std::vector<std::pair<std::size_t, std::size_t>>>> kernargs;

    std::pair<
        std::once_flag,
        std::unordered_map<std::uintptr_t, std::string>> function_names;

    std::pair<
        std::once_flag,
        std::unordered_map<
            std::uintptr_t,
            std::vector<std::pair<hsa_agent_t, Kernel_descriptor>>>> functions;
      
    std::tuple<
        std::once_flag,
        std::mutex,
        std::unordered_map<std::string, void*>> globals;

    using RAII_code_reader =
        std::unique_ptr<hsa_code_object_reader_t, 
                        std::function<void(hsa_code_object_reader_t*)>>;
    std::pair<
        std::mutex,
        std::vector<RAII_code_reader>> code_readers;

    const std::unordered_map<
        hsa_isa_t, std::vector<std::vector<char>>>& get_code_object_blobs() {

        std::call_once(code_object_blobs.first, [this]() {
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
                        auto& bv = this->code_object_blobs.second[triple_to_hsa_isa(bundle.triple)];
                        bv.push_back(bundle.blob);
                    }

                    it += tmp.bundled_code_size;
                };
            }
        });

        return code_object_blobs.second;
    }

    const std::unordered_map<
        std::string,
        std::pair<ELFIO::Elf64_Addr, ELFIO::Elf_Xword>>& get_symbol_addresses() {

        std::call_once(symbol_addresses.first, [this]() {
            dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void* psi_ptr) {

                if (!psi_ptr)
                    return 0;

                program_state_impl* t = static_cast<program_state_impl*>(psi_ptr);

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
                    t->symbol_addresses.second.emplace(std::move(s.name), std::make_pair(addr, s.size));
                }

                return 0;
            }, this);
        });

        return symbol_addresses.second;
    }

    std::unordered_map<std::string, void*>& get_globals() {
        std::call_once(std::get<0>(globals), [this]() { 
            std::get<2>(this->globals).reserve(this->get_symbol_addresses().size()); 
        });
        return std::get<2>(globals);
    }

    std::mutex& get_globals_mutex() {
        return std::get<1>(globals);
    }

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

    void associate_code_object_symbols_with_host_allocation(
        const ELFIO::elfio& reader,
        ELFIO::section* code_object_dynsym,
        hsa_agent_t agent,
        hsa_executable_t executable) {
        if (!code_object_dynsym) return;

        const auto undefined_symbols = copy_names_of_undefined_symbols(
            ELFIO::symbol_section_accessor{reader, code_object_dynsym});

        auto& g = get_globals();
        auto& g_mutex = get_globals_mutex();
        for (auto&& x : undefined_symbols) {

            if (g.find(x) != g.cend()) return;

            const auto it1 = get_symbol_addresses().find(x);

            if (it1 == get_symbol_addresses().cend()) {
                hip_throw(std::runtime_error{
                    "Global symbol: " + x + " is undefined."});
            }

            std::lock_guard<std::mutex> lck{g_mutex};

            if (g.find(x) != g.cend()) return;

            g.emplace(x, (void*)(it1->second.first));
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

    void load_code_object_and_freeze_executable(
        const std::string& file, hsa_agent_t agent, hsa_executable_t executable) {
        // TODO: the following sequence is inefficient, should be refactored
        //       into a single load of the file and subsequent ELFIO
        //       processing.
        if (file.empty()) return;

        static const auto cor_deleter = [] (hsa_code_object_reader_t* p) {
            if (!p) return;
            hsa_code_object_reader_destroy(*p);
            delete p;
        };

        RAII_code_reader tmp{new hsa_code_object_reader_t, cor_deleter};
        hsa_code_object_reader_create_from_memory(
            file.data(), file.size(), tmp.get());

        hsa_executable_load_agent_code_object(
            executable, agent, *tmp, nullptr, nullptr);

        hsa_executable_freeze(executable, nullptr);

        std::lock_guard<std::mutex> lck{code_readers.first};
        code_readers.second.push_back(move(tmp));
    }


    const std::unordered_map<
        hsa_agent_t, std::vector<hsa_executable_t>>& get_executables() {

        std::call_once(executables.first, [this]() {
            for (auto&& agent : hip_impl::all_hsa_agents()) {

                auto data = std::make_pair(this, &agent);

                hsa_agent_iterate_isas(agent, [](hsa_isa_t x, void* d) {

                    auto& p = *static_cast<decltype(data)*>(d);
                    auto& impl = *(p.first);
                    auto& code_object_blobs = impl.get_code_object_blobs();
                    const auto it = code_object_blobs.find(x);
                    if (it == code_object_blobs.cend()) return HSA_STATUS_SUCCESS;

                    hsa_agent_t a = *static_cast<hsa_agent_t*>(p.second);

                    for (auto&& blob : it->second) {
                        hsa_executable_t tmp = {};

                        hsa_executable_create_alt(
                            HSA_PROFILE_FULL,
                            HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                            nullptr,
                            &tmp);

                        // TODO: this is massively inefficient and only meant for
                        // illustration.
                        tmp = impl.load_executable(blob.data(), blob.size(), tmp, a);

                        if (tmp.handle) impl.executables.second[a].push_back(tmp);
                    }

                    return HSA_STATUS_SUCCESS;
                }, &data);
            }
        });

        return executables.second;
    }
    
    hsa_executable_t load_executable(const char* data,
                                     const size_t data_size,
                                     hsa_executable_t executable,
                                     hsa_agent_t agent) {
        ELFIO::elfio reader;
        std::string ts = std::string(data, data_size);
        std::stringstream tmp{ts};

        if (!reader.load(tmp)) return hsa_executable_t{};
        const auto code_object_dynsym = find_section_if(
            reader, [](const ELFIO::section* x) {
                return x->get_type() == SHT_DYNSYM;
        });

        associate_code_object_symbols_with_host_allocation(reader,
                                                           code_object_dynsym,
                                                           agent, executable);

        load_code_object_and_freeze_executable(ts, agent, executable);

        return executable;
    }

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

    const std::unordered_map<std::uintptr_t, std::string>& get_function_names() {

        std::call_once(function_names.first, [this]() {
            dl_iterate_phdr([](dl_phdr_info* info, std::size_t, void* p) {
                ELFIO::elfio tmp;
                const auto elf =
                    info->dlpi_addr ? info->dlpi_name : "/proc/self/exe";

                if (!tmp.load(elf)) return 0;

                const auto it = find_section_if(tmp, [](const ELFIO::section* x) {
                    return x->get_type() == SHT_SYMTAB;
                });

                if (!it) return 0;

                auto& impl = *static_cast<program_state_impl*>(p);
       
                auto names = impl.function_names_for(tmp, it);
                for (auto&& x : names) x.first += info->dlpi_addr;

                impl.function_names.second.insert(
                    std::make_move_iterator(names.begin()),
                    std::make_move_iterator(names.end()));

                return 0;
            }, this);
        });

        return function_names.second;
    }


};  // class program_state_impl

program_state::program_state() :
    impl(*new program_state_impl) {
}

program_state::~program_state() {
    delete(&impl);
}

void* program_state::global_addr_by_name(const char* name) {
    const auto it = impl.get_globals().find(name);
    if (it == impl.get_globals().end())
      return nullptr;
    else
      return it->second;
}

hsa_executable_t program_state::load_executable(const char* data,
                                                const size_t data_size,
                                                hsa_executable_t executable,
                                                hsa_agent_t agent) {
    return impl.load_executable(data, data_size, executable, agent);
}

const std::unordered_map<
    hsa_agent_t, std::vector<hsa_executable_t>>& program_state::executables() {
    return impl.get_executables();
}

inline
const std::unordered_map<
    std::string, std::vector<hsa_executable_symbol_t>>& kernels(program_state& ps) {

    std::call_once(ps.impl.kernels.first, [&ps]() {
        static const auto copy_kernels = [](
            hsa_executable_t, hsa_agent_t, hsa_executable_symbol_t x, void* p) {
            auto& ps = *static_cast<program_state*>(p);
            if (type(x) == HSA_SYMBOL_KIND_KERNEL) ps.impl.kernels.second[name(x)].push_back(x);

            return HSA_STATUS_SUCCESS;
        };

        for (auto&& agent_executables : ps.impl.get_executables()) {
            for (auto&& executable : agent_executables.second) {
                hsa_executable_iterate_agent_symbols(
                    executable, agent_executables.first, copy_kernels, &ps);
            }
        }
    });

    return ps.impl.kernels.second;
}

inline
const std::unordered_map<
    std::uintptr_t,
    std::vector<std::pair<hsa_agent_t, Kernel_descriptor>>>& functions(program_state& ps) {

    std::call_once(ps.impl.functions.first, [&ps]() {
        for (auto&& function : ps.impl.get_function_names()) {
            const auto it = kernels(ps).find(function.second);

            if (it == kernels(ps).cend()) continue;

            for (auto&& kernel_symbol : it->second) {
                ps.impl.functions.second[function.first].emplace_back(
                    agent(kernel_symbol),
                    Kernel_descriptor{kernel_object(kernel_symbol), it->first});
            }
        }
    });

    return ps.impl.functions.second;
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
const std::unordered_map<
    std::string, std::vector<std::pair<std::size_t, std::size_t>>>& kernargs(program_state& ps) {

    std::call_once(ps.impl.kernargs.first, [&ps]() {
        for (auto&& isa_blobs : ps.impl.get_code_object_blobs()) {
            for (auto&& blob : isa_blobs.second) {
                std::stringstream tmp{std::string{blob.cbegin(), blob.cend()}};

                ELFIO::elfio reader;

                if (!reader.load(tmp)) continue;

                read_kernarg_metadata(reader, ps.impl.kernargs.second);
            }
        }
    });

    return ps.impl.kernargs.second;
}

inline
std::string name(hip_impl::program_state& ps, std::uintptr_t function_address)
{
    const auto it = ps.impl.get_function_names().find(function_address);

    if (it == ps.impl.get_function_names().cend())  {
        hip_throw(std::runtime_error{
            "Invalid function passed to hipLaunchKernelGGL."});
    }

    return it->second;
}

inline
std::string name(hsa_agent_t agent)
{
    char n[64]{};
    hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, n);

    return std::string{n};
}

const Kernel_descriptor& program_state::kernel_descriptor(std::uintptr_t function_address,
                                                          hsa_agent_t agent) {


    auto it0 = functions(*this).find(function_address);

    if (it0 == functions(*this).cend()) {
        hip_throw(std::runtime_error{
            "No device code available for function: " +
            std::string(hip_impl::name(*this, function_address))});
    }

    const auto it1 = std::find_if(
        it0->second.cbegin(),
        it0->second.cend(),
        [=](const std::pair<hsa_agent_t, Kernel_descriptor>& x) {
        return x.first == agent;
    });

    if (it1 == it0->second.cend()) {
        hip_throw(std::runtime_error{
            "No code available for function: " + std::string(hip_impl::name(*this, function_address)) +
            ", for agent: " + name(agent)});
    }

    return it1->second;
}

inline
const std::vector<std::pair<std::size_t, std::size_t>>& 
kernargs_size_align_impl(program_state& ps, std::uintptr_t kernel) {

    auto it = ps.impl.get_function_names().find(kernel);
    if (it == ps.impl.get_function_names().cend()) {
        hip_throw(std::runtime_error{"Undefined __global__ function."});
    }

    auto it1 = kernargs(ps).find(it->second);
    if (it1 == kernargs(ps).end()) {
        hip_throw(std::runtime_error{
            "Missing metadata for __global__ function: " + it->second});
    }

    return it1->second;
}


kernargs_size_align program_state::get_kernargs_size_align(std::uintptr_t kernel) {
  kernargs_size_align t;
  t.handle = reinterpret_cast<const void*>(&kernargs_size_align_impl(*this, kernel));
  return t;
}
};

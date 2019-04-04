#include "../include/hip/hcc_detail/program_state.hpp"



#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>


namespace hip_impl {

class code_object_blobs_map_impl {
public:
    void populate() {
        std::call_once(f, [this]() {
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
    }
    friend class code_object_blobs_map;
private:
    std::unordered_map<hsa_isa_t, std::vector<std::vector<char>>> r;
    std::once_flag f;
};

code_object_blobs_map::code_object_blobs_map() {
    handle = new(code_object_blobs_map_impl);
}

code_object_blobs_map::~code_object_blobs_map() {
    if (handle)
        delete(handle);
}

const std::unordered_map<hsa_isa_t, std::vector<std::vector<char>>>&
code_object_blobs_map::get() {
  handle->populate();
  return handle->r;
}

class function_table_impl {
    void build_table() {
        std::call_once(f, [this]() {
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
    }

private:
    std::unordered_map<
        std::uintptr_t,
        std::vector<std::pair<hsa_agent_t, Kernel_descriptor>>> r;
    std::once_flag f;
};


function_table_impl* function_table::init_function_table() {
  return new(function_table_impl);
}

void function_table::release_function_table(function_table_impl* t) {
  delete(t);
}



};

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

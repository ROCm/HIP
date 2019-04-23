#include "../include/hip/hcc_detail/program_state.hpp"

#include <hsa/hsa.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// contains implementation of program_state_impl
#include "program_state.inl"

namespace hip_impl {

program_state::program_state() :
    impl(new program_state_impl) {
    if (!impl) hip_throw(std::runtime_error {
                         "Unknown error when constructing program state."});
}

program_state::~program_state() {
    delete(impl);
}

void* program_state::global_addr_by_name(const char* name) {
    const auto it = impl->get_globals().find(name);
    if (it == impl->get_globals().end())
      return nullptr;
    else
      return it->second;
}

hsa_executable_t program_state::load_executable(const char* data,
                                                const size_t data_size,
                                                hsa_executable_t executable,
                                                hsa_agent_t agent) {
    return impl->load_executable(data, data_size, executable, agent);
}

const std::unordered_map<
    hsa_agent_t, std::vector<hsa_executable_t>>& program_state::executables() {
    return impl->get_executables();
}

const Kernel_descriptor& program_state::kernel_descriptor(std::uintptr_t function_address,
                                                          hsa_agent_t agent) {
    return impl->kernel_descriptor(function_address, agent);
}

kernargs_size_align program_state::get_kernargs_size_align(std::uintptr_t kernel) {
  kernargs_size_align t;
  t.handle = reinterpret_cast<const void*>(&impl->kernargs_size_align(kernel));
  return t;
}
};

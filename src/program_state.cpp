#include "../include/hip/hcc_detail/program_state.hpp"
// contains implementation of program_state_impl
#include "program_state.inl"

#include <hsa/hsa.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace hip_impl {

    std::size_t kernargs_size_align::kernargs_size_align::size(std::size_t n) const{
        return (*reinterpret_cast<const std::vector<std::pair<std::size_t, std::size_t>>*>(handle))[n].first;
    }

    std::size_t kernargs_size_align::alignment(std::size_t n) const{
        return (*reinterpret_cast<const std::vector<std::pair<std::size_t, std::size_t>>*>(handle))[n].second;
    }

    program_state::program_state() : impl(new program_state_impl) {
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

    hipFunction_t program_state::kernel_descriptor(std::uintptr_t function_address,
        hsa_agent_t agent) {
        auto& kd = impl->kernel_descriptor(function_address, agent);
        return kd;
    }

    kernargs_size_align program_state::get_kernargs_size_align(std::uintptr_t kernel) {
        kernargs_size_align t;
        t.handle = reinterpret_cast<const void*>(&impl->kernargs_size_align(kernel));
        return t;
    }

    std::mutex executables_cache_mutex;
    std::vector<hsa_executable_t>& executables_cache(
        std::string elf, hsa_isa_t isa, hsa_agent_t agent) {
        static std::unordered_map<std::string,
            std::unordered_map<hsa_isa_t,
                std::unordered_map<hsa_agent_t, std::vector<hsa_executable_t>>>> cache;
        return cache[elf][isa][agent];
    }
};

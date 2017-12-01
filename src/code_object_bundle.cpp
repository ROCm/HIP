#include "../include/hip/hcc_detail/code_object_bundle.hpp"

#include <hsa/hsa.h>

#include <cstdint>
#include <string>
#include <vector>

hsa_isa_t hip_impl::triple_to_hsa_isa(const std::string& triple)
{
    static constexpr const char prefix[] = "hcc-amdgcn--amdhsa-gfx";
    static constexpr std::size_t prefix_sz = sizeof(prefix) - 1;

    hsa_isa_t r = {};

    auto idx = triple.find(prefix);

    if (idx != std::string::npos) {
        idx += prefix_sz;
        std::string tmp = "AMD:AMDGPU";
        while (idx != triple.size()) {
            tmp.push_back(':');
            tmp.push_back(triple[idx++]);
        }

        hsa_isa_from_name(tmp.c_str(), &r);
    }

    return r;
}

// DATA - STATICS
constexpr const char hip_impl::Bundled_code_header::magic_string_[];

// CREATORS
hip_impl::Bundled_code_header::Bundled_code_header(
    const std::vector<std::uint8_t>& x)
    : Bundled_code_header{x.cbegin(), x.cend()}
{}
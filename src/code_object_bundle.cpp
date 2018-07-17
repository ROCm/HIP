#include "../include/hip/hcc_detail/code_object_bundle.hpp"

#include <hsa/hsa.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace std;

inline
std::string transmogrify_triple(const std::string& triple)
{
    static constexpr const char old_prefix[]{"hcc-amdgcn--amdhsa-gfx"};
    static constexpr const char new_prefix[]{"hcc-amdgcn-amd-amdhsa--gfx"};

    if (triple.find(old_prefix) == 0) {
        return new_prefix + triple.substr(sizeof(old_prefix) - 1);
    }

    return (triple.find(new_prefix) == 0) ? triple : "";
}

inline
std::string isa_name(std::string triple)
{
    static constexpr const char offload_prefix[]{"hcc-"};

    triple = transmogrify_triple(triple);
    if (triple.empty()) return {};

    triple.erase(0, sizeof(offload_prefix) - 1);

    static hsa_isa_t tmp{};
    static const bool is_old_rocr{
        hsa_isa_from_name(triple.c_str(), &tmp) != HSA_STATUS_SUCCESS};

    if (is_old_rocr) {
        std::string tmp{triple.substr(triple.rfind('x') + 1)};
        triple.replace(0, std::string::npos, "AMD:AMDGPU");

        for (auto&& x : tmp) {
            triple.push_back(':');
            triple.push_back(x);
        }
    }

    return triple;
}

hsa_isa_t hip_impl::triple_to_hsa_isa(const std::string& triple) {
    const std::string isa{isa_name(std::move(triple))};

    if (isa.empty()) return hsa_isa_t({});

    hsa_isa_t r{};

    if(HSA_STATUS_SUCCESS != hsa_isa_from_name(isa.c_str(), &r)) {
        r.handle = 0;
    }

    return r;
}

// DATA - STATICS
constexpr const char hip_impl::Bundled_code_header::magic_string_[];

// CREATORS
hip_impl::Bundled_code_header::Bundled_code_header(const vector<char>& x)
    : Bundled_code_header{x.cbegin(), x.cend()} {}

hip_impl::Bundled_code_header::Bundled_code_header(
    const void* p) {  // This is a pretty terrible interface, useful only because
    // hipLoadModuleData is so poorly specified (for no fault of its own).
    if (!p) return;

    auto ph = static_cast<const Header_*>(p);

    if (!equal(magic_string_, magic_string_ + magic_string_sz_, ph->bundler_magic_string_)) {
        return;
    }

    size_t sz = sizeof(Header_) + ph->bundle_cnt_ * sizeof(Bundled_code::Header);
    auto pb = static_cast<const char*>(p) + sizeof(Header_);
    auto n = ph->bundle_cnt_;
    while (n--) {
        sz += reinterpret_cast<const Bundled_code::Header*>(pb)->bundle_sz;
        pb += sizeof(Bundled_code::Header);
    }

    read(static_cast<const char*>(p), static_cast<const char*>(p) + sz, *this);
}

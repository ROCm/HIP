#include "../include/hip/hcc_detail/code_object_bundle.hpp"

#include <hsa/hsa.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using namespace std;

hsa_isa_t hip_impl::triple_to_hsa_isa(const std::string& triple) {
    static constexpr const char OffloadKind[] = "hcc-";
    hsa_isa_t Isa = {0};

    // Check that Triple larger than hcc- and the prefix matches hcc-
    if ((triple.size() < sizeof(OffloadKind) - 1) || !std::equal(triple.c_str(),  triple.c_str() + sizeof(OffloadKind) - 1, OffloadKind)) {
        return Isa;
    }

    std::string validatedTriple = triple.substr(sizeof(OffloadKind) - 1);
    static constexpr const char newPrefix[] = "amdgcn-amd-amdhsa--gfx";

    // Check if the target triple matches the new prefix
    if ((validatedTriple.size() >= sizeof(newPrefix) - 1) && std::equal(validatedTriple.c_str(), validatedTriple.c_str() + sizeof(newPrefix) - 1, newPrefix)) {
        if (HSA_STATUS_SUCCESS != hsa_isa_from_name(validatedTriple.c_str(), &Isa)) {
            // If new prefix fails, try older prefix incase of older ROCR
            // Supports backwards compatibility with old naming
            Isa = {};
            auto it = std::find_if(
                validatedTriple.cbegin(),
                validatedTriple.cend(),
                [](char x) { return std::isdigit(x); });
            if (std::equal(validatedTriple.cbegin(), it, newPrefix)) {
                std::string tmp = "AMD:AMDGPU";
                while (it != validatedTriple.cend()) {
                    tmp.push_back(':');
                    tmp.push_back(*it++);
                }
                if (HSA_STATUS_SUCCESS != hsa_isa_from_name(tmp.c_str(), &Isa)) {
                    // If it also fails, then unsupported target triple
                    Isa.handle = 0;
                }
            }
        }
    }
    return Isa;
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

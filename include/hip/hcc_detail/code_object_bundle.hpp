/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <hsa/hsa.h>

#include <algorithm>
#include <cstdint>
#include <istream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace hip_impl {
hsa_isa_t triple_to_hsa_isa(const std::string& triple);

struct Bundled_code {
    union Header {
        struct {
            std::uint64_t offset;
            std::uint64_t bundle_sz;
            std::uint64_t triple_sz;
        };
        char cbuf[sizeof(offset) + sizeof(bundle_sz) + sizeof(triple_sz)];
    } header;
    std::string triple;
    std::vector<char> blob;
};

class Bundled_code_header {
    // DATA - STATICS
    static constexpr const char magic_string_[] = "__CLANG_OFFLOAD_BUNDLE__";
    static constexpr auto magic_string_sz_ = sizeof(magic_string_) - 1;

    // DATA
    union Header_ {
        struct {
            char bundler_magic_string_[magic_string_sz_];
            std::uint64_t bundle_cnt_;
        };
        char cbuf_[sizeof(bundler_magic_string_) + sizeof(bundle_cnt_)];
    } header_;
    std::vector<Bundled_code> bundles_;

    // FRIENDS - MANIPULATORS
    template <typename RandomAccessIterator>
    friend inline bool read(RandomAccessIterator f, RandomAccessIterator l,
                            Bundled_code_header& x) {
        if (f == l) return false;

        std::copy_n(f, sizeof(x.header_.cbuf_), x.header_.cbuf_);

        if (valid(x)) {
            x.bundles_.resize(x.header_.bundle_cnt_);

            auto it = f + sizeof(x.header_.cbuf_);
            for (auto&& y : x.bundles_) {
                std::copy_n(it, sizeof(y.header.cbuf), y.header.cbuf);
                it += sizeof(y.header.cbuf);

                y.triple.assign(it, it + y.header.triple_sz);

                std::copy_n(f + y.header.offset, y.header.bundle_sz, std::back_inserter(y.blob));

                it += y.header.triple_sz;

                x.bundled_code_size = std::max(x.bundled_code_size, 
                                               y.header.offset + y.header.bundle_sz);
            }

            return true;
        }

        return false;
    }
    friend inline bool read(const std::vector<char>& blob, Bundled_code_header& x) {
        return read(blob.cbegin(), blob.cend(), x);
    }
    friend inline bool read(std::istream& is, Bundled_code_header& x) {
        return read(
            std::vector<char>{std::istreambuf_iterator<char>{is}, std::istreambuf_iterator<char>{}},
            x);
    }

    // FRIENDS - ACCESSORS
    friend inline bool valid(const Bundled_code_header& x) {
        return std::equal(magic_string_, magic_string_ + magic_string_sz_,
                          x.header_.bundler_magic_string_);
    }
    friend inline const std::vector<Bundled_code>& bundles(const Bundled_code_header& x) {
        return x.bundles_;
    }

   public:
    // CREATORS
    Bundled_code_header() = default;
    template <typename RandomAccessIterator>
    Bundled_code_header(RandomAccessIterator f, RandomAccessIterator l);
    explicit Bundled_code_header(const std::vector<char>& blob);
    explicit Bundled_code_header(const void* maybe_blob);
    Bundled_code_header(const Bundled_code_header&) = default;
    Bundled_code_header(Bundled_code_header&&) = default;
    ~Bundled_code_header() = default;

    // MANIPULATORS
    Bundled_code_header& operator=(const Bundled_code_header&) = default;
    Bundled_code_header& operator=(Bundled_code_header&&) = default;

    size_t bundled_code_size = 0;
};

// CREATORS
template <typename RandomAccessIterator>
Bundled_code_header::Bundled_code_header(RandomAccessIterator f, RandomAccessIterator l)
    : Bundled_code_header{} {
    read(f, l, *this);
}
}  // Namespace hip_impl.

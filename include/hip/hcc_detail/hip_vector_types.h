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

/**
 *  @file  hcc_detail/hip_vector_types.h
 *  @brief Defines the different newt vector types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_VECTOR_TYPES_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_VECTOR_TYPES_H

#if defined(__HCC__) && (__hcc_workweek__ < 16032)
#error("This version of HIP requires a newer version of HCC.");
#endif

#include "hip/hcc_detail/host_defines.h"

#if !defined(_MSC_VER)|| __clang__
    #if __has_attribute(ext_vector_type)
        #define __NATIVE_VECTOR__(n, ...) __attribute__((ext_vector_type(n)))
    #else
        #define __NATIVE_VECTOR__(n, ...) [n]
    #endif

#if defined(__cplusplus)
    #include <array>
    #include <iosfwd>
    #include <type_traits>

    namespace hip_impl {
        template<typename, typename, unsigned int> struct Scalar_accessor;
    } // Namespace hip_impl.
    
    namespace std {
        template<typename T, typename U, unsigned int n>
        struct is_integral<hip_impl::Scalar_accessor<T, U, n>>
            : is_integral<T> {};
        template<typename T, typename U, unsigned int n>
        struct is_floating_point<hip_impl::Scalar_accessor<T, U, n>> 
            : is_floating_point<T> {};
    } // Namespace std.
    
    namespace hip_impl {
        template<typename T, typename Vector, unsigned int idx>
        struct Scalar_accessor {
            struct Address {
                const Scalar_accessor* p;

                __host__ __device__
                operator const T*() const noexcept {
                    return &reinterpret_cast<const T*>(p)[idx];
                }
                __host__ __device__
                operator const T*() const volatile noexcept {
                    return &reinterpret_cast<const T*>(p)[idx];
                }
                __host__ __device__
                operator T*() noexcept {
                    return &reinterpret_cast<T*>(
                        const_cast<Scalar_accessor*>(p))[idx];
                }
                __host__ __device__
                operator T*() volatile noexcept {
                    return &reinterpret_cast<T*>(
                        const_cast<Scalar_accessor*>(p))[idx];
                }
            };

            friend
            inline
            std::ostream& operator<<(std::ostream& os,
                                     const Scalar_accessor& x) noexcept {
                return os << x.data[idx];
            }
            friend
            inline
            std::istream& operator>>(std::istream& is,
                                     Scalar_accessor& x) noexcept {
                T tmp;
                is >> tmp;
                x.data[idx] = tmp;

                return is;
            }

            // Idea from https://t0rakka.silvrback.com/simd-scalar-accessor
            Vector data;

            __host__ __device__
            operator T() const noexcept { return data[idx]; }
            __host__ __device__
            operator T() const volatile noexcept { return data[idx]; }

#ifdef __HIP_ENABLE_VECTOR_SCALAR_ACCESSORY_ENUM_CONVERSION__
            // The conversions to enum are fairly ghastly, but unfortunately used in
            // some pre-existing, difficult to modify, code.
            template<
                typename U,
                typename std::enable_if<
                    !std::is_same<U, T>{} &&
                    std::is_enum<U>{} &&
                    std::is_convertible<
                        T, typename std::enable_if<std::is_enum<U>::value, std::underlying_type<U>>::type::type>{}>::type* = nullptr>
            __host__ __device__
            operator U() const noexcept { return static_cast<U>(data[idx]); }
            template<
                typename U,
                typename std::enable_if<
                    !std::is_same<U, T>{} &&
                    std::is_enum<U>{} &&
                    std::is_convertible<
                        T, typename std::enable_if<std::is_enum<U>::value, std::underlying_type<U>>::type::type>{}>::type* = nullptr>
            __host__ __device__
            operator U() const volatile noexcept { return static_cast<U>(data[idx]); }
#endif

            __host__ __device__
            operator T&() noexcept {
                return reinterpret_cast<
                    T (&)[sizeof(Vector) / sizeof(T)]>(data)[idx];
            }
            __host__ __device__
            operator volatile T&() volatile noexcept {
                return reinterpret_cast<
                    volatile T (&)[sizeof(Vector) / sizeof(T)]>(data)[idx];
            }

            __host__ __device__
            Address operator&() const noexcept { return Address{this}; }
            
            __host__ __device__
            Scalar_accessor& operator=(const Scalar_accessor& x) noexcept {
                data[idx] = x.data[idx];

                return *this;
            }
            __host__ __device__
            Scalar_accessor& operator=(T x) noexcept {
                data[idx] = x;

                return *this;
            }
            __host__ __device__
            volatile Scalar_accessor& operator=(T x) volatile noexcept {
                data[idx] = x;

                return *this;
            }

            __host__ __device__
            Scalar_accessor& operator++() noexcept {
                ++data[idx];
                return *this;
            }
            __host__ __device__
            T operator++(int) noexcept {
                auto r{data[idx]};
                ++data[idx];
                return *this;
            }
            __host__ __device__
            Scalar_accessor& operator--() noexcept {
                --data[idx];
                return *this;
            }
            __host__ __device__
            T operator--(int) noexcept {
                auto r{data[idx]};
                --data[idx];
                return *this;
            }

            // TODO: convertibility is too restrictive, constraint should be on
            //       the operator being invocable with a value of type U.
            template<
                typename U,
                typename std::enable_if<
                    std::is_convertible<U, T>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator+=(U x) noexcept {
                data[idx] += x;
                return *this;
            }
            template<
                typename U,
                typename std::enable_if<
                    std::is_convertible<U, T>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator-=(U x) noexcept {
                data[idx] -= x;
                return *this;
            }

            template<
                typename U,
                typename std::enable_if<
                    std::is_convertible<U, T>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator*=(U x) noexcept {
                data[idx] *= x;
                return *this;
            }
            template<
                typename U,
                typename std::enable_if<
                    std::is_convertible<U, T>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator/=(U x) noexcept {
                data[idx] /= x;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_convertible<U, T>{} &&
                                        std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator%=(U x) noexcept {
                data[idx] %= x;
                return *this;
            }

            template<
                typename U = T,
                typename std::enable_if<std::is_convertible<U, T>{} &&
                                        std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator>>=(U x) noexcept {
                data[idx] >>= x;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_convertible<U, T>{} &&
                                        std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator<<=(U x) noexcept {
                data[idx] <<= x;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_convertible<U, T>{} &&
                                        std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator&=(U x) noexcept {
                data[idx] &= x;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_convertible<U, T>{} &&
                                        std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator|=(U x) noexcept {
                data[idx] |= x;
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_convertible<U, T>{} &&
                                        std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Scalar_accessor& operator^=(U x) noexcept {
                data[idx] ^= x;
                return *this;
            }
        };
    } // Namespace hip_impl.

    template<typename T, unsigned int n> struct HIP_vector_base;

    template<typename T>
    struct HIP_vector_base<T, 1> {
        using Native_vec_ = T __NATIVE_VECTOR__(1, T);

        union {
            Native_vec_ data;
#if __HIP_CLANG_ONLY__
            struct {
                T x;
            };
#else
            hip_impl::Scalar_accessor<T, Native_vec_, 0> x;
#endif
        };

        using value_type = T;
        
        __host__ __device__
        HIP_vector_base& operator=(const HIP_vector_base& x) noexcept {
            #if __has_attribute(ext_vector_type)
                data = x.data;
            #else
                data[0] = x.data[0];
            #endif

            return *this;
        }
    };

    template<typename T>
    struct HIP_vector_base<T, 2> {
        using Native_vec_ = T __NATIVE_VECTOR__(2, T);

        union {
            Native_vec_ data;
#if __HIP_CLANG_ONLY__
            struct {
                T x;
                T y;
            };
#else
            hip_impl::Scalar_accessor<T, Native_vec_, 0> x;
            hip_impl::Scalar_accessor<T, Native_vec_, 1> y;
#endif
        };

        using value_type = T;
        
        __host__ __device__
        HIP_vector_base& operator=(const HIP_vector_base& x) noexcept {
            #if __has_attribute(ext_vector_type)
                data = x.data;
            #else
                data[0] = x.data[0];
                data[1] = x.data[1];
            #endif

            return *this;
        }
    };

    template<typename T>
    struct HIP_vector_base<T, 3> {
        struct Native_vec_ {
            T d[3];

            __host__ __device__
            constexpr
            Native_vec_() = default;
            __host__ __device__
            explicit
            constexpr
            Native_vec_(T x) noexcept : d{x, x, x} {}
            __host__ __device__
            constexpr
            Native_vec_(T x, T y, T z) noexcept : d{x, y, z} {}
            __host__ __device__
            constexpr
            Native_vec_(const Native_vec_&) = default;
            __host__ __device__
            constexpr
            Native_vec_(Native_vec_&&) = default;
            __host__ __device__
            ~Native_vec_() = default;

            __host__ __device__
            Native_vec_& operator=(const Native_vec_&) = default;
            __host__ __device__
            Native_vec_& operator=(Native_vec_&&) = default;

            __host__ __device__
            T& operator[](unsigned int idx) noexcept { return d[idx]; }
            __host__ __device__
            T operator[](unsigned int idx) const noexcept { return d[idx]; }

            __host__ __device__
            Native_vec_& operator+=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] += x.d[i];
                return *this;
            }
            __host__ __device__
            Native_vec_& operator-=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] -= x.d[i];
                return *this;
            }

            __host__ __device__
            Native_vec_& operator*=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] *= x.d[i];
                return *this;
            }
            __host__ __device__
            Native_vec_& operator/=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] /= x.d[i];
                return *this;
            }

            template<
                typename U = T,
                typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
            __host__ __device__
            Native_vec_ operator-() const noexcept
            {
                auto r{*this};
                for (auto&& x : r.d) x = -x;
                return r;
            }

            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Native_vec_ operator~() const noexcept
            {
                auto r{*this};
                for (auto&& x : r.d) x = ~x;
                return r;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Native_vec_& operator%=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] %= x.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Native_vec_& operator^=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] ^= x.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Native_vec_& operator|=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] |= x.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Native_vec_& operator&=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] &= x.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Native_vec_& operator>>=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] >>= x.d[i];
                return *this;
            }
            template<
                typename U = T,
                typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
            __host__ __device__
            Native_vec_& operator<<=(const Native_vec_& x) noexcept
            {
                for (auto i = 0u; i != 3u; ++i) d[i] <<= x.d[i];
                return *this;
            }

            using Vec3_cmp = int __attribute__((vector_size(4 * sizeof(int))));
            __host__ __device__
            Vec3_cmp operator==(const Native_vec_& x) const noexcept
            {
                return Vec3_cmp{d[0] == x.d[0], d[1] == x.d[1], d[2] == x.d[2]};
            }
        };

        union {
            Native_vec_ data;
            struct {
                T x;
                T y;
                T z;
            };
        };

        using value_type = T;
    };

    template<typename T>
    struct HIP_vector_base<T, 4> {
        using Native_vec_ = T __NATIVE_VECTOR__(4, T);

        union {
            Native_vec_ data;
#if __HIP_CLANG_ONLY__
            struct {
                T x;
                T y;
                T z;
                T w;
            };
#else
            hip_impl::Scalar_accessor<T, Native_vec_, 0> x;
            hip_impl::Scalar_accessor<T, Native_vec_, 1> y;
            hip_impl::Scalar_accessor<T, Native_vec_, 2> z;
            hip_impl::Scalar_accessor<T, Native_vec_, 3> w;
#endif            
        };

        using value_type = T;
        
        __host__ __device__
        HIP_vector_base& operator=(const HIP_vector_base& x) noexcept {
            #if __has_attribute(ext_vector_type)
                data = x.data;
            #else
                data[0] = x.data[0];
                data[1] = x.data[1];
                data[2] = x.data[2];
                data[3] = x.data[3];
            #endif

            return *this;
        }
    };

    template<typename T, unsigned int rank>
    struct HIP_vector_type : public HIP_vector_base<T, rank> {
        using HIP_vector_base<T, rank>::data;
        using typename HIP_vector_base<T, rank>::Native_vec_;

        inline __host__ __device__
        HIP_vector_type() = default;
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        explicit inline __host__ __device__
        HIP_vector_type(U x) noexcept
        {
            for (auto i = 0u; i != rank; ++i) data[i] = x;
        }
        template< // TODO: constrain based on type as well.
            typename... Us,
            typename std::enable_if<
                (rank > 1) && sizeof...(Us) == rank>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type(Us... xs) noexcept
        {
            #if __has_attribute(ext_vector_type)
                new (&data) Native_vec_{static_cast<T>(xs)...};
            #else
                new (&data) std::array<T, rank>{static_cast<T>(xs)...};
            #endif
        }
        inline __host__ __device__
        HIP_vector_type(const HIP_vector_type&) = default;
        inline __host__ __device__
        HIP_vector_type(HIP_vector_type&&) = default;
        inline __host__ __device__
        ~HIP_vector_type() = default;

        inline __host__ __device__
        HIP_vector_type& operator=(const HIP_vector_type&) = default;
        inline __host__ __device__
        HIP_vector_type& operator=(HIP_vector_type&&) = default;

        // Operators
        inline __host__ __device__
        HIP_vector_type& operator++() noexcept
        {
            return *this += HIP_vector_type{1};
        }
        inline __host__ __device__
        HIP_vector_type operator++(int) noexcept
        {
            auto tmp(*this);
            ++*this;
            return tmp;
        }

        inline __host__ __device__
        HIP_vector_type& operator--() noexcept
        {
            return *this -= HIP_vector_type{1};
        }
        inline __host__ __device__
        HIP_vector_type operator--(int) noexcept
        {
            auto tmp(*this);
            --*this;
            return tmp;
        }

        inline __host__ __device__
        HIP_vector_type& operator+=(const HIP_vector_type& x) noexcept
        {
            data += x.data;
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator+=(U x) noexcept
        {
            return *this += HIP_vector_type{x};
        }

        inline __host__ __device__
        HIP_vector_type& operator-=(const HIP_vector_type& x) noexcept
        {
            data -= x.data;
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator-=(U x) noexcept
        {
            return *this -= HIP_vector_type{x};
        }

        inline __host__ __device__
        HIP_vector_type& operator*=(const HIP_vector_type& x) noexcept
        {
            data *= x.data;
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator*=(U x) noexcept
        {
            return *this *= HIP_vector_type{x};
        }

        inline __host__ __device__
        HIP_vector_type& operator/=(const HIP_vector_type& x) noexcept
        {
            data /= x.data;
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator/=(U x) noexcept
        {
            return *this /= HIP_vector_type{x};
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type operator-() noexcept
        {
            auto tmp(*this);
            tmp.data = -tmp.data;
            return tmp;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type operator~() noexcept
        {
            HIP_vector_type r{*this};
            r.data = ~r.data;
            return r;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator%=(const HIP_vector_type& x) noexcept
        {
            data %= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator^=(const HIP_vector_type& x) noexcept
        {
            data ^= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator|=(const HIP_vector_type& x) noexcept
        {
            data |= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator&=(const HIP_vector_type& x) noexcept
        {
            data &= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator>>=(const HIP_vector_type& x) noexcept
        {
            data >>= x.data;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        inline __host__ __device__
        HIP_vector_type& operator<<=(const HIP_vector_type& x) noexcept
        {
            data <<= x.data;
            return *this;
        }
    };

    template<typename T, unsigned int n>
    inline __host__ __device__
    HIP_vector_type<T, n> operator+(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} += y;
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    HIP_vector_type<T, n> operator+(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} += HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    HIP_vector_type<T, n> operator+(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} += y;
    }

    template<typename T, unsigned int n>
    inline __host__ __device__
    HIP_vector_type<T, n> operator-(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    HIP_vector_type<T, n> operator-(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    HIP_vector_type<T, n> operator-(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }

    template<typename T, unsigned int n>
    inline __host__ __device__
    HIP_vector_type<T, n> operator*(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= y;
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    HIP_vector_type<T, n> operator*(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    HIP_vector_type<T, n> operator*(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= y;
    }

    template<typename T, unsigned int n>
    inline __host__ __device__
    HIP_vector_type<T, n> operator/(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= y;
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    HIP_vector_type<T, n> operator/(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    HIP_vector_type<T, n> operator/(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= y;
    }

    template<typename T, unsigned int n>
    inline __host__ __device__
    bool operator==(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        auto tmp = x.data == y.data;
        for (auto i = 0u; i != n; ++i) if (tmp[i] == 0) return false;
        return true;
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    bool operator==(const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return x == HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    bool operator==(U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} == y;
    }

    template<typename T, unsigned int n>
    inline __host__ __device__
    bool operator!=(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x == y);
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    bool operator!=(const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return !(x == y);
    }
    template<typename T, unsigned int n, typename U>
    inline __host__ __device__
    bool operator!=(U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x == y);
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator%(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator%(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator%(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator^(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator^(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator^(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator|(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator|(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator|(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator&(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator&(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator&(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator>>(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator>>(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator>>(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator<<(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator<<(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= HIP_vector_type<T, n>{y};
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_arithmetic<U>::value>::type,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline __host__ __device__
    HIP_vector_type<T, n> operator<<(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }

    #define __MAKE_VECTOR_TYPE__(CUDA_name, T) \
        using CUDA_name##1 = HIP_vector_type<T, 1>;\
        using CUDA_name##2 = HIP_vector_type<T, 2>;\
        using CUDA_name##3 = HIP_vector_type<T, 3>;\
        using CUDA_name##4 = HIP_vector_type<T, 4>;
#else
    #define __MAKE_VECTOR_TYPE__(CUDA_name, T) \
        typedef struct {\
            T x;\
        } CUDA_name##1;\
        typedef struct {\
            T x;\
            T y;\
        } CUDA_name##2;\
        typedef struct {\
            T x;\
            T y;\
            T z;\
        } CUDA_name##3;\
        typedef struct {\
            T x;\
            T y;\
            T z;\
            T w;\
        } CUDA_name##4;
#endif

__MAKE_VECTOR_TYPE__(uchar, unsigned char);
__MAKE_VECTOR_TYPE__(char, char);
__MAKE_VECTOR_TYPE__(ushort, unsigned short);
__MAKE_VECTOR_TYPE__(short, short);
__MAKE_VECTOR_TYPE__(uint, unsigned int);
__MAKE_VECTOR_TYPE__(int, int);
__MAKE_VECTOR_TYPE__(ulong, unsigned long);
__MAKE_VECTOR_TYPE__(long, long);
__MAKE_VECTOR_TYPE__(ulonglong, unsigned long long);
__MAKE_VECTOR_TYPE__(longlong, long long);
__MAKE_VECTOR_TYPE__(float, float);
__MAKE_VECTOR_TYPE__(double, double);

#ifdef __cplusplus
#define DECLOP_MAKE_ONE_COMPONENT(comp, type) \
    static inline __device__ __host__ \
    type make_##type(comp x) { type r{x}; return r; }

#define DECLOP_MAKE_TWO_COMPONENT(comp, type) \
    static inline __device__ __host__ \
    type make_##type(comp x, comp y) { type r{x, y}; return r; }

#define DECLOP_MAKE_THREE_COMPONENT(comp, type) \
    static inline __device__ __host__ \
    type make_##type(comp x, comp y, comp z) { type r{x, y, z}; return r; }

#define DECLOP_MAKE_FOUR_COMPONENT(comp, type) \
    static inline __device__ __host__ \
    type make_##type(comp x, comp y, comp z, comp w) { \
        type r{x, y, z, w}; \
        return r; \
    }
#else
 #define DECLOP_MAKE_ONE_COMPONENT(comp, type) \
     static inline __device__ __host__ \
     type make_##type(comp x) { type r; r.x =x; return r; }

 #define DECLOP_MAKE_TWO_COMPONENT(comp, type) \
     static inline __device__ __host__ \
     type make_##type(comp x, comp y) { type r; r.x=x; r.y=y; return r; }

 #define DECLOP_MAKE_THREE_COMPONENT(comp, type) \
     static inline __device__ __host__ \
     type make_##type(comp x, comp y, comp z) { type r; r.x=x; r.y=y; r.z=z; return r; }

 #define DECLOP_MAKE_FOUR_COMPONENT(comp, type) \
     static inline __device__ __host__ \
     type make_##type(comp x, comp y, comp z, comp w) { \
         type r; r.x=x; r.y=y; r.z=z; r.w=w; \
         return r; \
     }
#endif

DECLOP_MAKE_ONE_COMPONENT(unsigned char, uchar1);
DECLOP_MAKE_TWO_COMPONENT(unsigned char, uchar2);
DECLOP_MAKE_THREE_COMPONENT(unsigned char, uchar3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned char, uchar4);

DECLOP_MAKE_ONE_COMPONENT(signed char, char1);
DECLOP_MAKE_TWO_COMPONENT(signed char, char2);
DECLOP_MAKE_THREE_COMPONENT(signed char, char3);
DECLOP_MAKE_FOUR_COMPONENT(signed char, char4);

DECLOP_MAKE_ONE_COMPONENT(unsigned short, ushort1);
DECLOP_MAKE_TWO_COMPONENT(unsigned short, ushort2);
DECLOP_MAKE_THREE_COMPONENT(unsigned short, ushort3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned short, ushort4);

DECLOP_MAKE_ONE_COMPONENT(signed short, short1);
DECLOP_MAKE_TWO_COMPONENT(signed short, short2);
DECLOP_MAKE_THREE_COMPONENT(signed short, short3);
DECLOP_MAKE_FOUR_COMPONENT(signed short, short4);

DECLOP_MAKE_ONE_COMPONENT(unsigned int, uint1);
DECLOP_MAKE_TWO_COMPONENT(unsigned int, uint2);
DECLOP_MAKE_THREE_COMPONENT(unsigned int, uint3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned int, uint4);

DECLOP_MAKE_ONE_COMPONENT(signed int, int1);
DECLOP_MAKE_TWO_COMPONENT(signed int, int2);
DECLOP_MAKE_THREE_COMPONENT(signed int, int3);
DECLOP_MAKE_FOUR_COMPONENT(signed int, int4);

DECLOP_MAKE_ONE_COMPONENT(float, float1);
DECLOP_MAKE_TWO_COMPONENT(float, float2);
DECLOP_MAKE_THREE_COMPONENT(float, float3);
DECLOP_MAKE_FOUR_COMPONENT(float, float4);

DECLOP_MAKE_ONE_COMPONENT(double, double1);
DECLOP_MAKE_TWO_COMPONENT(double, double2);
DECLOP_MAKE_THREE_COMPONENT(double, double3);
DECLOP_MAKE_FOUR_COMPONENT(double, double4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long, ulong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long, ulong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long, ulong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long, ulong4);

DECLOP_MAKE_ONE_COMPONENT(signed long, long1);
DECLOP_MAKE_TWO_COMPONENT(signed long, long2);
DECLOP_MAKE_THREE_COMPONENT(signed long, long3);
DECLOP_MAKE_FOUR_COMPONENT(signed long, long4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long long, ulonglong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long long, ulonglong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long long, ulonglong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long long, ulonglong4);

DECLOP_MAKE_ONE_COMPONENT(signed long long, longlong1);
DECLOP_MAKE_TWO_COMPONENT(signed long long, longlong2);
DECLOP_MAKE_THREE_COMPONENT(signed long long, longlong3);
DECLOP_MAKE_FOUR_COMPONENT(signed long long, longlong4);
#else // defined(_MSC_VER)
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

typedef union { char data; } char1;
typedef union { char data[2]; } char2;
typedef union { char data[4]; } char4;
typedef union { char4 data; } char3;
typedef union { __m64 data; } char8;
typedef union { __m128i data; } char16;

typedef union { unsigned char data; } uchar1;
typedef union { unsigned char data[2]; } uchar2;
typedef union { unsigned char data[4]; } uchar4;
typedef union { uchar4 data; } uchar3;
typedef union { __m64 data; } uchar8;
typedef union { __m128i data; } uchar16;

typedef union { short data; } short1;
typedef union { short data[2]; } short2;
typedef union { __m64 data; } short4;
typedef union { short4 data; } short3;
typedef union { __m128i data; } short8;
typedef union { __m128i data[2]; } short16;

typedef union { unsigned short data; } ushort1;
typedef union { unsigned short data[2]; } ushort2;
typedef union { __m64 data; } ushort4;
typedef union { ushort4 data; } ushort3;
typedef union { __m128i data; } ushort8;
typedef union { __m128i data[2]; } ushort16;

typedef union { int data; } int1;
typedef union { __m64 data; } int2;
typedef union { __m128i data; } int4;
typedef union { int4 data; } int3;
typedef union { __m128i data[2]; } int8;
typedef union { __m128i data[4];} int16;

typedef union { unsigned int data; } uint1;
typedef union { __m64 data; } uint2;
typedef union { __m128i data; } uint4;
typedef union { uint4 data; } uint3;
typedef union { __m128i data[2]; } uint8;
typedef union { __m128i data[4]; } uint16;

#if !defined(_WIN64)
typedef union { int data; } long1;
typedef union { __m64 data; } long2;
typedef union { __m128i data; } long4;
typedef union { long4 data; } long3;
typedef union { __m128i data[2]; } long8;
typedef union { __m128i data[4]; } long16;

typedef union { unsigned int data; } ulong1;
typedef union { __m64 data; } ulong2;
typedef union { __m128i data; } ulong4;
typedef union { ulong4 data; } ulong3;
typedef union { __m128i data[2]; } ulong8;
typedef union { __m128i data[4]; } ulong16;
#else // defined(_WIN64)
typedef union { __m64 data; } long1;
typedef union { __m128i data; } long2;
typedef union { __m128i data[2]; } long4;
typedef union { long4 data; } long3;
typedef union { __m128i data[4]; } long8;
typedef union { __m128i data[8]; } long16;

typedef union { __m64 data; } ulong1;
typedef union { __m128i data; } ulong2;
typedef union { __m128i data[2]; } ulong4;
typedef union { ulong4 data; } ulong3;
typedef union { __m128i data[4]; } ulong8;
typedef union { __m128i data[8]; } ulong16;
#endif // defined(_WIN64)

typedef union { __m64 data; } longlong1;
typedef union { __m128i data; } longlong2;
typedef union { __m128i data[2]; } longlong4;
typedef union { longlong4 data; } longlong3;
typedef union { __m128i data[4]; } longlong8;
typedef union { __m128i data[8]; } longlong16;

typedef union { __m64 data; } ulonglong1;
typedef union { __m128i data; } ulonglong2;
typedef union { __m128i data[2]; } ulonglong4;
typedef union { ulonglong4 data; } ulonglong3;
typedef union { __m128i data[4]; } ulonglong8;
typedef union { __m128i data[8]; } ulonglong16;

typedef union { float data; } float1;
typedef union { __m64 data; } float2;
typedef union { __m128 data; } float4;
typedef union { float4 data; } float3;
typedef union { __m256 data; } float8;
typedef union { __m256 data[2]; } float16;

typedef union { double data; } double1;
typedef union { __m128d data; } double2;
typedef union { __m256d data; } double4;
typedef union { double4 data; } double3;
typedef union { __m256d data[2]; } double8;
typedef union { __m256d data[4]; } double16;

#endif // defined(_MSC_VER)
#endif

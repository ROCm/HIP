/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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

//
// Created by alexv on 25/10/16.
//
#pragma once

#include "concepts.hpp"
#include "helpers.hpp"

#include "hc.hpp"
#include "hcc_acc.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace glo_tests
{
    namespace
    {
        struct New_grid_launch_tag {};
        struct Old_grid_launch_tag {};
    }

    template<FunctionalProcedure F, typename... Ts>
    using is_new_grid_launch_t = typename std::conditional<
        std::is_callable<F(Ts...)>{},
        New_grid_launch_tag,
        Old_grid_launch_tag>::type;

    // TODO: - dispatch rank should be derived from the domain dimensions passed
    //         in, and not always assumed to be 3;

    template<FunctionalProcedure K, typename... Ts>
        requires(Domain<K> == {Ts...})
    static
    inline
    void grid_launch_impl(
        New_grid_launch_tag,
        dim3 num_blocks,
        dim3 dim_blocks,
        int group_mem_bytes,
        hipStream_t stream,
        K k,
        Ts&&... args)
    {
        const auto d = hc::extent<3>{
            num_blocks.z * dim_blocks.z,
            num_blocks.y * dim_blocks.y,
            num_blocks.x * dim_blocks.x}.tile_with_dynamic(
            dim_blocks.z,
            dim_blocks.y,
            dim_blocks.x,
            group_mem_bytes);
        hc::accelerator_view* av = nullptr;

        if (hipHccGetAcceleratorView(stream, &av) != HIP_SUCCESS) {
            throw std::runtime_error{"Failed to retrieve accelerator_view!"};
        }

        hc::parallel_for_each(*av, d, [=](hc::tiled_index<3> idx) [[hc]] {
            k(args...);
        });
    }

    template<FunctionalProcedure K, typename... Ts>
        requires(Domain<K> == {hipLaunchParm, Ts...})
    static
    inline
    void grid_launch_impl(
        Old_grid_launch_tag,
        dim3 num_blocks,
        dim3 dim_blocks,
        int group_mem_bytes,
        hipStream_t stream,
        K k,
        Ts&&... args)
    {
        grid_launch_impl(
            New_grid_launch_tag{},
            std::move(num_blocks),
            std::move(dim_blocks),
            group_mem_bytes,
            std::move(stream),
            std::move(k),
            hipLaunchParm{},
            std::forward<Ts>(args)...);
    }

    template<FunctionalProcedure K, typename... Ts>
        requires(Domain<K> == {Ts...})
    static
    inline
    std::enable_if_t<!std::is_function<K>::value> grid_launch(
        dim3 num_blocks,
        dim3 dim_blocks,
        int group_mem_bytes,
        hipStream_t stream,
        K k,
        Ts&& ... args)
    {
        grid_launch_impl(
            is_new_grid_launch_t<K, Ts...>{},
            std::move(num_blocks),
            std::move(dim_blocks),
            group_mem_bytes,
            std::move(stream),
            std::move(k),
            std::forward<Ts>(args)...);
    }

    namespace
    {
        template<typename T>
        constexpr
        inline
        T&& forward(std::remove_reference_t<T>& x) [[hc]]
        {
            return static_cast<T&&>(x);
        }

        template<FunctionalProcedure K, K* k>
        struct Forwarder {
            template<typename... Ts>
            void operator()(Ts&&...args) const [[hc]]
            {
                k(forward<Ts>(args)...);
            }
        };
    }

    template<FunctionalProcedure K, K* k, typename... Ts>
        requires(Domain<K> == {Ts...})
    static
    inline
    void grid_launch(
        New_grid_launch_tag,
        dim3 num_blocks,
        dim3 dim_blocks,
        int group_mem_bytes,
        hipStream_t stream,
        Ts&&... args)
    {
        grid_launch_impl(
            New_grid_launch_tag{},
            std::move(num_blocks),
            std::move(dim_blocks),
            group_mem_bytes,
            std::move(stream),
            Forwarder<K, k>{},
            std::forward<Ts>(args)...);
    }

    template<FunctionalProcedure K, K* k, typename... Ts>
        requires(Domain<K> == {Ts...})
    static
    inline
    void grid_launch(
        Old_grid_launch_tag,
        dim3 num_blocks,
        dim3 dim_blocks,
        int group_mem_bytes,
        hipStream_t stream,
        Ts&&... args)
    {
        grid_launch<K, k>(
            New_grid_launch_tag{},
            std::move(num_blocks),
            std::move(dim_blocks),
            group_mem_bytes,
            std::move(stream),
            hipLaunchParm{},
            std::forward<Ts>(args)...);
    }

    template<FunctionalProcedure K, K* k, typename... Ts>
        requires(Domain<K> == {Ts...})
    static
    inline
    std::enable_if_t<std::is_function<K>::value> grid_launch(
        dim3 num_blocks,
        dim3 dim_blocks,
        int group_mem_bytes,
        hipStream_t stream,
        Ts&&... args)
    {
        grid_launch<K, k>(
            is_new_grid_launch_t<K*, Ts...>{},
            std::move(num_blocks),
            std::move(dim_blocks),
            group_mem_bytes,
            std::move(stream),
            std::forward<Ts>(args)...);
    }

    namespace
    {
        template<typename K, typename = void> struct Wrapper;

        template<FunctionalProcedure K>
        struct Wrapper<K, std::enable_if_t<!std::is_function<K>::value>> {
            template<typename... Ts>
               requires(Domain<K> == {Ts...})
            void operator()(Ts&&... args) const
            {
                grid_launch(std::forward<Ts>(args)...);
            }
        };

        template<FunctionalProcedure K>
        struct Wrapper<K, std::enable_if_t<std::is_function<K>::value>> {
            template<typename... Ts>
            void operator()(Ts&&...) const {}
        };
    }

    // TODO: these are temporary, they need to be uglified and them completely
    //       removed once we enable C++14 support and can have proper generic,
    //       variadic lambdas.
    #define make_lambda_wrapper21(                                             \
        kernel_name,                                                           \
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,  \
        p16, p17, p18, p19)                                                    \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_,                                 \
            std::decay_t<decltype(p11)> _p11_,                                 \
            std::decay_t<decltype(p12)> _p12_,                                 \
            std::decay_t<decltype(p13)> _p13_,                                 \
            std::decay_t<decltype(p14)> _p14_,                                 \
            std::decay_t<decltype(p15)> _p15_,                                 \
            std::decay_t<decltype(p16)> _p16_,                                 \
            std::decay_t<decltype(p17)> _p17_,                                 \
            std::decay_t<decltype(p18)> _p18_,                                 \
            std::decay_t<decltype(p19)> _p19_) [[hc]] {                        \
            kernel_name(                                                       \
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,  \
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_,  \
               _p18_, _p19_);                                                  \
        }
    #define make_lambda_wrapper20(                                             \
        kernel_name,                                                           \
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,  \
        p16, p17, p18)                                                         \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_,                                 \
            std::decay_t<decltype(p11)> _p11_,                                 \
            std::decay_t<decltype(p12)> _p12_,                                 \
            std::decay_t<decltype(p13)> _p13_,                                 \
            std::decay_t<decltype(p14)> _p14_,                                 \
            std::decay_t<decltype(p15)> _p15_,                                 \
            std::decay_t<decltype(p16)> _p16_,                                 \
            std::decay_t<decltype(p17)> _p17_,                                 \
            std::decay_t<decltype(p18)> _p18_) [[hc]] {                        \
            kernel_name(                                                       \
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,  \
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_,  \
               _p18_);                                                         \
        }
    #define make_lambda_wrapper19(                                             \
        kernel_name,                                                           \
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,  \
        p16, p17)                                                              \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_,                                 \
            std::decay_t<decltype(p11)> _p11_,                                 \
            std::decay_t<decltype(p12)> _p12_,                                 \
            std::decay_t<decltype(p13)> _p13_,                                 \
            std::decay_t<decltype(p14)> _p14_,                                 \
            std::decay_t<decltype(p15)> _p15_,                                 \
            std::decay_t<decltype(p16)> _p16_,                                 \
            std::decay_t<decltype(p17)> _p17_) [[hc]] {                        \
            kernel_name(                                                       \
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,  \
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_); \
        }
    #define make_lambda_wrapper18(                                             \
        kernel_name,                                                           \
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,  \
        p16)                                                                   \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_,                                 \
            std::decay_t<decltype(p11)> _p11_,                                 \
            std::decay_t<decltype(p12)> _p12_,                                 \
            std::decay_t<decltype(p13)> _p13_,                                 \
            std::decay_t<decltype(p14)> _p14_,                                 \
            std::decay_t<decltype(p15)> _p15_,                                 \
            std::decay_t<decltype(p16)> _p16_) [[hc]] {                        \
            kernel_name(                                                       \
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,  \
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_);        \
        }
    #define make_lambda_wrapper17(                                             \
        kernel_name,                                                           \
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15)  \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_,                                 \
            std::decay_t<decltype(p11)> _p11_,                                 \
            std::decay_t<decltype(p12)> _p12_,                                 \
            std::decay_t<decltype(p13)> _p13_,                                 \
            std::decay_t<decltype(p14)> _p14_,                                 \
            std::decay_t<decltype(p15)> _p15_) [[hc]] {                        \
            kernel_name(                                                       \
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,  \
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_);               \
        }
    #define make_lambda_wrapper16(                                             \
        kernel_name,                                                           \
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14)       \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_,                                 \
            std::decay_t<decltype(p11)> _p11_,                                 \
            std::decay_t<decltype(p12)> _p12_,                                 \
            std::decay_t<decltype(p13)> _p13_,                                 \
            std::decay_t<decltype(p14)> _p14_) [[hc]] {                        \
            kernel_name(                                                       \
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,  \
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_);                      \
        }
    #define make_lambda_wrapper15(                                             \
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13)\
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_,                                 \
            std::decay_t<decltype(p11)> _p11_,                                 \
            std::decay_t<decltype(p12)> _p12_,                                 \
            std::decay_t<decltype(p13)> _p13_) [[hc]] {                        \
            kernel_name(                                                       \
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,  \
                _p9_, _p10_, _p11_, _p12_, _p13_);                             \
        }
    #define make_lambda_wrapper14(                                             \
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)    \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_,                                 \
            std::decay_t<decltype(p11)> _p11_,                                 \
            std::decay_t<decltype(p12)> _p12_) [[hc]] {                        \
            kernel_name(                                                       \
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,  \
                _p9_, _p10_, _p11_, _p12_);                                    \
        }
    #define make_lambda_wrapper13(                                             \
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)         \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_,                                 \
            std::decay_t<decltype(p11)> _p11_) [[hc]] {                        \
            kernel_name(                                                       \
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,  \
                _p9_, _p10_, _p11_);                                           \
        }
    #define make_lambda_wrapper12(                                             \
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)              \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_,                                   \
            std::decay_t<decltype(p9)> _p9_,                                   \
            std::decay_t<decltype(p10)> _p10_) [[hc]] {                        \
            kernel_name(                                                       \
                 _p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_,   \
                _p10_);                                                        \
        }
    #define make_lambda_wrapper11(                                             \
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9)                   \
        [](                                                                    \
           std::decay_t<decltype(p0)> _p0_,                                    \
           std::decay_t<decltype(p1)> _p1_,                                    \
           std::decay_t<decltype(p2)> _p2_,                                    \
           std::decay_t<decltype(p3)> _p3_,                                    \
           std::decay_t<decltype(p4)> _p4_,                                    \
           std::decay_t<decltype(p5)> _p5_,                                    \
           std::decay_t<decltype(p6)> _p6_,                                    \
           std::decay_t<decltype(p7)> _p7_,                                    \
           std::decay_t<decltype(p8)> _p8_,                                    \
           std::decay_t<decltype(p9)> _p9_) [[hc]] {                           \
            kernel_name(                                                       \
                _p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_);   \
        }
    #define make_lambda_wrapper10(                                             \
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8)                       \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_,                                   \
            std::decay_t<decltype(p8)> _p8_) [[hc]] {                          \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_); \
        }
    #define make_lambda_wrapper9(kernel_name, p0, p1, p2, p3, p4, p5, p6, p7)  \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_,                                   \
            std::decay_t<decltype(p7)> _p7_) [[hc]] {                          \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_);       \
        }
    #define make_lambda_wrapper8(kernel_name, p0, p1, p2, p3, p4, p5, p6)      \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_,                                   \
            std::decay_t<decltype(p6)> _p6_) [[hc]] {                          \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_);             \
        }
    #define make_lambda_wrapper7(kernel_name, p0, p1, p2, p3, p4, p5)          \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_,                                   \
            std::decay_t<decltype(p5)> _p5_) [[hc]] {                          \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_);                   \
        }
    #define make_lambda_wrapper6(kernel_name, p0, p1, p2, p3, p4)              \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_,                                   \
            std::decay_t<decltype(p4)> _p4_) [[hc]] {                          \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_);                         \
        }
    #define make_lambda_wrapper5(kernel_name, p0, p1, p2, p3)                  \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_,                                   \
            std::decay_t<decltype(p3)> _p3_) [[hc]] {                          \
            kernel_name(_p0_, _p1_, _p2_, _p3_);                               \
        }
    #define make_lambda_wrapper4(kernel_name, p0, p1, p2)                      \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_,                                   \
            std::decay_t<decltype(p2)> _p2_) [[hc]] {                          \
            kernel_name(_p0_, _p1_, _p2_);                                     \
        }
    #define make_lambda_wrapper3(kernel_name, p0, p1)                          \
        [](                                                                    \
            std::decay_t<decltype(p0)> _p0_,                                   \
            std::decay_t<decltype(p1)> _p1_) [[hc]] {                          \
            kernel_name(_p0_, _p1_);                                           \
        }
    #define make_lambda_wrapper2(kernel_name, p0)                              \
        [](std::decay_t<decltype(p0)> _p0_) [[hc]] {                           \
            kernel_name(_p0_);                                                 \
        }
    #define make_lambda_wrapper1(kernel_name)                                  \
        []() [[hc]] { kernel_name(lp); }

    #define make_lambda_wrapper(...)                                           \
        overload_macro(make_lambda_wrapper, __VA_ARGS__)

    #define hipLaunchKernelV3(                                                 \
        kernel_name,                                                           \
        num_blocks,                                                            \
        dim_blocks,                                                            \
        group_mem_bytes,                                                       \
        stream,                                                                \
        ...)                                                                   \
    {                                                                          \
        glo_tests::grid_launch(                                                \
            num_blocks,                                                        \
            dim_blocks,                                                        \
            group_mem_bytes,                                                   \
            stream,                                                            \
            make_lambda_wrapper(kernel_name, __VA_ARGS__),                     \
            ##__VA_ARGS__);                                                    \
    }

    #define hipLaunchKernel(                                                 \
        kernel_name,                                                           \
        num_blocks,                                                            \
        dim_blocks,                                                            \
        group_mem_bytes,                                                       \
        stream,                                                                \
        ...)                                                                   \
    {                                                                          \
        hipLaunchKernelV3(                                                     \
            kernel_name,                                                     \
            num_blocks,                                                        \
            dim_blocks,                                                        \
            group_mem_bytes,                                                   \
            stream,                                                            \
            hipLaunchParm{},                                                   \
            ##__VA_ARGS__);                                                    \
    }
}

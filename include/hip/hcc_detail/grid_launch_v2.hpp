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

namespace hip_impl
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

        hc::parallel_for_each(*av, d, [=](const hc::tiled_index<3>& idx) [[hc]] {
            k(args...);
        });
    }

    template<FunctionalProcedure K, typename... Ts>
        requires(Domain<K> == {hipLaunchParm, Ts...})
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

    // TODO: these are temporary, they need to be completely removed once we
    //       enable C++14 support and can have proper generic, variadic lambdas.
    #define make_kernel_lambda_hip_26(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,\
        p16, p17, p18, p19, p20, p21, p22, p23, p24)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_,\
           const std::decay_t<decltype(p16)>& _p16_,\
           const std::decay_t<decltype(p17)>& _p17_,\
           const std::decay_t<decltype(p18)>& _p18_,\
           const std::decay_t<decltype(p19)>& _p19_,\
           const std::decay_t<decltype(p20)>& _p20_,\
           const std::decay_t<decltype(p21)>& _p21_,\
           const std::decay_t<decltype(p22)>& _p22_,\
           const std::decay_t<decltype(p23)>& _p23_,\
           const std::decay_t<decltype(p24)>& _p24_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_,\
                _p18_, _p19_, _p20_, _p21_, _p22_, _p23_, _p24_);\
        }
    #define make_kernel_lambda_hip_25(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,\
        p16, p17, p18, p19, p20, p21, p22, p23)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_,\
           const std::decay_t<decltype(p16)>& _p16_,\
           const std::decay_t<decltype(p17)>& _p17_,\
           const std::decay_t<decltype(p18)>& _p18_,\
           const std::decay_t<decltype(p19)>& _p19_,\
           const std::decay_t<decltype(p20)>& _p20_,\
           const std::decay_t<decltype(p21)>& _p21_,\
           const std::decay_t<decltype(p22)>& _p22_,\
           const std::decay_t<decltype(p23)>& _p23_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_,\
                _p18_, _p19_, _p20_, _p21_, _p22_, _p23_);\
        }
    #define make_kernel_lambda_hip_24(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,\
        p16, p17, p18, p19, p20, p21, p22)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_,\
           const std::decay_t<decltype(p16)>& _p16_,\
           const std::decay_t<decltype(p17)>& _p17_,\
           const std::decay_t<decltype(p18)>& _p18_,\
           const std::decay_t<decltype(p19)>& _p19_,\
           const std::decay_t<decltype(p20)>& _p20_,\
           const std::decay_t<decltype(p21)>& _p21_,\
           const std::decay_t<decltype(p22)>& _p22_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_,\
                _p18_, _p19_, _p20_, _p21_, _p22_);\
        }
    #define make_kernel_lambda_hip_23(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,\
        p16, p17, p18, p19, p20, p21)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_,\
           const std::decay_t<decltype(p16)>& _p16_,\
           const std::decay_t<decltype(p17)>& _p17_,\
           const std::decay_t<decltype(p18)>& _p18_,\
           const std::decay_t<decltype(p19)>& _p19_,\
           const std::decay_t<decltype(p20)>& _p20_,\
           const std::decay_t<decltype(p21)>& _p21_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_,\
                _p18_, _p19_, _p20_, _p21_);\
        }
    #define make_kernel_lambda_hip_22(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,\
        p16, p17, p18, p19, p20)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_,\
           const std::decay_t<decltype(p16)>& _p16_,\
           const std::decay_t<decltype(p17)>& _p17_,\
           const std::decay_t<decltype(p18)>& _p18_,\
           const std::decay_t<decltype(p19)>& _p19_,\
           const std::decay_t<decltype(p20)>& _p20_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_,\
                _p18_, _p19_, _p20_);\
        }
    #define make_kernel_lambda_hip_21(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,\
        p16, p17, p18, p19)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_,\
           const std::decay_t<decltype(p16)>& _p16_,\
           const std::decay_t<decltype(p17)>& _p17_,\
           const std::decay_t<decltype(p18)>& _p18_,\
           const std::decay_t<decltype(p19)>& _p19_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_,\
                _p18_, _p19_);\
        }
    #define make_kernel_lambda_hip_20(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,\
        p16, p17, p18)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_,\
           const std::decay_t<decltype(p16)>& _p16_,\
           const std::decay_t<decltype(p17)>& _p17_,\
           const std::decay_t<decltype(p18)>& _p18_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_,\
                _p18_);\
        }
    #define make_kernel_lambda_hip_19(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,\
        p16, p17)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_,\
           const std::decay_t<decltype(p16)>& _p16_,\
           const std::decay_t<decltype(p17)>& _p17_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_, _p17_);\
        }
    #define make_kernel_lambda_hip_18(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,\
        p16)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_,\
           const std::decay_t<decltype(p16)>& _p16_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_, _p16_);\
        }
    #define make_kernel_lambda_hip_17(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_,\
           const std::decay_t<decltype(p15)>& _p15_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_, _p15_);\
        }
    #define make_kernel_lambda_hip_16(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_,\
           const std::decay_t<decltype(p14)>& _p14_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_, _p14_);\
        }
    #define make_kernel_lambda_hip_15(\
        kernel_name,\
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_,\
           const std::decay_t<decltype(p13)>& _p13_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_, _p13_);\
        }
    #define make_kernel_lambda_hip_14(\
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_,\
           const std::decay_t<decltype(p12)>& _p12_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_, _p12_);\
        }
    #define make_kernel_lambda_hip_13(\
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_,\
           const std::decay_t<decltype(p11)>& _p11_) [[hc]] {\
            kernel_name(\
                _p0_,  _p1_,  _p2_,  _p3_,  _p4_,  _p5_,  _p6_,  _p7_,  _p8_,\
                _p9_, _p10_, _p11_);\
        }
    #define make_kernel_lambda_hip_12(\
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_,\
           const std::decay_t<decltype(p10)>& _p10_) [[hc]] {\
            kernel_name(\
                 _p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_,\
                _p10_);\
        }
    #define make_kernel_lambda_hip_11(\
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_,\
           const std::decay_t<decltype(p9)>& _p9_) [[hc]] {\
            kernel_name(\
                _p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_);\
        }
    #define make_kernel_lambda_hip_10(\
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_,\
           const std::decay_t<decltype(p8)>& _p8_) [[hc]] {\
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_);\
        }
    #define make_kernel_lambda_hip_9(\
        kernel_name, p0, p1, p2, p3, p4, p5, p6, p7)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_,\
           const std::decay_t<decltype(p7)>& _p7_) [[hc]] {\
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_);\
        }
    #define make_kernel_lambda_hip_8(kernel_name, p0, p1, p2, p3, p4, p5, p6)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_,\
           const std::decay_t<decltype(p6)>& _p6_) [[hc]] {\
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_);\
        }
    #define make_kernel_lambda_hip_7(kernel_name, p0, p1, p2, p3, p4, p5)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_,\
           const std::decay_t<decltype(p5)>& _p5_) [[hc]] {\
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_);\
        }
    #define make_kernel_lambda_hip_6(kernel_name, p0, p1, p2, p3, p4)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_,\
           const std::decay_t<decltype(p4)>& _p4_) [[hc]] {\
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_);\
        }
    #define make_kernel_lambda_hip_5(kernel_name, p0, p1, p2, p3)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_,\
           const std::decay_t<decltype(p3)>& _p3_) [[hc]] {\
            kernel_name(_p0_, _p1_, _p2_, _p3_);\
        }
    #define make_kernel_lambda_hip_4(kernel_name, p0, p1, p2)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_,\
           const std::decay_t<decltype(p2)>& _p2_) [[hc]] {\
            kernel_name(_p0_, _p1_, _p2_);\
        }
    #define make_kernel_lambda_hip_3(kernel_name, p0, p1)\
        [](const std::decay_t<decltype(p0)>& _p0_,\
           const std::decay_t<decltype(p1)>& _p1_) [[hc]] {\
            kernel_name(_p0_, _p1_);\
        }
    #define make_kernel_lambda_hip_2(kernel_name, p0)\
        [](const std::decay_t<decltype(p0)>& _p0_) [[hc]] {\
            kernel_name(_p0_);\
        }
    #define make_kernel_lambda_hip_1(kernel_name)\
        []() [[hc]] { kernel_name(lp); }

    #define make_kernel_lambda_hip_(...)\
        overload_macro_hip_(make_kernel_lambda_hip_, __VA_ARGS__)

    #define hipLaunchKernelGGL(\
        kernel_name,\
        num_blocks,\
        dim_blocks,\
        group_mem_bytes,\
        stream,\
        ...)\
    {\
        hip_impl::grid_launch(\
            num_blocks,\
            dim_blocks,\
            group_mem_bytes,\
            stream,\
            make_kernel_lambda_hip_(kernel_name, __VA_ARGS__),\
            ##__VA_ARGS__);\
    }

    #define hipLaunchKernel(\
        kernel_name,\
        num_blocks,\
        dim_blocks,\
        group_mem_bytes,\
        stream,\
        ...)\
    {\
        hipLaunchKernelGGL(\
            kernel_name,\
            num_blocks,\
            dim_blocks,\
            group_mem_bytes,\
            stream,\
            hipLaunchParm{},\
            ##__VA_ARGS__);\
    }
}

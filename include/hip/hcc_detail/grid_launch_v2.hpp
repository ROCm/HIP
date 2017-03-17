//
// Created by alexv on 25/10/16.
//
#pragma once

#include "concepts.hpp"
#include "helpers.hpp"

#include "hc.hpp"
#include "hcc_acc.h"

//#include <hip/hcc.h>
//#include <hip/hip_runtime.h>

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
        grid_launch(
            std::move(num_blocks),
            std::move(dim_blocks),
            group_mem_bytes,
            std::move(stream),
            [](decltype(std::decay_t<Ts>(args))... f_args) [[hc]] {
                k(f_args...);
            },
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

    template<typename, typename = void> struct Wrapper;

    template<FunctionalProcedure K>
    struct Wrapper<K, std::enable_if_t<!std::is_function<K>::value>> {
        template<typename... Ts>
            requires(Domain<K> == {Ts...})
        void operator()(Ts&&... args) const
        {
            grid_launch_impl(
                is_new_grid_launch_t<K, Ts...>{},
                std::forward<Ts>(args)...);
        }
    };

    template<FunctionalProcedure K>
    struct Wrapper<K, std::enable_if_t<std::is_function<K>::value>> {
        template<typename... Ts>
        void operator()(Ts&&...) const {}
    };
#warning "GGL hipLaunchKernel defined"
    #define hipLaunchKernel(                                                 \
        kernel_name,                                                           \
        num_blocks,                                                            \
        dim_blocks,                                                            \
        group_mem_bytes,                                                       \
        stream,                                                                \
        ...)                                                                   \
    {                                                                          \
        using F = decltype(kernel_name);                                       \
        if (!std::is_function<F>::value) {                                     \
            glo_tests::Wrapper<F>{}(                                           \
                num_blocks,                                                    \
                dim_blocks,                                                    \
                group_mem_bytes,                                               \
                stream,                                                        \
                kernel_name,                                                   \
                ##__VA_ARGS__);                                                \
        }                                                                      \
        else {                                                                 \
            glo_tests::grid_launch<F, &kernel_name>(                           \
                num_blocks,                                                    \
                dim_blocks,                                                    \
                group_mem_bytes,                                               \
                stream,                                                        \
                ##__VA_ARGS__);                                                \
        }                                                                      \
    }
}

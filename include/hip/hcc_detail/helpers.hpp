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

#pragma once

#include <type_traits> // For std::conditional, std::decay, std::enable_if,
                       // std::false_type, std result_of and std::true_type.
#include <utility>     // For std::declval.

namespace std
{   // TODO: these should be removed as soon as possible.
    #if (__cplusplus < 201406L)
        template<typename...>
        using void_t = void;

        #if (__cplusplus < 201402L)
            template<bool cond, typename T = void>
            using enable_if_t = typename enable_if<cond, T>::type;
            template<bool cond, typename T, typename U>
            using conditional_t = typename conditional<cond, T, U>::type;
            template<typename T>
            using decay_t = typename decay<T>::type;
            template<FunctionalProcedure F, typename... Ts>
            using result_of_t = typename result_of<F(Ts...)>::type;
            template<typename T>
            using remove_reference_t = typename remove_reference<T>::type;
            template<
                FunctionalProcedure F,
                unsigned int n = 0u,
                typename = void>
            struct is_callable_impl : is_callable_impl<F, n + 1u> {};

            // Pointer to member function, call through non-pointer.
            template<FunctionalProcedure F, typename C, typename... Ts>
            struct is_callable_impl<
                F(C, Ts...),
                0u,
                void_t<decltype((declval<C>().*declval<F>())(declval<Ts>()...))>
            > : true_type {
            };

            // Pointer to member function, call through pointer.
            template<FunctionalProcedure F, typename C, typename... Ts>
            struct is_callable_impl<
                F(C, Ts...),
                1u,
                void_t<decltype(((*declval<C>()).*declval<F>())(declval<Ts>()...))>
            > : std::true_type {
            };

            // Pointer to member data, call through non-pointer, no args.
            template<FunctionalProcedure F, typename C>
            struct is_callable_impl<
                F(C),
                2u,
                void_t<decltype(declval<C>().*declval<F>())>
            > : true_type {
            };

            // Pointer to member data, call through pointer, no args.
            template<FunctionalProcedure F, typename C>
            struct is_callable_impl<
                F(C),
                3u,
                void_t<decltype(*declval<C>().*declval<F>())>
            > : true_type {
            };

            // General call, n args.
            template<FunctionalProcedure F, typename... Ts>
            struct is_callable_impl<
                F(Ts...),
                4u,
                void_t<decltype(declval<F>()(declval<Ts>()...))>
            > : true_type {
            };

            // Not callable.
            template<FunctionalProcedure F>
            struct is_callable_impl<F, 5u> : false_type {};

            template<typename Call>
            struct is_callable : is_callable_impl<Call> {};
        #else
            template<typename, typename = void>
            struct is_callable_impl : false_type {};

            template<FunctionalProcedure F, typename... Ts>
            struct is_callable_impl<
                F(Ts...),
                void_t<result_of_t<F(Ts...)>>> : true_type {};

            template<typename F>
            struct is_callable : is_callable_impl<F> {};
        #endif
        template<typename...>
        struct disjunction : false_type {};
        template<typename B1>
        struct disjunction<B1> : B1 {};
        template<typename B1, typename... Bs>
        struct disjunction<B1, Bs...>
            : conditional_t<B1{} == true, B1, disjunction<Bs...>>
        {};
    #endif
}

namespace // Only for documentation, macros ignore namespaces.
{
    #define count_macro_args_impl_hip_(\
         _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15,\
         _16, _17, _18, _19, _20, _21,  _n, ...) _n
    #define count_macro_args_hip_(...)\
        count_macro_args_impl_hip_(\
            , ##__VA_ARGS__, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,\
            8, 7, 6, 5, 4, 3, 2, 1, 0)

    #define overloaded_macro_expand_hip_(macro, arg_cnt) macro##arg_cnt
    #define overload_macro_impl_hip_(macro, arg_cnt)\
        overloaded_macro_expand_hip_(macro, arg_cnt)
    #define overload_macro_hip_(macro, ...)\
        overload_macro_impl_hip_(macro, count_macro_args_hip_(__VA_ARGS__))\
            (__VA_ARGS__)
}

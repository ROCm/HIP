//
// Created by alexv on 08/11/16.
//
#pragma once

#include <type_traits> // For std::conditional, std::decay, std::enable_if,
                       // std::false_type, std result_of and std::true_type.
#include <utility>     // For std::declval.

namespace std
{
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
    #endif
}

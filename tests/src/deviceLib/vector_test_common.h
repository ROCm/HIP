/*
Copyright (c) 2015-2017 Advanced Micro Devices, Inc. All rights reserved.

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

#include <type_traits>

template<bool b, typename T = void>
using Enable_if_t = typename std::enable_if<b, T>::type;

__host__ __device__
std::false_type is_vec4(...);
__host__ __device__
std::false_type is_vec3(...);
__host__ __device__
std::false_type is_vec2(...);
__host__ __device__
std::false_type is_vec1(...);

template<typename T>
__host__ __device__
auto is_vec4(const T&) -> decltype(std::declval<T>().xyzw, std::true_type{});
template<
    typename T, Enable_if_t<decltype(!is_vec4(std::declval<T>())){}>* = nullptr>
__host__ __device__
auto is_vec3(const T&) -> decltype(std::declval<T>().xyz, std::true_type{});
template<
    typename T,
    Enable_if_t<
        !decltype(is_vec4(std::declval<T>())){} &&
        !decltype(is_vec3(std::declval<T>())){}>* = nullptr>
__host__ __device__
auto is_vec2(const T&) -> decltype(std::declval<T>().xy, std::true_type{});
template<
    typename T,
    Enable_if_t<
        !decltype(is_vec4(std::declval<T>())){} &&
        !decltype(is_vec3(std::declval<T>())){} &&
        !decltype(is_vec2(std::declval<T>())){}>* = nullptr>
__host__ __device__
auto is_vec1(const T&) -> decltype(std::declval<T>().x, std::true_type{});

template<typename T, int dimension>
__host__ __device__
constexpr
bool is_vec() {
    return (dimension == 1) ? decltype(is_vec1(std::declval<T>())){} :
        ((dimension == 2) ? decltype(is_vec2(std::declval<T>())){} :
        ((dimension == 3) ? decltype(is_vec3(std::declval<T>())){} :
            decltype(is_vec4(std::declval<T>())){}));
}
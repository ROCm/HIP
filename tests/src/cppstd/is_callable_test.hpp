/*
Copyright (c) 2017 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hcc_detail/helpers.hpp>
#include <memory>

using hip_impl::is_callable;

template<int N>
struct callable_rank : callable_rank<N-1>
{};

template<>
struct callable_rank<0>
{};

struct test1
{
    struct is_callable_class
    {
        void operator()(int) const
        {
        }
    };
    struct callable_test_param {};

    void is_callable_function(int)
    {
    }

    struct is_callable_rank_class
    {
        void operator()(int, callable_rank<3>) const
        {
        }

        void operator()(int, callable_rank<4>) const
        {
        }
    };

    static_assert(is_callable<is_callable_class(int)>::value, "Not callable");
    static_assert(is_callable<is_callable_class(long)>::value, "Not callable");
    static_assert(is_callable<is_callable_class(double)>::value, "Not callable");
    static_assert(is_callable<is_callable_class(const int&)>::value, "Not callable");
    static_assert(is_callable<is_callable_class(const long&)>::value, "Not callable");
    static_assert(is_callable<is_callable_class(const double&)>::value, "Not callable");
    static_assert(not is_callable<is_callable_class(callable_test_param)>::value, "callable failed");
    static_assert(not is_callable<is_callable_class()>::value, "callable failed");
    static_assert(not is_callable<is_callable_class(int, int)>::value, "callable failed");

    typedef void (*is_callable_function_pointer)(int);
    static_assert(is_callable<is_callable_function_pointer(int)>::value, "Not callable");
    static_assert(is_callable<is_callable_function_pointer(long)>::value, "Not callable");
    static_assert(is_callable<is_callable_function_pointer(double)>::value, "Not callable");
    static_assert(is_callable<is_callable_function_pointer(const int&)>::value, "Not callable");
    static_assert(is_callable<is_callable_function_pointer(const long&)>::value, "Not callable");
    static_assert(is_callable<is_callable_function_pointer(const double&)>::value, "Not callable");
    static_assert(not is_callable<is_callable_function_pointer(callable_test_param)>::value, "callable failed");
    static_assert(not is_callable<is_callable_function_pointer()>::value, "callable failed");
    static_assert(not is_callable<is_callable_function_pointer(int, int)>::value, "callable failed");

    static_assert(is_callable<is_callable_rank_class(int, callable_rank<3>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(long, callable_rank<3>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(double, callable_rank<3>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const int&, callable_rank<3>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const long&, callable_rank<3>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const double&, callable_rank<3>)>::value, "Not callable");

    static_assert(is_callable<is_callable_rank_class(int, callable_rank<4>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(long, callable_rank<4>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(double, callable_rank<4>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const int&, callable_rank<4>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const long&, callable_rank<4>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const double&, callable_rank<4>)>::value, "Not callable");

    static_assert(is_callable<is_callable_rank_class(int, callable_rank<5>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(long, callable_rank<5>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(double, callable_rank<5>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const int&, callable_rank<5>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const long&, callable_rank<5>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const double&, callable_rank<5>)>::value, "Not callable");

    static_assert(is_callable<is_callable_rank_class(int, callable_rank<6>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(long, callable_rank<6>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(double, callable_rank<6>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const int&, callable_rank<6>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const long&, callable_rank<6>)>::value, "Not callable");
    static_assert(is_callable<is_callable_rank_class(const double&, callable_rank<6>)>::value, "Not callable");

    static_assert(not is_callable<is_callable_rank_class(int, callable_rank<1>)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(long, callable_rank<1>)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(double, callable_rank<1>)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(const int&, callable_rank<1>)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(const long&, callable_rank<1>)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(const double&, callable_rank<1>)>::value, "callable failed");

    static_assert(not is_callable<is_callable_rank_class(callable_test_param, callable_test_param)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(callable_rank<3>, callable_test_param)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(callable_rank<4>, callable_test_param)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(callable_test_param, callable_rank<3>)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(callable_test_param, callable_rank<4>)>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class()>::value, "callable failed");
    static_assert(not is_callable<is_callable_rank_class(int, int)>::value, "callable failed");
};

struct test2
{
    typedef int(callable_rank<0>::*fn)(int);

    static_assert(is_callable<fn(callable_rank<0>&, int)>::value, "Failed");
    static_assert(is_callable<fn(callable_rank<1>&, int)>::value, "Failed");
    static_assert(not is_callable<fn(callable_rank<0>&)>::value, "Failed");
    static_assert(not is_callable<fn(callable_rank<0> const&, int)>::value, "Failed");
};

struct test3
{
    typedef int(callable_rank<0>::*fn)(int);

    typedef callable_rank<0>* T;
    typedef callable_rank<1>* DT;
    typedef const callable_rank<0>* CT;
    typedef std::unique_ptr<callable_rank<0>> ST;

    static_assert(is_callable<fn(T&, int)>::value, "Failed");
    static_assert(is_callable<fn(DT&, int)>::value, "Failed");
    static_assert(is_callable<fn(const T&, int)>::value, "Failed");
    static_assert(is_callable<fn(T&&, int)>::value, "Failed");
    static_assert(is_callable<fn(ST, int)>::value, "Failed");
    static_assert(not is_callable<fn(CT&, int)>::value, "Failed");

};

struct test4
{
    typedef int(callable_rank<0>::*fn);

    static_assert(not is_callable<fn()>::value, "Failed");
};

struct test5
{
    typedef int(callable_rank<0>::*fn);

    static_assert(is_callable<fn(callable_rank<0>&)>::value, "Failed");
    static_assert(is_callable<fn(callable_rank<0>&&)>::value, "Failed");
    static_assert(is_callable<fn(const callable_rank<0>&)>::value, "Failed");
    static_assert(is_callable<fn(callable_rank<1>&)>::value, "Failed");
};

struct test6
{
    typedef int(callable_rank<0>::*fn);

    typedef callable_rank<0>* T;
    typedef callable_rank<1>* DT;
    typedef const callable_rank<0>* CT;
    typedef std::unique_ptr<callable_rank<0>> ST;

    static_assert(is_callable<fn(T&)>::value, "Failed");
    static_assert(is_callable<fn(DT&)>::value, "Failed");
    static_assert(is_callable<fn(const T&)>::value, "Failed");
    static_assert(is_callable<fn(T&&)>::value, "Failed");
    static_assert(is_callable<fn(ST)>::value, "Failed");
    static_assert(is_callable<fn(CT&)>::value, "Failed");

};

struct test7
{
    typedef void(*fp)(callable_rank<0>&, int);

    static_assert(is_callable<fp(callable_rank<0>&, int)>::value, "Failed");
    static_assert(is_callable<fp(callable_rank<1>&, int)>::value, "Failed");
    static_assert(not is_callable<fp(const callable_rank<0>&, int)>::value, "Failed");
    static_assert(not is_callable<fp()>::value, "Failed");
    static_assert(not is_callable<fp(callable_rank<0>&)>::value, "Failed");
};

struct test8
{
    typedef void(&fp)(callable_rank<0>&, int);

    static_assert(is_callable<fp(callable_rank<0>&, int)>::value, "Failed");
    static_assert(is_callable<fp(callable_rank<1>&, int)>::value, "Failed");
    static_assert(not is_callable<fp(const callable_rank<0>&, int)>::value, "Failed");
    static_assert(not is_callable<fp()>::value, "Failed");
    static_assert(not is_callable<fp(callable_rank<0>&)>::value, "Failed");
};

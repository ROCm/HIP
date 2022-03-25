#pragma once

#include <hip_test_common.hh>
#include <map>

#ifndef RTC_ENABLED

__global__ void Set(int* Ad, int val);
#include "vectorADD.inl"

#else

/**
 * @brief Wrapper Macros that create a string representation of the kernel name.
 * In the case of kernel templates, an empty variadic template is used to ensure compatibility with the
 * launchKernel template when RTC is not enabled. If the kernel is inside a namespace, use the "_NS"
 * version of the Macro.
 */
#define FUNCTION_WRAPPER(param) static std::string param(#param)
#define TEMPLATE_WRAPPER(param) template <typename...> std::string param(#param)
#define FUNCTION_WRAPPER_NS(param, namespace) static std::string param(#namespace "::" #param)
#define TEMPLATE_WRAPPER_NS(param, namespace)                                                      \
  template <typename...> std::string param(#namespace "::" #param)

FUNCTION_WRAPPER(Set);

namespace HipTest {
TEMPLATE_WRAPPER_NS(vectorADD, HipTest);
}

#endif

// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// Checks that HIP header file is included after #pragma once,
// which goes before include guard controlling macro.
// CHECK: #pragma once
// CHECK-NEXT: #include <hip/hip_runtime.h>
#pragma once
#ifndef HEADERS_TEST_10_H
// CHECK: #ifndef HEADERS_TEST_10_H
// CHECK-NOT: #include <hip/hip_runtime.h>
#define HEADERS_TEST_10_H
#include <stdio.h>
static int counter = 0;
#endif // HEADERS_TEST_10_H

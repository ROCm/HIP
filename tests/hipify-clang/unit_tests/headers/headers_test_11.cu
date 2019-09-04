// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

// Checks that HIP header file is included after include guard controlling macro,
// which goes before #pragma once.
// CHECK: #ifndef HEADERS_TEST_10_H
// CHECK-NEXT: #include <hip/hip_runtime.h>
#ifndef HEADERS_TEST_10_H
// CHECK: #pragma once
#pragma once
// CHECK-NOT: #include <hip/hip_runtime.h>
#define HEADERS_TEST_10_H
#include <stdio.h>
static int counter = 0;
#endif // HEADERS_TEST_10_H

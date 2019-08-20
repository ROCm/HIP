file(GLOB files *.cpp)
hostcall_add_tests("${files}")

set_tests_properties(hostcall-basic-print-things PROPERTIES
  PASS_REGULAR_EXPRESSION "(42 -> 0)\r?\nkernel: kernel1; tid: 0")

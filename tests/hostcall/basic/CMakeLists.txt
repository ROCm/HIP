file(GLOB files *.cpp)
foreach(file ${files})
    hostcall_add_test(basic ${file})
endforeach()

set_tests_properties(hostcall-basic-print-things PROPERTIES
  PASS_REGULAR_EXPRESSION "(42 -> 0)\r?\nkernel: kernel1; tid: 0")

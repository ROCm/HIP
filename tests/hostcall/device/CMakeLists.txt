file(GLOB files *.cpp)
foreach(file ${files})
    hostcall_add_test(testname device ${file})
endforeach()

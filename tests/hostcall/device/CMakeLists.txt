file(GLOB files *.cpp)
foreach(file ${files})
    hostcall_add_test(device ${file})
endforeach()

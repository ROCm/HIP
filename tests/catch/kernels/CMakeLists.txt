if(NOT DEFINED RTC_ENABLED)
    set(TEST_SRC
        Set.cpp
    )

    add_library(KERNELS EXCLUDE_FROM_ALL OBJECT ${TEST_SRC})
    target_compile_options(KERNELS PUBLIC -std=c++17)
endif()

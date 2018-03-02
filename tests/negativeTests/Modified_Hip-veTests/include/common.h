ifndef HIP_NEGATIVE_TESTS_INCLUDE_COMMON_H
#define HIP_NEGATIVE_TESTS_INCLUDE_COMMON_H

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>

#define CHECK_ERRORS(status) \
  if(status != hipSuccess) { \
    std::cout<<"Error: "<<hipGetErrorString(status)<<" at "<<__LINE__<<" in "<<__FILE__<<std::endl; \
  }

#endif

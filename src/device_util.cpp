#include"hip_runtime.h"
#include<hc.hpp>
#include<grid_launch.h>

__device__ unsigned int test__popc(unsigned int input)
{
    return hc::__popcount_u32_b32(input);
}


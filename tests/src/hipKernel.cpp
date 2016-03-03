#include"test_common.h"

__global__ void Empty(hipLaunchParm lp, int param){}

int main(){
hipLaunchKernel(HIP_KERNEL_NAME(Empty), dim3(1), dim3(1), 0, 0, 0);
hipDeviceSynchronize();
}

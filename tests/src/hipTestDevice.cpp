#include"test_common.h"
#include"hip_runtime.h"

__device__ void test(){
    test__popc(1);
}

__global__ void Test(hipLaunchParm lp, int a){
    test();
}

int main(){}

#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<hip/hcc_detail/hipComplex.h>

#define LEN 64
#define SIZE 64<<2

__global__  void getSqAbs(hipLaunchParm lp, float *A, float *B, float *C){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    C[tx] = hipCsqabsf(make_hipFloatComplex(A[tx], B[tx]));
}

int main(){
    float *A, *Ad, *B, *Bd, *C, *Cd;
    A = new float[LEN];
    B = new float[LEN];
    C = new float[LEN];
    for(uint32_t i=0;i<LEN;i++){
        A[i] = i*1.0f;
        B[i] = i*1.0f;
        C[i] = i*1.0f;
    }

    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Cd, SIZE);
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernel(getSqAbs, dim3(1), dim3(LEN), 0, 0, Ad, Bd, Cd);
    hipMemcpy(C, Cd, SIZE, hipMemcpyDeviceToHost);
    std::cout<<A[11]<<" "<<B[11]<<" "<<C[11]<<std::endl;
}

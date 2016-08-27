#include<iostream>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

#define LEN 1030
#define SIZE LEN << 2

__global__ void cpy(hipLaunchParm lp, uint32_t *Out, uint32_t *In, uint32_t *Vald)
{
    memcpy(Out, In, SIZE, Vald);
}

__global__ void set(hipLaunchParm lp, uint32_t *ptr, uint8_t val, size_t size)
{
    memset(ptr, val, size);
}

int main()
{
    uint32_t *A, *Ad, *B, *Bd;
    uint32_t *Val, *Vald;
    A = new uint32_t[LEN];
    B = new uint32_t[LEN];
    Val = new uint32_t;
    *Val = 0;
    for(int i=0;i<LEN;i++){
        A[i] = i *1.0f;
        B[i] = 0.0f;
    }
    hipMalloc((void**)&Ad, SIZE);
    hipMalloc((void**)&Bd, SIZE);
    hipMalloc((void**)&Vald, sizeof(uint32_t));
    hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice);
    hipLaunchKernel(cpy, dim3(1), dim3(LEN/4), 0, 0, Bd, Ad, Vald);
    hipLaunchKernel(set, dim3(1), dim3(LEN/4), 0, 0, Bd, 0x1, SIZE);
    hipMemcpy(B, Bd, SIZE, hipMemcpyDeviceToHost);
    hipMemcpy(Val, Vald, sizeof(uint32_t), hipMemcpyDeviceToHost);
    for(int i=LEN-16;i<LEN;i++){
        std::cout<<A[i]<<" "<<B[i]<<std::endl;
    }
    std::cout<<*Val<<std::endl;
}

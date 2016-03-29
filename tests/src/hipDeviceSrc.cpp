#include"hipDeviceHeader.h"

#define LEN 1024*1024
#define SIZE LEN * sizeof(int)

__global__ void deviceAdd(hipLaunchParm lp, int *Ad, int *Bd, int *Cd){
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    Cd[tx] = add(Ad[tx], Bd[tx]);
}

int main(){
    int *A, *B, *C, *Ad, *Bd, *Cd;
    A = new int[LEN];
    B = new int[LEN];
    C = new int[LEN];

    for(int i=0;i<LEN;i++){
        A[i] = 1;
        B[i] = 1;
    }

    HIPCHECK(hipMalloc((void**)&Ad, SIZE));
    HIPCHECK(hipMalloc((void**)&Bd, SIZE));
    HIPCHECK(hipMalloc((void**)&Cd, SIZE));

    HIPCHECK(hipMemcpy(Ad, A, SIZE, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(Bd, B, SIZE, hipMemcpyHostToDevice));
    hipLaunchKernel(HIP_KERNEL_NAME(deviceAdd), dim3(LEN/1024), dim3(1024), 0, 0, Ad, Bd, Cd);

}

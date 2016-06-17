#include"test_common.h"

#define SIZE 1024*1024

int main(){
    float *A, *Ad;
    HIPCHECK(hipHostMalloc((void**)&A,SIZE, hipHostMallocDefault));
    HIPCHECK(hipMalloc((void**)&Ad, SIZE));
    hipStream_t stream;
    HIPCHECK(hipStreamCreate(&stream));
    for(int i=0;i<SIZE;i++){
    HIPCHECK(hipMemcpyAsync(Ad, A, SIZE, hipMemcpyHostToDevice, stream));
    HIPCHECK(hipDeviceSynchronize());
    }
}

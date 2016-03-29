#include"hipDeviceHeader.h"

__host__ __device__ int add(int &a, int &b){
    return a+b;
}

#include"hipDeviceHeader.h"

__device__ int add(int &a, int &b){
    return a+b;
}

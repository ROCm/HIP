#include"gxxHipApi.h"

memManager::memManager(const memManager &obj)
{
    devPtr = obj.devPtr;
    hstPtr = obj.hstPtr;
    size = obj.size;
}

void memManager::H2D()
{
    hipMemcpy(devPtr, hstPtr, size, hipMemcpyHostToDevice);
}

void memManager::D2H()
{
    hipMemcpy(hstPtr, devPtr, size, hipMemcpyDeviceToHost);
}

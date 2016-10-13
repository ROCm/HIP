#include<hip/hip_runtime_api.h>
#include<iostream>
#include<assert.h>

int main()
{
    size_t heap;
    assert(hipSuccess == hipDeviceGetLimit(&heap, hipLimitMallocHeapSize));
    assert(heap == 4194304);
}

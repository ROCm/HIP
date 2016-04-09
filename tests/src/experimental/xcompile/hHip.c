#include "gHipApi.h"
#include "hip_runtime.h"

#define LEN 1024*1024
#define SIZE LEN * sizeof(float)

__global__ void Add(hipLaunchParm lp, float *Ad, float *Bd, float *Cd, size_t len)
{
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if(tx < len)
    {
        Cd[tx] = Ad[tx] + Bd[tx];
    }
}

int main()
{
    mem_manager *a, *b, *c;
    a = mem_manager_start(SIZE);
    b = mem_manager_start(SIZE);
    c = mem_manager_start(SIZE);
    a->malloc_hst(a);
    b->malloc_hst(b);
    c->malloc_hst(c);
    a->malloc_hip(a);
    b->malloc_hip(b);
    c->malloc_hip(c);
    memset_hst(a, 1.0f);
    memset_hst(b, 2.0f);
    a->h2d(a);
    b->h2d(b);
    hipLaunchKernel(HIP_KERNEL_NAME(Add), dim3(LEN/1024), dim3(1024), 0, 0, (float*)a->dev_ptr, (float*)b->dev_ptr, (float*)c->dev_ptr, LEN);
    c->d2h(c);
    assert(((float*)c->hst_ptr)[10] == 3.0f);


}

#include"gxxHipApi.h"
#include<vector>
#include"hip_runtime.h"

#define LEN 1024*1024
#define SIZE LEN * sizeof(float)

class memManager;

template<typename T>
__global__ void Add(hipLaunchParm lp, T* Ad, T* Bd, T* Cd, size_t Len)
{
    int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    if(tx < Len)
    {
        Cd[tx] = Ad[tx] + Bd[tx];
    }
}

int main()
{
    std::vector<class memManager> Vec(3);
    for(int i=0;i<Vec.size();i++){
        Vec[i] = memManager(SIZE);
    }

    for(int i=0;i<3;i++)
    {
        Vec[i].setHstPtr(new float[LEN]);
        Vec[i].memAlloc<float>();
    }

    for(int i=0;i<Vec.size()-1;i++)
    {
        Vec[i].hostMemSet((i+1)*1.0f);
        Vec[i].H2D();
    }

    hipLaunchKernel(HIP_KERNEL_NAME(Add), dim3(LEN/1024), dim3(1024), 0, 0, Vec[0].getDevPtr<float>(), Vec[1].getDevPtr<float>(), Vec[2].getDevPtr<float>(), LEN);

    Vec[2].D2H();
    assert(Vec[0].getHstPtr<float>()[10] + Vec[1].getHstPtr<float>()[10] == Vec[2].getHstPtr<float>()[10]);
}

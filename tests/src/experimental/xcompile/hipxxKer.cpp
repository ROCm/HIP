#include"hip_runtime.h"
#include"hip_runtime_api.h"
#include"gxxApi1.h"

#define len 1024*1024
#define size len * sizeof(float)

__global__ void Kern(hipLaunchParm lp, float *A)
{
	int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
	A[tx] += 1.0f;
}

int main()
{
	float *A, *Ad;
	for(int i=0;i<len;i++)
	{
		A[i] = 1.0f;
	}
	Ad = (float*)mallocHip(size);
	memcpyHipH2D(Ad, A, size);
	hipLaunchKernel(HIP_KERNEL_NAME(Kern), dim3(len/1024), dim3(1024), 0, 0, A);
	memcpyHipD2H(A, Ad, size);
	for(int i=0;i<len;i++)
	{
		assert(A[i] == 2.0f);
	}
}

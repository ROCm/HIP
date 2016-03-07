#include"test_common.h"

__global__ void Inc(hipLaunchParm lp, float *Ad){
int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
Ad[tx] = Ad[tx] + float(1);
}

int main(){
	float *A, *Ad;
	const size_t size = N * sizeof(float);
	A = (float*)malloc(size);
	HIPCHECK(hipHostRegister(A, size, 0));

	for(int i=0;i<N;i++){
		A[i] = float(1);
	}

	HIPCHECK(hipMalloc(&Ad, size));

	HIPCHECK(hipMemcpy(Ad, A, size, hipMemcpyHostToDevice));

	hipLaunchKernel(HIP_KERNEL_NAME(Inc), dim3(N/512), dim3(512), 0, 0, Ad);
	HIPCHECK(hipDeviceSynchronize());

	HIPCHECK(hipMemcpy(A, Ad, size, hipMemcpyDeviceToHost));

	HIPASSERT(A[10] == 2.0f);
	HIPCHECK(hipHostUnregister(A));
	passed();
}

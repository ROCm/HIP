/* HIT_START
 * BUILD: %t %s ../test_common.cpp EXCLUDE_HIP_PLATFORM nvcc
 * RUN: %t
 * HIT_END
 */

#include<iostream>
#include<string.h>
#include"hip/hip_runtime.h"
#include"hip/hip_runtime_api.h"
#include "test_common.h"

#define N 512
using namespace std;

texture<int, hipTextureType1D,hipReadModeElementType> tex;

bool testResult=true;

__global__ void kernel(int *out)
{

	int x=blockIdx.x * blockDim.x + threadIdx.x;
	out[x]=tex1Dfetch(tex,x);

}



void runTest(void);

int main(int argc, char** argv) {
    runTest();

    if (testResult) {
        passed();
    } else {
        exit(EXIT_FAILURE);
    }
}

void runTest()

{
string out;
int *tex_buf;
int val[N],i,output[N];
size_t size=0;

for(i=0;i<N;i++)
{
val[i]=i;
output[i]=0;
}
hipChannelFormatDesc chan_desc=hipCreateChannelDesc(32,0,0,0,hipChannelFormatKindUnsigned);

hipMalloc(&tex_buf,N*sizeof(int));

hipMemcpy(tex_buf,val,N*sizeof(int),hipMemcpyHostToDevice);

tex.addressMode[0]=hipAddressModeWrap;
tex.filterMode=hipFilterModeLinear;
tex.normalized=true;

hipBindTexture(&size,&tex,(void *)tex_buf,&chan_desc,N*sizeof(int)); 


dim3 dimBlock(64,1,1);
dim3 dimGrid(N/dimBlock.x,1,1);

hipLaunchKernelGGL(kernel,dim3(dimGrid), dim3(dimBlock), 0,0,output);

hipDeviceSynchronize();

hipMemcpy(output,tex_buf,N*sizeof(int),hipMemcpyDeviceToHost);

for(i=0;i<N;i++)
 {
   if(output[i] != val[i])
	{
	  testResult=false;
	  return;
	}
 }
hipUnbindTexture(&tex);

hipFree(tex_buf);


}








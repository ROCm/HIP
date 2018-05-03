/* HIT_START
 * BUILD: %t %s ../test_common.cpp
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

bool testResult=true;

__global__ void increment_values(float *val,hipTextureObject_t obj)
  {
	int k=blockIdx.x * blockDim.x + threadIdx.x;
	val[k]=tex1Dfetch<float>(obj,k);
	
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


//Allocating the required buffer on gpu device
float *tex_buf,*tex_buf_check;
float val[N],output[N];
int i;
for(i=0;i<N;i++)
    val[i]=(i+1)*(i+1);
hipMalloc(&tex_buf,N*sizeof(float));


hipMalloc(&tex_buf_check,N*sizeof(float));


hipMemcpy(tex_buf,val,N*sizeof(float),hipMemcpyHostToDevice);

hipMemset(tex_buf_check,0,N*sizeof(float));
hipResourceDesc res_lin;

memset(&res_lin,0,sizeof(res_lin));

res_lin.resType=hipResourceTypeLinear;
res_lin.res.linear.devPtr=tex_buf;
res_lin.res.linear.desc.f=hipChannelFormatKindFloat;
res_lin.res.linear.desc.x=32;
res_lin.res.linear.sizeInBytes=N*sizeof(float);

hipTextureDesc tex_desc;
memset(&tex_desc,0,sizeof(tex_desc));
tex_desc.readMode=hipReadModeElementType;


//Creating texture object

hipTextureObject_t tex_obj=0;

hipCreateTextureObject(&tex_obj,&res_lin,&tex_desc,NULL );

dim3 dimBlock(64,1,1);
dim3 dimGrid(N/dimBlock.x,1,1);

for(i=0;i<N;i++)
output[i]=0;


hipLaunchKernelGGL(increment_values,dim3(dimGrid),dim3(dimBlock), 0,0, tex_buf_check,tex_obj  );
hipDeviceSynchronize();



hipMemcpy(output,tex_buf_check,N*sizeof(float),hipMemcpyDeviceToHost);



for(i=0;i<N;i++)
	if(output[i] != val[i])
 	  {
	    testResult=false;
	  }


hipDestroyTextureObject(tex_obj);
hipFree(tex_buf);
hipFree(tex_buf_check);

}

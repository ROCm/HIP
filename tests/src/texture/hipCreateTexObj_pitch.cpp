/* HIT_START
 * BUILD: %t %s ../test_common.cpp
 * RUN: %t
 * HIT_END
 */

#include <iostream>
#include<string.h>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "test_common.h"

#define Width 64
#define Height 64

using namespace std;

bool testResult=true;

__global__ void increment_values(hipTextureObject_t obj, int width, int height, size_t pitch, float *tex_buf)
  {
        int k=blockIdx.x * blockDim.x + threadIdx.x;
	int l=blockIdx.y * blockDim.y + threadIdx.y;
	float *ptr=(float*)(((char *)tex_buf) + (l*pitch));
	ptr[k]=100.0f;

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

	void *tex_buf,*tex_buf_check;
	int pit_size=Width*Height,i,j;
	float val[Width][Height],output[Width][Height];
	string out;
	size_t pitch;

//	cout<<"The values to be fed in the buffer are:"<<endl;
	for(i=0;i<Width;i++)
		for(j=0;j<Height;j++)
		{
			val[i][j]=(i+1)*(j+1);
//			cout<<" "<<val[i][j];	
		}
//	cout<<endl;


	out=hipGetErrorString(hipMallocPitch(&tex_buf,&pitch,Width*sizeof(float),Height));
//	cout<<"The output of hipMalloc() is: "<<out<<endl;
//	cout<<"The value of the pitch is: "<<pitch<<endl;
	out=hipGetErrorString(hipMemcpy2D(tex_buf,pitch,val,Width*sizeof(float),Width*sizeof(float),Height,hipMemcpyHostToDevice));
//	cout<<"The output of hipMemcpy2D() is: "<<out<<endl;

	hipResourceDesc res_pit;

	memset(&res_pit,0,sizeof(res_pit));

	res_pit.resType=hipResourceTypePitch2D;
	res_pit.res.pitch2D.width=128;
	res_pit.res.pitch2D.height=128;
	res_pit.res.pitch2D.desc.x=32;
	res_pit.res.pitch2D.pitchInBytes=pit_size;
	res_pit.res.pitch2D.devPtr=tex_buf;

	hipTextureDesc tex_desc;
	memset(&tex_desc,0,sizeof(tex_desc));

	tex_desc.addressMode[0]=hipAddressModeWrap;
	tex_desc.filterMode=hipFilterModeLinear;
	tex_desc.readMode=hipReadModeElementType;

	hipTextureObject_t tex_obj=0;

	out=hipGetErrorString(hipCreateTextureObject(&tex_obj,&res_pit,&tex_desc,NULL));
//	cout<<"The output of hipCreateTextureObject() api is: "<<out<<endl;
	int batchx=128*128/2;
	dim3 dimBlock(64,1,1);
	dim3 dimGrid(batchx/dimBlock.x,1,1);



	hipLaunchKernelGGL(increment_values,dim3(dimGrid),dim3(dimBlock),0,0,tex_obj,Width,Height,pitch,tex_buf);

	hipDeviceSynchronize();

//	cout<<"Copying the modified buffer data to the host memory."<<endl;
	out=hipGetErrorString(hipMemcpy2D(output,Width*sizeof(float),tex_buf,pitch,Width*sizeof(float),Height,hipMemcpyDeviceToHost));
//	cout<<"The output of hipMemcpy2D() is: "<<out<<endl;

	for(i=0;i<Width;i++)
		for(j=0;j<Height;j++)
			{
				if(output[i][j]!=100)
				{
				testResult=false;	
				return;
				}
			}

	hipDestroyTextureObject(tex_obj);
	hipFree(tex_buf);
	hipFree(tex_buf_check);

}

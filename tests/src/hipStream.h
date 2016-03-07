#ifndef HIPSTREAM_H
#define HIPSTREAM_H
#include<hip_runtime.h>

#define NUM_STREAMS 4

/*
* H2H - 1
* H2D - 2
* KER - 3
* D2D - 4
* D2H - 5
*/

template<typename T>
void H2HAsync(T *Dst, T *Src, size_t size, hipStream_t stream){
	HIPCHECK(hipMemcpyAsync(Dst, Src, size, hipMemcpyHostToHost, stream));
}

template<typename T>
void H2DAsync(T *Dst, T *Src, size_t size, hipStream_t stream){
	HIPCHECK(hipMemcpyAsync(Dst, Src, size, hipMemcpyHostToDevice, stream));
}

template<typename T>
void D2DAsync(T *Dst, T *Src, size_t size, hipStream_t stream){
	HIPCHECK(hipMemcpyAsync(Dst, Src, size, hipMemcpyDeviceToDevice, stream));
}

template<typename T>
void D2HAsync(T *Dst, T *Src, size_t size, hipStream_t stream){
	HIPCHECK(hipMemcpyAsync(Dst, Src, size, hipMemcpyDeviceToHost, stream));
}

template<typename T>
void H2H(T *Dst, T *Src, size_t size){
	HIPCHECK(hipMemcpy(Dst, Src, size, hipMemcpyHostToHost));
}

template<typename T>
void H2D(T *Dst, T *Src, size_t size){
	HIPCHECK(hipMemcpy(Dst, Src, size, hipMemcpyHostToDevice));
}

template<typename T>
void D2D(T *Dst, T *Src, size_t size){
	HIPCHECK(hipMemcpy(Dst, Src, size, hipMemcpyDeviceToDevice));
}

template<typename T>
void D2H(T *Dst, T *Src, size_t size){
	HIPCHECK(hipMemcpy(Dst, Src, size, hipMemcpyDeviceToHost));
}

template<typename T>
__global__ void Inc(hipLaunchParm lp, T *In){
int tx = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
In[tx] = In[tx] + 1;
}

template<typename T>
void initArrays(T **Ad, T **Ah,
		size_t N, bool usePinnedHost=false){
	size_t NBytes = N * sizeof(T);
	if(Ad){
		HIPCHECK( hipMalloc(Ad, NBytes));
	}
	if(usePinnedHost){
		HIPCHECK( hipMallocHost(Ah, NBytes));
	}
	else{
		*Ah = new T[N];
		HIPASSERT(*Ah != NULL);
	}
}

template<typename T>
void initArrays(T **Ad, size_t N, 
		bool deviceMemory = false, 
		bool usePinnedHost = false){
	size_t NBytes = N * sizeof(T);
	if(deviceMemory){
		HIPCHECK( hipMalloc(Ad, NBytes)); 
	}else{
		if(usePinnedHost){
			HIPCHECK(hipMallocHost(Ad, NBytes));
		}else{
			*Ad = new T[N];
			HIPASSERT(*Ad != NULL);
		}
	}
}

template<typename T>
void setArray(T* Array, int N, T val){
for(int i=0;i<N;i++){
Array[i] = val;
}
}


#endif

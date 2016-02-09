/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once

#include <cuda_runtime_api.h>


#ifdef __cplusplus
extern "C" {
#endif

    //TODO -move to include/hip_runtime_api.h as a common implementation.
/**
* Memory copy types
*
*/
typedef enum hipMemcpyKind {
hipMemcpyHostToHost
,hipMemcpyHostToDevice
,hipMemcpyDeviceToHost
,hipMemcpyDeviceToDevice
,hipMemcpyDefault
} hipMemcpyKind ;

// hipErrorNoDevice.

/*typedef enum hipTextureFilterMode 
{
    hipFilterModePoint = cudaFilterModePoint,  ///< Point filter mode.
//! @warning cudaFilterModeLinear is not supported.
} hipTextureFilterMode;*/
#define hipFilterModePoint cudaFilterModePoint


typedef cudaEvent_t hipEvent_t;
typedef cudaStream_t hipStream_t;
//typedef cudaChannelFormatDesc hipChannelFormatDesc;
#define hipChannelFormatDesc cudaChannelFormatDesc

inline static hipError_t hipCUDAErrorTohipError(cudaError_t cuError) {
switch(cuError) {
case cudaSuccess:
    return hipSuccess;
case cudaErrorMemoryAllocation:
    return hipErrorMemoryAllocation;
case cudaErrorInvalidDevicePointer:
case cudaErrorInitializationError:
    return hipErrorMemoryFree;
default:
    return hipErrorUnknown;
}
}
// TODO   match the error enum names of hip and cuda 
inline static cudaError_t hipErrorToCudaError(hipError_t hError) {
switch(hError) {
case hipSuccess:
    return cudaSuccess;
case hipErrorMemoryAllocation:
    return cudaErrorMemoryAllocation;
case hipErrorMemoryFree:
    return cudaErrorInitializationError;
default:
    return cudaErrorUnknown;
}
}

inline static cudaMemcpyKind hipMemcpyKindToCudaMemcpyKind(hipMemcpyKind kind) {
switch(kind) {
case hipMemcpyHostToHost:
    return cudaMemcpyHostToHost;
case hipMemcpyHostToDevice:
    return cudaMemcpyHostToDevice;
case hipMemcpyDeviceToHost:
    return cudaMemcpyDeviceToHost;
default:
    return cudaMemcpyDefault;
}
}

inline static hipError_t hipDeviceReset() {
    return hipCUDAErrorTohipError(cudaDeviceReset());
}

inline static hipError_t hipGetLastError() {
    return hipCUDAErrorTohipError(cudaGetLastError());
}

inline static hipError_t hipMalloc(void** ptr, size_t size) {
    return hipCUDAErrorTohipError(cudaMalloc(ptr, size));
}

inline static hipError_t hipFree(void* ptr) {
    return hipCUDAErrorTohipError(cudaFree(ptr));
}

inline static hipError_t hipMallocHost(void** ptr, size_t size) {
    return hipCUDAErrorTohipError(cudaMallocHost(ptr, size));
}
inline static hipError_t hipFreeHost(void* ptr) {
    return hipCUDAErrorTohipError(cudaFreeHost(ptr));
}
inline static hipError_t hipSetDevice(int device) {
    return hipCUDAErrorTohipError(cudaSetDevice(device));
}
inline static hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyKind) {
  return hipCUDAErrorTohipError(cudaMemcpy(dst, src, sizeBytes, hipMemcpyKindToCudaMemcpyKind(copyKind)));
}


inline static hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind copyKind, hipStream_t stream=0) {
  return hipCUDAErrorTohipError(cudaMemcpyAsync(dst, src, sizeBytes, hipMemcpyKindToCudaMemcpyKind(copyKind), stream));
}


inline static hipError_t hipMemcpyToSymbol(const char *	symbolName, const void* src, size_t sizeBytes, size_t	offset = 0, hipMemcpyKind copyType = hipMemcpyHostToDevice) {
	return hipCUDAErrorTohipError(cudaMemcpyToSymbol(symbolName, src, sizeBytes, offset, hipMemcpyKindToCudaMemcpyKind(copyType)));
}
inline static hipError_t hipDeviceSynchronize() {
    return hipCUDAErrorTohipError(cudaDeviceSynchronize());
}

inline static const char* hipGetErrorString(hipError_t error){
    return cudaGetErrorString( hipErrorToCudaError(error) );
}

inline static hipError_t hipGetDeviceCount(int * count){
    return hipCUDAErrorTohipError(cudaGetDeviceCount(count));
}

inline static hipError_t hipGetDevice(int * device){
    return hipCUDAErrorTohipError(cudaGetDevice(device));
}

inline static hipError_t hipMemset(void* devPtr,int value, size_t count) {
    return hipCUDAErrorTohipError(cudaMemset(devPtr, value, count));
}

inline static hipError_t hipDeviceGetProperties(hipDeviceProp_t *p_prop, int device)
{
	cudaDeviceProp cdprop;
	cudaError_t cerror;
	cerror = cudaGetDeviceProperties(&cdprop,device);
	strcpy(p_prop->name,cdprop.name);
	p_prop->totalGlobalMem = cdprop.totalGlobalMem ;
	p_prop->sharedMemPerBlock = cdprop.sharedMemPerBlock;
	p_prop->regsPerBlock = cdprop.regsPerBlock;
	p_prop->warpSize = cdprop.warpSize ;
	for (int i=0 ; i<3; i++) {
		p_prop->maxThreadsDim[i] = cdprop.maxThreadsDim[i];
		p_prop->maxGridSize[i] = cdprop.maxGridSize[i];
	}
	p_prop->maxThreadsPerBlock = cdprop.maxThreadsPerBlock ;
	p_prop->clockRate = cdprop.clockRate;
	p_prop->totalConstMem = cdprop.totalConstMem ;
	p_prop->major = cdprop.major ;
	p_prop->minor = cdprop. minor ;
	p_prop->multiProcessorCount = cdprop.multiProcessorCount ;
	p_prop->l2CacheSize = cdprop.l2CacheSize ;
	p_prop->maxThreadsPerMultiProcessor = cdprop.maxThreadsPerMultiProcessor ;
	p_prop->computeMode = cdprop.computeMode ;

	// Same as clock-rate:
	p_prop->clockInstructionRate = cdprop.clockRate; 

	int ccVers = p_prop->major*100 + p_prop->minor * 10;

    p_prop->arch.hasGlobalInt32Atomics       =  (ccVers >= 110);
    p_prop->arch.hasGlobalFloatAtomicExch    =  (ccVers >= 110);
    p_prop->arch.hasSharedInt32Atomics       =  (ccVers >= 120);
    p_prop->arch.hasSharedFloatAtomicExch    =  (ccVers >= 120);

    p_prop->arch.hasFloatAtomicAdd           =  (ccVers >= 200);

    p_prop->arch.hasGlobalInt64Atomics       =  (ccVers >= 120);
    p_prop->arch.hasSharedInt64Atomics       =  (ccVers >= 110);

    p_prop->arch.hasDoubles                  =  (ccVers >= 130);

    p_prop->arch.hasWarpVote                 =  (ccVers >= 120);
    p_prop->arch.hasWarpBallot               =  (ccVers >= 200);
    p_prop->arch.hasWarpShuffle              =  (ccVers >= 300);
    p_prop->arch.hasFunnelShift              =  (ccVers >= 350);

    p_prop->arch.hasThreadFenceSystem        =  (ccVers >= 200);
    p_prop->arch.hasSyncThreadsExt           =  (ccVers >= 200);

    p_prop->arch.hasSurfaceFuncs             =  (ccVers >= 200);
    p_prop->arch.has3dGrid                   =  (ccVers >= 200);
    p_prop->arch.hasDynamicParallelism       =  (ccVers >= 350);

    p_prop->concurrentKernels = cdprop.concurrentKernels;

    return hipCUDAErrorTohipError(cerror);
}

inline static hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device)
{
    cudaDeviceAttr cdattr;
    cudaError_t cerror;
    
    switch (attr) {
    case hipDeviceAttributeMaxThreadsPerBlock:
        cdattr = cudaDevAttrMaxThreadsPerBlock; break;
    case hipDeviceAttributeMaxBlockDimX:
        cdattr = cudaDevAttrMaxBlockDimX; break;
    case hipDeviceAttributeMaxBlockDimY:
        cdattr = cudaDevAttrMaxBlockDimY; break;
    case hipDeviceAttributeMaxBlockDimZ:
        cdattr = cudaDevAttrMaxBlockDimZ; break;
    case hipDeviceAttributeMaxGridDimX:
        cdattr = cudaDevAttrMaxGridDimX; break;
    case hipDeviceAttributeMaxGridDimY:
        cdattr = cudaDevAttrMaxGridDimY; break;
    case hipDeviceAttributeMaxGridDimZ:
        cdattr = cudaDevAttrMaxGridDimZ; break;
    case hipDeviceAttributeMaxSharedMemoryPerBlock:
        cdattr = cudaDevAttrMaxSharedMemoryPerBlock; break;
    case hipDeviceAttributeTotalConstantMemory:
        cdattr = cudaDevAttrTotalConstantMemory; break;
    case hipDeviceAttributeWarpSize:
        cdattr = cudaDevAttrWarpSize; break;
    case hipDeviceAttributeMaxRegistersPerBlock:
        cdattr = cudaDevAttrMaxRegistersPerBlock; break;
    case hipDeviceAttributeClockRate:
        cdattr = cudaDevAttrClockRate; break;
    case hipDeviceAttributeMultiprocessorCount:
        cdattr = cudaDevAttrMultiProcessorCount; break;
    case hipDeviceAttributeComputeMode:
        cdattr = cudaDevAttrComputeMode; break;
    case hipDeviceAttributeL2CacheSize:
        cdattr = cudaDevAttrL2CacheSize; break;
    case hipDeviceAttributeMaxThreadsPerMultiProcessor:
        cdattr = cudaDevAttrMaxThreadsPerMultiProcessor; break;
    case hipDeviceAttributeComputeCapabilityMajor:
        cdattr = cudaDevAttrComputeCapabilityMajor; break;
    case hipDeviceAttributeComputeCapabilityMinor:
        cdattr = cudaDevAttrComputeCapabilityMinor; break;
    default:
        cerror = cudaErrorInvalidValue; break;
    }

    cerror = cudaDeviceGetAttribute(pi, cdattr, device);

    return hipCUDAErrorTohipError(cerror);
}

inline static hipError_t hipMemGetInfo( size_t* free, size_t* total)
{
    return hipCUDAErrorTohipError(cudaMemGetInfo(free,total));
}

inline static hipError_t hipEventCreate( hipEvent_t* event)
{
    return hipCUDAErrorTohipError(cudaEventCreate(event));
}
 
inline static hipError_t hipEventRecord( hipEvent_t event, hipStream_t stream = NULL)
{
    return hipCUDAErrorTohipError(cudaEventRecord(event,stream));
}

inline static hipError_t hipEventSynchronize( hipEvent_t event)
{
    return hipCUDAErrorTohipError(cudaEventSynchronize(event));
}

inline static hipError_t hipEventElapsedTime( float *ms, hipEvent_t start, hipEvent_t stop)
{
    return hipCUDAErrorTohipError(cudaEventElapsedTime(ms,start,stop));
}

inline static hipError_t hipEventDestroy( hipEvent_t event)
{
    return hipCUDAErrorTohipError(cudaEventDestroy(event));
}


inline static hipError_t hipStreamCreateWithFlags(hipStream_t *stream, unsigned int flags)
{
    return hipCUDAErrorTohipError(cudaStreamCreateWithFlags(stream, flags));
}


inline static hipError_t hipStreamCreate(hipStream_t *stream)
{
    return hipCUDAErrorTohipError(cudaStreamCreate(stream));
}


inline static hipError_t hipStreamDestroy(hipStream_t stream) 
{
    return hipCUDAErrorTohipError(cudaStreamDestroy(stream));
}


inline static hipError_t hipDriverGetVersion(int *driverVersion) 
{
	cudaError_t err = cudaDriverGetVersion(driverVersion);

	// Override driver version to match version reported on HCC side.
    *driverVersion = 4;

	return hipCUDAErrorTohipError(err);
}


inline static hipError_t hipDeviceCanAccessPeer ( int* canAccessPeer, int  device, int  peerDevice )
{
    return hipCUDAErrorTohipError(cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice));
}

inline static hipError_t  hipDeviceDisablePeerAccess ( int  peerDevice )
{
    return hipCUDAErrorTohipError(cudaDeviceDisablePeerAccess ( peerDevice ));
};

inline static hipError_t  hipDeviceEnablePeerAccess ( int  peerDevice, unsigned int  flags )
{
    return hipCUDAErrorTohipError(cudaDeviceEnablePeerAccess ( peerDevice, flags ));
}

inline static hipError_t hipMemcpyPeer ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count )
{
    return hipCUDAErrorTohipError(cudaMemcpyPeer ( dst, dstDevice, src, srcDevice, count ));
};

inline static hipError_t hipMemcpyPeerAsync ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count, hipStream_t stream=0 )
{
    return hipCUDAErrorTohipError(cudaMemcpyPeerAsync ( dst, dstDevice, src, srcDevice, count, stream ));
};




#ifdef __cplusplus
}
#endif

#ifdef __CUDACC__

template <class T, int dim, enum cudaTextureReadMode readMode>
inline static hipError_t  hipBindTexture(size_t *offset,
			                             const struct texture<T, dim, readMode> &tex,
										 const void *devPtr,
									     size_t size=UINT_MAX)
{
		return  hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, size));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
inline static hipError_t  hipBindTexture(size_t *offset, 
                                     struct texture<T, dim, readMode> *tex, 
                                     const void *devPtr, 
                                     const struct hipChannelFormatDesc *desc, 
                                     size_t size=UINT_MAX) 
{
		return  hipCUDAErrorTohipError(cudaBindTexture(offset, tex, devPtr, desc, size));
}

template <class T, int dim, enum cudaTextureReadMode readMode>
inline static hipError_t  hipUnbindTexture(struct texture<T, dim, readMode> *tex)
{
		return  hipCUDAErrorTohipError(cudaUnbindTexture(tex));
}

template <class T>
inline static hipChannelFormatDesc hipCreateChannelDesc()
{
		return cudaCreateChannelDesc<T>();
}
#endif

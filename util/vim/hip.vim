" Vim syntax file
" Language:	HIP
" Maintainer:	Aditya Atluri <aditya.atluri@amd.com>

if exists("b:current_syntax")
  finish
endif

" Read the C syntax to start with
if version < 600
  so <sfile>:p:h/cpp.vim
else
  runtime! syntax/cpp.vim
  unlet b:current_syntax
endif

syn case ignore

" hip keywords:

syn keyword hipType __global__ __host__ __device__ __constant__ __shared__ __kernel__
syn keyword hipKeyword hipThreadIdx_x hipThreadIdx_y hipThreadIdx_z
syn keyword hipKeyword hipBlockDim_x hipBlockDim_y hipBlockDim_z
syn keyword hipKeyword hipBlockIdx_x hipBlockIdx_y hipBlockIdx_z
syn keyword hipKeyword hipGridIdx_x hipGridIdx_y hipGridIdx_z
syn keyword hipKeyword hipGridDim_x hipGridDim_y hipGridDim_z

syn keyword hipType uint uint1 uint2 uint3 uint4
syn keyword hipType int1 int2 int3 int4
syn keyword hipType float1 float2 float3 float4
syn keyword hipType char1 char2 char3 char4
syn keyword hipType uchar1 uchar2 uchar3 uchar4
syn keyword hipType short1 short2 short3 short4
syn keyword hipType dim1 dim2 dim3 dim4
syn keyword hipType hipLaunchParm

" Atomic functions
syn keyword hipFunctionName atomicAdd atomicAnd atomicCAS atomicDec atomicExch
syn keyword hipFunctionName atomicInc atomicMax atomicMin atomicOr atomicSub atomicXor

" Texture functions
syn keyword hipFunctionName tex1D tex1Dfetch tex2D

" Type conversion functions
syn keyword hipFunctionName __float_as_int __int_as_float __float2int_rn
syn keyword hipFunctionName __float2int_rz __float2int_ru __float2int_rd
syn keyword hipFunctionName __float2uint_rn __float2uint_rz __float2uint_ru
syn keyword hipFunctionName __float2uint_rd __int2float_rn __int2float_rz
syn keyword hipFunctionName __int2float_ru __int2float_rd 
syn keyword hipFunctionName __uint2float_rn __uint2float_rz __uint2float_ru __uint2float_rd

" Intrinsic Math functions
syn keyword hipFunctionName __fadd_rz __fmul_rz __fdividef
syn keyword hipFunctionName __mul24 __umul24
syn keyword hipFunctionName __mulhi __umulhi
syn keyword hipFunctionName __mul64hi __umul64hi

" Math functions
syn keyword hipFunctionName min umin fminf fmin max umax fmaxf fmax
syn keyword hipFunctionName abs fabsf fabs sqrtf sqrt
syn keyword hipFunctionName sinf __sinf sin cosf __cosf cos sincosf __sincosf 
syn keyword hipFunctionName expf __expf exp logf __logf log 

" Runtime Data Types
syn keyword hipType hipDeviceProp_t 
syn keyword hipType hipError_t 
syn keyword hipType hipStream_t
syn keyword hipType hipEvent_t

" Runtime functions
syn keyword hipFunctionName hipBindTexture hipBindTextureToArray 
syn keyword hipFunctionName hipChooseDevice hipConfigureCall hipCreateChannelDesc 
syn keyword hipFunctionName hipD3D10GetDevice hipD3D10MapResources
syn keyword hipFunctionName hipD3D10RegisterResource hipD3D10ResourceGetMappedArray
syn keyword hipFunctionName hipD3D10ResourceGetMappedPitch
syn keyword hipFunctionName hipD3D10ResourceGetMappedPointer
syn keyword hipFunctionName hipD3D10ResourceGetMappedSize
syn keyword hipFunctionName hipD3D10ResourceGetSurfaceDimensions
syn keyword hipFunctionName hipD3D10ResourceSetMapFlags
syn keyword hipFunctionName hipD3D10SetDirect3DDevice
syn keyword hipFunctionName hipD3D10UnmapResources
syn keyword hipFunctionName hipD3D10UnregisterResource
syn keyword hipFunctionName hipD3D9GetDevice
syn keyword hipFunctionName hipD3D9GetDirect3DDevice
syn keyword hipFunctionName hipD3D9MapResources 
syn keyword hipFunctionName hipD3D9RegisterResource 
syn keyword hipFunctionName hipD3D9ResourceGetMappedArray 
syn keyword hipFunctionName hipD3D9ResourceGetMappedPitch 
syn keyword hipFunctionName hipD3D9ResourceGetMappedPointer 
syn keyword hipFunctionName hipD3D9ResourceGetMappedSize 
syn keyword hipFunctionName hipD3D9ResourceGetSurfaceDimensions 
syn keyword hipFunctionName hipD3D9ResourceSetMapFlags 
syn keyword hipFunctionName hipD3D9SetDirect3DDevice 
syn keyword hipFunctionName hipD3D9UnmapResources 
syn keyword hipFunctionName hipD3D9UnregisterResource 
syn keyword hipFunctionName hipGetDeviceProperties 
syn keyword hipFunctionName hipDeviceSynchronize 
syn keyword hipFunctionName hipDeviceReset 
syn keyword hipFunctionName hipEventCreate 
syn keyword hipFunctionName hipEventDestroy 
syn keyword hipFunctionName hipEventElapsedTime 
syn keyword hipFunctionName hipEventQuery 
syn keyword hipFunctionName hipEventRecord 
syn keyword hipFunctionName hipEventSynchronize 
syn keyword hipFunctionName hipFree 
syn keyword hipFunctionName hipFreeArray 
syn keyword hipFunctionName hipHostMalloc  
syn keyword hipFunctionName hipHostFree  
syn keyword hipFunctionName hipHostGetDevicePointer  
syn keyword hipFunctionName hipHostGetFlags  
syn keyword hipFunctionName hipHostRegister  
syn keyword hipFunctionName hipHostUnregister  

syn keyword hipFunctionName hipGetChannelDesc 
syn keyword hipFunctionName hipGetDevice 
syn keyword hipFunctionName hipGetDeviceCount 
syn keyword hipFunctionName hipGetDeviceProperties 
syn keyword hipFunctionName hipGetErrorString 
syn keyword hipFunctionName hipGetLastError 
syn keyword hipFunctionName hipGetSymbolAddress 
syn keyword hipFunctionName hipGetSymbolSize 
syn keyword hipFunctionName hipGetTextureAlignmentOffset 
syn keyword hipFunctionName hipGetTextureReference 
syn keyword hipFunctionName hipGLMapBufferObject 
syn keyword hipFunctionName hipGLRegisterBufferObject 
syn keyword hipFunctionName hipGLSetGLDevice 
syn keyword hipFunctionName hipGLUnmapBufferObject 
syn keyword hipFunctionName hipGLUnregisterBufferObject 
syn keyword hipFunctionName hipLaunchKernel
syn keyword hipFunctionName hipLaunchParm
syn keyword hipFunctionName hipMalloc 
syn keyword hipFunctionName hipMalloc3D 
syn keyword hipFunctionName hipMalloc3DArray 
syn keyword hipFunctionName hipMallocArray 
syn keyword hipFunctionName hipMallocHost 
syn keyword hipFunctionName hipMallocPitch 
syn keyword hipFunctionName hipMemcpy 
syn keyword hipFunctionName hipMemcpyAsync
syn keyword hipFunctionName hipMemcpy2D 
syn keyword hipFunctionName hipMemcpy2DArrayToArray 
syn keyword hipFunctionName hipMemcpy2DFromArray 
syn keyword hipFunctionName hipMemcpy2DToArray 
syn keyword hipFunctionName hipMemcpy3D 
syn keyword hipFunctionName hipMemcpyArrayToArray 
syn keyword hipFunctionName hipMemcpyFromArray 
syn keyword hipFunctionName hipMemcpyFromSymbol 
syn keyword hipFunctionName hipMemcpyToArray 
syn keyword hipFunctionName hipMemcpyToSymbol 
syn keyword hipFunctionName hipMemset 
syn keyword hipFunctionName hipMemset2D 
syn keyword hipFunctionName hipMemset3D 
syn keyword hipFunctionName hipSetDevice 
syn keyword hipFunctionName hipSetupArgument 
syn keyword hipFunctionName hipStreamCreateWithFlags
syn keyword hipFunctionName hipStreamCreate 
syn keyword hipFunctionName hipStreamDestroy 
syn keyword hipFunctionName hipStreamQuery 
syn keyword hipFunctionName hipStreamSynchronize 
syn keyword hipFunctionName hipThreadExit 
syn keyword hipFunctionName hipThreadSynchronize 
syn keyword hipFunctionName hipUnbindTexture 
syn keyword hipFunctionName hipDeviceCanAccessPeer 
syn keyword hipFunctionName hipDeviceEnablePeerAccess
syn keyword hipFunctionName hipDeviceDisablePeerAccess
syn keyword hipFunctionName hipMemcpyPeer
syn keyword hipFunctionName hipMemcpyPeerAsync


" HIP Flags
syn keyword hipFlags hipFilterModePoint 
syn keyword hipFlags hipMemcpyHostToDevice 
syn keyword hipFlags hipMemcpyDeviceToDevice 
syn keyword hipFlags hipMemcpyHostToHost
syn keyword hipFlags hipMemcpyDeviceToHost 
syn keyword hipFlags hipMemcpyHostToHost 
syn keyword hipFlags hipMemcpyDeviceToDevice 
syn keyword hipFlags hipMemcpyDefault 
syn keyword hipFlags hipReadModeElementType 
syn keyword hipFlags hipSuccess 
syn keyword hipFlags hipErrorNotReady 
syn keyword hipFlags hipTextureType1D 


syn keyword hipFlags hipHostMallocDefault
syn keyword hipFlags hipHostMallocPortable
syn keyword hipFlags hipHostMallocMapped
syn keyword hipFlags hipHostMallocWriteCombined
syn keyword hipFlags hipHostMallocCoherent
syn keyword hipFlags hipHostMallocNonCoherent

syn keyword hipFlags hipHostRegisterDefault
syn keyword hipFlags hipHostRegisterPortable
syn keyword hipFlags hipHostRegisterMapped
syn keyword hipFlags hipHostRegisterIoMemory

" Define the default highlighting.
" For version 5.7 and earlier: only when not done already
" For version 5.8 and later: only when an item doesn't have highlighting yet
if version >= 508 || !exists("did_c_syn_inits")
  if version < 508
    let did_c_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif

  HiLink hipType      Type
  HiLink hipKeyword   Statement
	HiLink hipFlags			Boolean
  HiLink hipFunctionName Function


  delcommand HiLink
endif

let b:current_syntax = "hip"

# Table Comparing Syntax for Different Compute APIs

|Term|CUDA|HIP|OpenCL|
|---|---|---|---|
|Device|`int deviceId`|`int deviceId`|`cl_device`|
|Queue|`cudaStream_t`|`hipStream_t`|`cl_command_queue`|
|Event|`cudaEvent_t`|`hipEvent_t`|`cl_event`|
|Memory|`void *`|`void *`|`cl_mem`|
|||||
| |grid|grid|NDRange|
| |block|block|work-group|
| |thread|thread|work-item|
| |warp|warp|sub-group|
|||||
|Thread-<br>index | threadIdx.x | threadIdx.x  | get_local_id(0) |
|Block-<br>index  | blockIdx.x  | blockIdx.x  | get_group_id(0) |
|Block-<br>dim    | blockDim.x  | blockDim.x  | get_local_size(0) |
|Grid-dim     | gridDim.x   | gridDim.x  | get_num_groups(0) |
|||||
|Device Kernel|`__global__`|`__global__`|`__kernel`|
|Device Function|`__device__`|`__device__`|Implied in device compilation|
|Host Function|`__host_` (default)|`__host_` (default)|Implied in host compilation|
|Host + Device Function|`__host__` `__device__`|`__host__` `__device__`| No equivalent|
|Kernel Launch|`<<< >>>`|`hipLaunchKernel`/`hipLaunchKernelGGL`/`<<< >>>`|`clEnqueueNDRangeKernel`|
||||||
|Global Memory|`__global__`|`__global__`|`__global`|
|Group Memory|`__shared__`|`__shared__`|`__local`|
|Constant|`__constant__`|`__constant__`|`__constant`|
||||||
||`__syncthreads`|`__syncthreads`|`barrier(CLK_LOCAL_MEMFENCE)`|
|Atomic Builtins|`atomicAdd`|`atomicAdd`|`atomic_add`|
|Precise Math|`cos(f)`|`cos(f)`|`cos(f)`|
|Fast Math|`__cos(f)`|`__cos(f)`|`native_cos(f)`|
|Vector|`float4`|`float4`|`float4`|

### Notes
The indexing functions (starting with `thread-index`) show the terminology for a 1D grid.  Some APIs use reverse order of xyz / 012 indexing for 3D grids.


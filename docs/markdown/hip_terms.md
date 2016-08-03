# Table Comparing Syntax for Different Compute APIs

|Term|CUDA|HIP|HC|C++AMP|OpenCL|
|---|---|---|---|---|---|
|Device|`int deviceId`|`int deviceId`|`hc::accelerator`|`concurrency::`<br>`accelerator`|`cl_device`
|Queue|`cudaStream_t`|`hipStream_t`|`hc::`<br>`accelerator_view`|`concurrency::`<br>`accelerator_view`|`cl_command_queue`
|Event|`cudaEvent_t`|`hipEvent_t`|`hc::`<br>`completion_future`|`concurrency::`<br>`completion_future`|`cl_event`
|Memory|`void *`|`void *`|`void *`; `hc::array`; `hc::array_view`|`concurrency::array`;<br>`concurrency::array_view`|`cl_mem`
|||||
| |grid|grid|extent|extent|NDRange
| |block|block|tile|tile|work-group
| |thread|thread|thread|thread|work-item
| |warp|warp|wavefront|N/A|sub-group
|||||
|Thread-<br>index | threadIdx.x | hipThreadIdx_x | t_idx.local[0] | t_idx.local[0] | get_local_id(0) |
|Block-<br>index  | blockIdx.x  | hipBlockIdx_x  | t_idx.tile[0]  | t_idx.tile[0]  | get_group_id(0) |
|Block-<br>dim    | blockDim.x  | hipBlockDim_x  | t_ext.tile_dim[0]| t_idx.tile_dim0 | get_local_size(0) |
|Grid-dim     | gridDim.x   | hipGridDim_x   | t_ext[0]| t_ext[0] | get_global_size(0) |
|||||
|Device Kernel|`__global__`|`__global__`|lambda inside `hc::`<br>`parallel_for_each` or [[hc]]|`restrict(amp)`|`__kernel`
|Device Function|`__device__`|`__device__`|`[[hc]]` (detected automatically in many case)|`restrict(amp)`|Implied in device compilation
|Host Function|`__host_` (default)|`__host_` (default)|`[[cpu]]`  (default)|`restrict(cpu)` (default)|Implied in host compilation.
|Host + Device Function|`__host__` `__device__`|`__host__` `__device__`|  `[[hc]]` `[[cpu]]`|`restrict(amp,cpu)`|No equivalent
|Kernel Launch|`<<< >>>`|`hipLaunchKernel`|`hc::`<br>`parallel_for_each`|`concurrency::`<br>`parallel_for_each`|`clEnqueueNDRangeKernel`
||||||
|Global Memory|`__global__`|`__global__`|Unnecessary / Implied|Unnecessary / Implied|`__global`
|Group Memory|`__shared__`|`__shared__`|`tile_static`|`tile_static`|`__local`
|Constant|`__constant__`|`__constant__`|Unnecessary / Implied|Unnecessary / Implied|`__constant`
||||||
||`__syncthreads`|`__syncthreads`|`tile_static.barrier()`|`t_idx.barrier()`|`barrier(CLK_LOCAL_MEMFENCE)`
|Atomic Builtins|`atomicAdd`|`atomicAdd`|`hc::atomic_fetch_add`|`concurrency::`<br>`atomic_fetch_add`|`atomic_add`
|Precise Math|`cos(f)`|`cos(f)`|`hc::`<br>`precise_math::cos(f)`|`concurrency::`<br>`precise_math::cos(f)`|`cos(f)`
|Fast Math|`__cos(f)`|`__cos(f)`|`hc::`<br>`fast_math::cos(f)`|`concurrency::`<br>`fast_math::cos(f)`|`native_cos(f)`
|Vector|`float4`|`float4`|`hc::`<br>`short_vector::float4`|`concurrency::`<br>`graphics::float_4`|`float4`

###Notes
1. For HC and C++AMP, assume a captured _tiled_ext_ named "t_ext" and captured _extent_ named "ext".  These languages use captured variables to pass information to the kernel rather than using special built-in functions so the exact variable name may vary.
2. The indexing functions (starting with `thread-index`) show the terminology for a 1D grid.  Some APIs use reverse order of xyz / 012 indexing for 3D grids.
3. HC allows tile dimensions to be specified at runtime while C++AMP requires that tile dimensions be specified at compile-time.  Thus hc syntax for tile dims is `t_ext.tile_dim[0]` while C++AMP is t_ext.tile_dim0.


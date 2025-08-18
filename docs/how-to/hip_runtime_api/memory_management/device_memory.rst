.. meta::
  :description: This chapter describes the device memory of the HIP ecosystem
                ROCm software.
  :keywords: AMD, ROCm, HIP, GPU, device memory, global, constant, texture, surface, shared

.. _device_memory:

********************************************************************************
Device memory
********************************************************************************

Device memory is random access memory that is physically located on a GPU. In
general it is memory with a bandwidth that is an order of magnitude higher
compared to RAM available to the host. That high bandwidth is only available to
on-device accesses, accesses from the host or other devices have to go over a
special interface which is considerably slower, usually the PCIe bus or the AMD
Infinity Fabric.

On certain architectures like APUs, the GPU and CPU share the same physical
memory.

There is also a special local data share on-chip directly accessible to the
:ref:`compute units <hardware_implementation>`, that can be used for shared
memory.

The physical device memory can be used to back up several different memory
spaces in HIP, as described in the following.

Global memory
================================================================================

Global memory is the general read-write accessible memory visible to all threads
on a given device. Since variables located in global memory have to be marked
with the ``__device__`` qualifier, this memory space is also referred to as
device memory.

Without explicitly copying it, it can only be accessed by the threads within a
kernel operating on the device, however :ref:`unified_memory` can be used to
let the runtime manage this, if desired.

Allocating global memory
--------------------------------------------------------------------------------

This memory needs to be explicitly allocated.

It can be allocated from the host via the :ref:`HIP runtime memory management
functions <memory_management_reference>` like :cpp:func:`hipMalloc`, or can be
defined using the ``__device__`` qualifier on variables.

It can also be allocated within a kernel using ``malloc`` or ``new``.
The specified amount of memory is allocated by each thread that executes the
instructions. The recommended way to allocate the memory depends on the use
case. If the memory is intended to be shared between the threads of a block, it
is generally beneficial to allocate one large block of memory, due to the way
the memory is accessed.

.. note::
   Memory allocated within a kernel can only be freed in kernels, not by the HIP
   runtime on the host, like :cpp:func:`hipFree`. It is also not possible to
   free device memory allocated on the host, with :cpp:func:`hipMalloc` for
   example, in a kernel.


An example for how to share memory allocated within a kernel by only one thread
is given in the following example. In case the device memory is only needed for
communication between the threads in a single block, :ref:`shared_memory` is the
better option, but is also limited in size.

.. code-block:: cpp

  __global__ void kernel_memory_allocation(TYPE* pointer){
    // The pointer is stored in shared memory, so that all
    // threads of the block can access the pointer
    __shared__ int *memory;

    size_t blockSize = blockDim.x;
    constexpr size_t elementsPerThread = 1024;
    if(threadIdx.x == 0){
      // allocate memory in one contiguous block
      memory = new int[blockDim.x * elementsPerThread];
    }
    __syncthreads();

    // load pointer into thread-local variable to avoid
    // unnecessary accesses to shared memory
    int *localPtr = memory;

    // work with allocated memory, e.g. initialization
    for(int i = 0; i < elementsPerThread; ++i){
      // access in a contiguous way
      localPtr[i * blockSize + threadIdx.x] = i;
    }

    // synchronize to make sure no thread is accessing the memory before freeing
    __syncthreads();
    if(threadIdx.x == 0){
      delete[] memory;
    }
  }

Copying between device and host
--------------------------------------------------------------------------------

When not using :ref:`unified_memory`, memory has to be explicitly copied between
the device and the host, using the HIP runtime API.

.. code-block:: cpp

  size_t elements = 1 << 20;
  size_t size_bytes = elements * sizeof(int);

  // allocate host and device memory
  int *host_pointer = new int[elements];
  int *device_input, *device_result;
  HIP_CHECK(hipMalloc(&device_input, size_bytes));
  HIP_CHECK(hipMalloc(&device_result, size_bytes));

  // copy from host to the device
  HIP_CHECK(hipMemcpy(device_input, host_pointer, size_bytes, hipMemcpyHostToDevice));

  // Use memory on the device, i.e. execute kernels

  // copy from device to host, to e.g. get results from the kernel
  HIP_CHECK(hipMemcpy(host_pointer, device_result, size_bytes, hipMemcpyDeviceToHost));

  // free memory when not needed any more
  HIP_CHECK(hipFree(device_result));
  HIP_CHECK(hipFree(device_input));
  delete[] host_pointer;

Constant memory
================================================================================

Constant memory is read-only storage visible to all threads on a given device.
It is a limited segment backed by device memory, that takes a different caching
route than normal device memory accesses. It needs to be set by the host before
kernel execution.

In order to get the highest bandwidth from the constant memory, all threads of
a warp have to access the same memory address. If they access different
addresses, the accesses get serialized and the bandwidth is therefore reduced.

Using constant memory
--------------------------------------------------------------------------------

Constant memory can not be dynamically allocated, and the size has to be
specified during compile time. If the values can not be specified during compile
time, they have to be set by the host before the kernel, that accesses the
constant memory, is called.

.. code-block:: cpp

  constexpr size_t const_array_size = 32;
  __constant__ double const_array[const_array_size];

  void set_constant_memory(double* values){
    hipMemcpyToSymbol(const_array, values, const_array_size * sizeof(double));
  }

  __global__ void kernel_using_const_memory(double* array){

    int warpIdx = threadIdx.x / warpSize;
    // uniform access of warps to const_array for best performance
    array[blockDim.x] *= const_array[warpIdx];
  }

Texture memory
================================================================================

Texture memory is special read-only memory visible to all threads on a given
device and accessible through additional APIs. Its origins come from graphics
APIs, and provides performance benefits when accessing memory in a pattern where
the addresses are close to each other in a 2D or 3D representation of the
memory. It also provides additional features like filtering and addressing for
out-of-bounds accesses, which are further explained in :ref:`texture_fetching`.

The original use of the texture cache was also to take pressure off the global
memory and other caches, however on modern GPUs, that support textures, the L1
cache and texture cache are combined, so the main purpose is to make use of the
texture specific features.

To find out whether textures are supported on a device, query
:cpp:enumerator:`hipDeviceAttributeImageSupport`.

Using texture memory
--------------------------------------------------------------------------------

Textures are more complex than just a region of memory, so their layout has to
be specified. They are represented by ``hipTextureObject_t`` and created using
:cpp:func:`hipCreateTextureObject`.

The underlying memory is a 1D, 2D or 3D ``hipArray_t``, that needs to be
allocated using :cpp:func:`hipMallocArray`.

On the device side, texture objects are accessed using the ``tex1D/2D/3D``
functions.

The texture management functions can be found in the :ref:`Texture management
API reference <texture_management_reference>`

A full example for how to use textures can be found in the `ROCm texture
management example <https://github.com/ROCm/rocm-examples/blob/develop/HIP-Basic/texture_management/main.hip>`_

Surface memory
================================================================================

A read-write version of texture memory. It is created in the same way as a
texture, but with :cpp:func:`hipCreateSurfaceObject`.

Since surfaces are also cached in the read-only texture cache, the changes
written back to the surface can't be observed in the same kernel. A new kernel
has to be launched in order to see the updated surface.

The corresponding functions are listed in the :ref:`Surface object API reference
<surface_object_reference>`.

.. _shared_memory:

Shared memory
================================================================================

Shared memory is read-write memory, that is only visible to the threads within a
block. It is allocated per thread block, and needs to be either statically
allocated at compile time, or can be dynamically allocated when launching the
kernel, but not during kernel execution. Its general use-case is to share
variables between the threads within a block, but can also be used as scratch
pad memory.

Shared memory is not backed by the same physical memory as the other address
spaces. It is on-chip memory local to the :ref:`compute units
<hardware_implementation>`, providing low-latency, high-bandwidth access,
comparable to the L1 cache. It is however limited in size, and as it is
allocated per block, can restrict how many blocks can be scheduled to a compute
unit concurrently, thereby potentially reducing occupancy.

An overview of the size of the local data share (LDS), that backs up shared
memory, is given in the
:doc:`GPU hardware specifications <rocm:reference/gpu-arch-specs>`.

Allocate shared memory
--------------------------------------------------------------------------------

Memory can be dynamically allocated by declaring an ``extern __shared__`` array,
whose size can be set during kernel launch, which can then be accessed in the
kernel.

.. code-block:: cpp

  extern __shared__ int dynamic_shared[];
  __global__ void kernel(int array1SizeX, int array1SizeY, int array2Size){
    // at least (array1SizeX * array1SizeY + array2Size) * sizeof(int) bytes
    // dynamic shared memory need to be allocated when the kernel is launched
    int* array1 = dynamic_shared;
    // array1 is interpreted as 2D of size:
    int array1Size = array1SizeX * array1SizeY;

    int* array2 = &(array1[array1Size]);

    if(threadIdx.x < array1SizeX && threadIdx.y < array1SizeY){
      // access array1 with threadIdx.x + threadIdx.y * array1SizeX
    }
    if(threadIdx.x < array2Size){
      // access array2 threadIdx.x
    }
  }

A more in-depth example on dynamically allocated shared memory can be found in
the  `ROCm dynamic shared example
<https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/dynamic_shared>`_.

To statically allocate shared memory, just declare it in the kernel. The memory
is allocated per block, not per thread. If the kernel requires more shared
memory than is available to the architecture, the compilation fails.

.. code-block:: cpp

  __global__ void kernel(){
    __shared__ int array[128];
    __shared__ double result;
  }

A more in-depth example on statically allocated shared memory can be found in
the  `ROCm shared memory example
<https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/shared_memory>`_.


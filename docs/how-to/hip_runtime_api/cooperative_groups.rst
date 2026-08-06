.. meta::
  :description: This topic describes how to use cooperative groups in HIP
  :keywords: AMD, ROCm, HIP, cooperative groups

.. _cooperative_groups_how-to:

*******************************************************************************
Cooperative groups
*******************************************************************************

The cooperative groups API is an extension to the HIP programming model, which
provides developers with a flexible, dynamic grouping mechanism for the
communicating threads. Cooperative groups let you define your own set of thread
groups which may fit your use-cases better than those defined by the hardware.
This lets you specify the level of granularity for thread communication which
can lead to more efficient parallel decompositions.

The API is accessible in the ``cooperative_groups`` namespace after the
``hip_cooperative_groups.h`` header is included. The header contains the following
elements:

* Static functions to create groups and subgroups.
* Hardware-accelerated operations over the whole group, like shuffles.
* Data types of cooperative groups.
* Synchronize member function of the groups.
* Get group properties member functions.

Cooperative groups thread model
================================================================================

The thread hierarchy abstractions of cooperative groups are depicted in the following figures: :ref:`grid hierarchy <coop_thread_top_hierarchy>` and :ref:`block hierarchy <coop_thread_bottom_hierarchy>`.

.. _coop_thread_top_hierarchy:

.. figure:: ../../data/how-to/hip_runtime_api/cooperative_groups/thread_hierarchy_coop_top.svg
  :alt: Diagram depicting nested rectangles of varying color. The outermost one
        titled "Grid", inside sets of different sized rectangles layered on
        one another titled "Block". Each "Block" containing sets of uniform
        rectangles layered on one another titled "Warp". Each of the "Warp"
        titled rectangles filled with downward pointing arrows inside.

  Cooperative group thread hierarchy in grids.

The **multi grid** is an abstraction of potentially multiple simultaneous
launches of the same kernel over multiple devices. The **grid** in cooperative
groups is a single dispatch of kernels for execution like the original grid.

.. note::

  * The ability to synchronize over a grid or multi grid requires the kernel to
    be launched using the specific cooperative groups API.

  * Multi grid deprecated since ROCm 5.0.

The **block** is the same as the :ref:`inherent_thread_model` block entity.

.. note::

  Explicit warp-level thread handling is absent from the Cooperative groups API. In order to exploit the known hardware SIMD width on which built-in functionality translates to simpler logic, you can use the group partitioning part of the API, such as ``tiled_partition``.

.. _coop_thread_bottom_hierarchy:

.. figure:: ../../data/how-to/hip_runtime_api/cooperative_groups/thread_hierarchy_coop_bottom.svg
  :alt: The new level between block thread and threads.

  Cooperative group thread hierarchy in blocks.

The cooperative groups API introduce a new level between block thread and threads. The :ref:`thread-block tile <coop_thread_block_tile>` give the opportunity to have tiles in the thread block, while the :ref:`coalesced group <coop_coalesced_groups>` holds the active threads of the parent group. These groups further discussed in the :ref:`groups types <coop_group_types>` section.

For details on memory model, check the :ref:`memory model description <memory_hierarchy>`.

.. _coop_group_types:

Group types
===========

Group types are based on the levels of synchronization and data sharing among threads.

Thread-block group
------------------

Represents an intra-block cooperative groups type where the participating threads within the group are the same threads that participated in the currently executing ``block``.

.. code-block:: cpp

  class thread_block;

Constructed via:

.. code-block:: cpp

  thread_block g = this_thread_block();

The ``group_index()`` , ``thread_index()`` , ``thread_rank()`` , ``size()``, ``cg_type()``, ``is_valid()`` , ``sync()``, ``barrier_arrive()``, ``barrier_wait()`` and ``group_dim()`` member functions are public of the thread_block class. For further details, check the :ref:`thread_block references <thread_block_ref>` .

Grid group
------------

Represents an inter-block cooperative groups type where the group's participating threads span multiple blocks running the same kernel on the same device. Use the cooperative launch API to enable synchronization across the grid group.

.. code-block:: cpp

  class grid_group;

Constructed via:

.. code-block:: cpp

  grid_group g = this_grid();

The ``thread_rank()`` , ``size()``, ``cg_type()``, ``is_valid()``, ``sync()``, ``barrier_arrive()`` and ``barrier_wait()`` member functions
are public of the ``grid_group`` class. For further details, check the :ref:`grid_group references <grid_group_ref>`.

Multi-grid group
------------------

Represents an inter-device cooperative groups type where the participating threads within the group span multiple devices that run the same kernel on the devices. Use the cooperative launch API to enable synchronization across the multi-grid group.

.. code-block:: cpp

  class multi_grid_group;

Constructed via:

.. code-block:: cpp

  // Kernel must be launched with the cooperative multi-device API
  multi_grid_group g = this_multi_grid();

The ``num_grids()`` , ``grid_rank()`` , ``thread_rank()``, ``size()``, ``cg_type()``, ``is_valid()`` ,
and ``sync()`` member functions are public of the ``multi_grid_group`` class. For
further details check the :ref:`multi_grid_group references <multi_grid_group_ref>` .

.. _coop_thread_block_tile:

Thread-block tile
------------------

This constructs a templated class derived from ``thread_group``. The template defines the tile
size of the new thread group at compile time. This group type also supports sub-wave level intrinsics.

.. code-block:: cpp

  template <unsigned int Size, typename ParentT = void>
  class thread_block_tile;

Constructed via:

.. code-block:: cpp

  template <unsigned int Size, typename ParentT>
  _CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)


.. note::

  * Size must be a power of 2 and not larger than warp (wavefront) size.
  * ``shfl()`` functions support integer or float type.

The ``thread_rank()`` , ``size()``, ``cg_type()``, ``is_valid()``, ``sync()``, ``meta_group_rank()``, ``meta_group_size()``, ``shfl()``, ``shfl_down()``, ``shfl_up()``, ``shfl_xor()``, ``ballot()``, ``any()``, ``all()``, ``match_any()`` and ``match_all()`` member functions are public of the ``thread_block_tile`` class. For further details, check the :ref:`thread_block_tile references <thread_block_tile_ref>` .

.. _coop_coalesced_groups:

Coalesced groups
------------------

Threads (64 threads on CDNA and 32 threads on RDNA) in a warp cannot execute different instructions simultaneously, so conditional branches are executed serially within the warp. When threads encounter a conditional branch, they can diverge, resulting in some threads being disabled if they do not meet the condition to execute that branch. The active threads are referred to as coalesced, and coalesced group represents an active thread group within a warp.

.. note::

  The NVIDIA GPU's independent thread scheduling presents the appearance that threads on different branches execute concurrently.

.. warning::

  AMD GPUs do not support independent thread scheduling. Some CUDA application can rely on this feature and the ported HIP version on AMD GPUs can deadlock, when they try to make use of independent thread scheduling. 	

This group type also supports sub-wave level intrinsics.

.. code-block:: cpp

  class coalesced_group;

Constructed via:

.. code-block:: cpp

  coalesced_group active = coalesced_threads();

.. note::

  ``shfl()`` functions support integer or float type.

The ``thread_rank()`` , ``size()``, ``cg_type()``, ``is_valid()``, ``sync()``, ``meta_group_rank()``, ``meta_group_size()``, ``shfl()``, ``shfl_down()``, ``shfl_up()``, ``ballot()``, ``any()``, ``all()``, ``match_any()`` and ``match_all()`` member functions are public of the ``coalesced_group`` class. For more information, see :ref:`coalesced_group references <coalesced_group_ref>` .

.. _coop_cluster_group:

Cluster group
-------------

Represents a group of thread blocks that execute together on the device. A cluster
can be 1D, 2D, or 3D and can contain up to 15 workgroups. Each block in the cluster
runs on a separate Workgroup Processor (WGP). The cluster group provides
synchronization and shared-memory mapping across those blocks.

.. code-block:: cpp

  class cluster_group;

Constructed via:

.. code-block:: cpp

  cluster_group g = this_cluster();

The following member functions are public on the ``cluster_group`` class:

* ``sync()`` — synchronizes all threads in the cluster.
* ``barrier_arrive()`` — arrives at the cluster barrier and returns an ``arrival_token``.
* ``barrier_wait(arrival_token&&)`` — waits on the token returned by ``barrier_arrive()``.
* ``block_index()`` — returns the 3D index of the calling block within the cluster.
* ``block_rank()`` — returns the rank of the calling block within ``[0, num_blocks())``.
* ``thread_index()`` — returns the 3D index of the calling thread within the cluster.
* ``thread_rank()`` — returns the rank of the calling thread within ``[0, num_threads())``.
* ``dim_blocks()`` — returns the dimensions of the launched cluster in units of blocks.
* ``num_blocks()`` — returns the total number of blocks in the cluster.
* ``dim_threads()`` — returns the dimensions of the launched cluster in units of threads.
* ``num_threads()`` (alias: ``size()``) — returns the total number of threads in the cluster.
* ``map_shared_rank<T>(T* addr, int rank)`` — returns the address of a shared-memory variable in the block with the given rank.
* ``query_shared_rank(const void* addr)`` — returns the block rank that owns the given shared-memory address.

.. note::

  ``map_shared_rank()`` and ``query_shared_rank()`` require a device that reports
  multi-block cluster support. Check ``hipDeviceProp_t::clusterLaunch`` at
  runtime before relying on these functions.

Cooperative groups simple example
=================================

The difference to the original block model in the ``reduce_sum`` device function is the following.

.. tab-set::
  .. tab-item:: Original Block
    :sync: original-block

    .. code-block:: cuda

      __device__ int reduce_sum(int *shared, int val) {

          // Thread ID
          const unsigned int thread_id = threadIdx.x;

          // Every iteration the number of active threads
          // halves, until we processed all values
          for(unsigned int i = blockDim.x / 2; i > 0; i /= 2) {
              // Store value in shared memory with thread ID
              shared[thread_id] = val;

              // Synchronize all threads
              __syncthreads();

              // Active thread sum up
              if(thread_id < i)
                  val += shared[thread_id + i];

              // Synchronize all threads in the group
              __syncthreads();
          }

          // ...
      }

  .. tab-item:: Cooperative groups
    :sync: cooperative-groups

    .. code-block:: cuda

      __device__ int reduce_sum(thread_group g,
                                int *shared,
                                int val) {

          // Thread ID
          const unsigned int group_thread_id = g.thread_rank();

          // Every iteration the number of active threads
          // halves, until we processed all values
          for(unsigned int i = g.size() / 2; i > 0; i /= 2) {
              // Store value in shared memroy with thread ID
              shared[group_thread_id] = val;

              // Synchronize all threads in the group
              g.sync();

              // Active thread sum up
              if(group_thread_id < i)
                  val += shared[group_thread_id + i];

              // Synchronize all threads in the group
              g.sync();
          }

          // ...
      }

The ``reduce_sum()`` function call and input data initialization difference to the original block model is the following.

.. tab-set::
  .. tab-item:: Original Block
    :sync: original-block

    .. code-block:: cuda

      __global__ void sum_kernel(...) {

          // ...

          // Workspace array in shared memory
          __shared__ unsigned int workspace[2048];

          // ...

          // Perform reduction
          output = reduce_sum(workspace, input);

          // ...
      }

  .. tab-item:: Cooperative groups
    :sync: cooperative-groups

    .. code-block:: cuda

      __global__ void sum_kernel(...) {

          // ...

          // Workspace array in shared memory
          __shared__ unsigned int workspace[2048];

          // ...

          // Initialize the thread_block
          thread_block thread_block_group = this_thread_block();
          // Perform reduction
          output = reduce_sum(thread_block_group, workspace, input);

          // ...
      }

At the device function, the input group type is the ``thread_group``, which is the parent class of all the cooperative groups type. With this, you can write generic functions, which can work with any type of cooperative groups.

.. _coop_synchronization:

Synchronization
===============

With each group type, the synchronization requires using the correct cooperative groups launch API.

**Check the kernel launch capability**

.. tab-set::
  .. tab-item:: Thread-block
    :sync: thread-block

    Do not need kernel launch validation.

  .. tab-item:: Grid
    :sync: grid

    Confirm the cooperative launch capability on the single AMD GPU:

    .. code-block:: cpp

        int device               = 0;
        int supports_coop_launch = 0;
        // Check support
        // Use hipDeviceAttributeCooperativeMultiDeviceLaunch when launching across multiple devices
        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(
            hipDeviceGetAttribute(&supports_coop_launch, hipDeviceAttributeCooperativeLaunch, device));
        if(!supports_coop_launch)
        {
            std::cout << "Skipping, device " << device << " does not support cooperative groups"
                      << std::endl;
            return 0;
        }

  .. tab-item:: Multi-grid
    :sync: multi-grid

    Confirm the cooperative launch capability over multiple GPUs:

    .. code-block:: cpp

        // Check support of cooperative groups
        std::vector<int> deviceIDs;
        for(int deviceID = 0; deviceID < device_count; deviceID++) {
        #ifdef __HIP_PLATFORM_AMD__
            int supports_coop_launch = 0;
            HIP_CHECK(
                hipDeviceGetAttribute(
                    &supports_coop_launch,
                    hipDeviceAttributeCooperativeMultiDeviceLaunch,
                    deviceID));
            if(!supports_coop_launch) {
                std::cout << "Skipping, device " << deviceID << " does not support cooperative groups"
                          << std::endl;
            }
            else
        #endif
            {
                std::cout << deviceID << std::endl;
                // Collect valid deviceIDs.
                deviceIDs.push_back(deviceID);
            }
        }

**Kernel launch**

.. tab-set::
  .. tab-item:: Thread-block
    :sync: thread-block

    You can access the new block representation using the original kernel launch methods.

    .. code-block:: cpp

        void* params[] = {&d_vector, &d_block_reduced, &d_partition_reduced};
        // Launching kernel from host.
        HIP_CHECK(hipLaunchKernelGGL(vector_reduce_kernel<partition_size>,
                                     dim3(num_blocks),
                                     dim3(threads_per_block),
                                     0,
                                     hipStreamDefault,
                                     &d_vector,
                                     &d_block_reduced,
                                     &d_partition_reduced));

  .. tab-item:: Grid
    :sync: grid

    Launch the cooperative kernel on a single GPU:

    .. code-block:: cpp

        void* params[] = {};
        // Launching kernel from host.
        HIP_CHECK(hipLaunchCooperativeKernel(vector_reduce_kernel<partition_size>,
                                             dim3(num_blocks),
                                             dim3(threads_per_block),
                                             0,
                                             0,
                                             hipStreamDefault));

  .. tab-item:: Multi-grid
    :sync: multi-grid

    Launch the cooperative kernel over multiple GPUs:

    .. code-block:: cpp

        hipLaunchParams *launchParamsList = (hipLaunchParams*)malloc(sizeof(hipLaunchParams) * deviceIDs.size());
        for(int deviceID : deviceIDs) {

            // Set device
            HIP_CHECK(hipSetDevice(deviceID));

            // Create stream
            hipStream_t stream;
            HIP_CHECK(hipStreamCreate(&stream));

            // Parameters
            void* params[] = {&(d_vector[deviceID]), &(d_block_reduced[deviceID]), &(d_partition_reduced[deviceID])};

            // Set launchParams
            launchParamsList[deviceID].func = (void*)vector_reduce_kernel<partition_size>;
            launchParamsList[deviceID].gridDim = dim3(1);
            launchParamsList[deviceID].blockDim = dim3(threads_per_block);
            launchParamsList[deviceID].sharedMem = 0;
            launchParamsList[deviceID].stream = stream;
            launchParamsList[deviceID].args = params;
        }

        HIP_CHECK(hipLaunchCooperativeKernelMultiDevice(launchParamsList,
                                                        (int)deviceIDs.size(),
                                                        hipCooperativeLaunchMultiDeviceNoPreSync));

**Device side synchronization**

.. tab-set::
  .. tab-item:: Thread-block
    :sync: thread-block

    The device side code of the thread_block synchronization over single GPUs:

    .. code-block:: cpp

      thread_block g = this_thread_block();
      g.sync();

  .. tab-item:: Grid
    :sync: grid

    The device side code of the grid synchronization over single GPUs:

    .. code-block:: cpp

      grid_group grid = this_grid();
      grid.sync();

  .. tab-item:: Multi-grid
    :sync: multi-grid

    The device side code of the multi-grid synchronization over multiple GPUs:

    .. code-block:: cpp

      multi_grid_group multi_grid = this_multi_grid();
      multi_grid.sync();

Split barrier
-------------

``barrier_arrive()`` and ``barrier_wait()`` split synchronization into two
separate steps, allowing useful work to be performed in between. This is
supported on ``thread_block``, ``grid_group``, and ``cluster_group``.

``barrier_arrive()`` signals that the calling thread has reached the barrier
and returns an ``arrival_token``. The thread can then continue executing
work that does not depend on other threads. ``barrier_wait()`` consumes the
token and blocks until all threads in the group have called
``barrier_arrive()``, at which point it is safe to read results written by
other threads.

.. code-block:: cpp

  auto tok = g.barrier_arrive();

  // Work that does not depend on other threads' results can go here.

  g.barrier_wait(std::move(tok));

The following example uses a split barrier on a ``thread_block`` to overlap
a write to shared memory with a global memory write, avoiding an idle wait:

.. code-block:: cpp

  __global__ void split_barrier_example(float* out, float* in) {
      namespace cg = cooperative_groups;

      __shared__ float mid[32];
      size_t i = threadIdx.x;
      auto tb = cg::this_thread_block();

      out[i] = in[i] * 2.0f;

      auto tok = tb.barrier_arrive();

      if (i == 0) {
          for (size_t j = 0; j < 32; j++)
              mid[j] = in[j];
      }

      tb.barrier_wait(std::move(tok));

      out[i] += mid[i];
  }

.. _cg_operations:

Operations
==========

All cooperative groups operations receive the same arguments:

* ``group`` is either a ``coalesced_group`` or a ``thread_block_tile``

* ``val`` needs to be a type ``T`` that is trivially copyable, default constructible, and up to 32 bytes in size.

* ``op`` must be a function object, which includes lambdas or functors which define ``operator()``. The following predefined functors in the ``cooperative_groups`` namespace:

  + ``cooperative_groups::plus`` (addition)

  + ``cooperative_groups::less`` (minimum)

  + ``cooperative_groups::greater`` (maximum)

  + ``cooperative_groups::bit_and`` (bitwise and)

  + ``cooperative_groups::bit_or`` (bitwise or)

  + ``cooperative_groups::bit_xor`` (bitwise xor)

Overloads without the ``op`` parameter use ``cooperative_groups::plus``.

Reduce
---------
Performs a group-wide reduce. Participation of all the threads belonging to the group is expected, with each thread contributing the same per-thread value. Behaviour is undefined if one of the threads of the group does not participate.

.. code-block:: cpp

  auto reduce(const TyGroup& group, T&& val, Operation&& op)

Defined in cooperative_groups/hip_reduce.h. Performs a reduction operation ``op`` on the specified group, contributing the value ``val``
The parameters are described here: :ref:`cg_operations`

**Performance**

On AMD, although all types ``T`` fulfilling the description above can be used with the functors in the ``cooperative_groups`` namespace, only some of them will receive hardware acceleration in the form of DPP instructions. Essentially only the types supported by ``__reduce_*_sync`` operations would potentially receive acceleration :ref:`hip_cpp_language_extensions:Warp reduction functions` The macro ``HIP_ENABLE_EXTRA_WARP_SYNC_TYPES`` might be needed to enable the hardware acceleration on some types.

For arithmetic reduces (``plus``, ``less`` and ``greater``):

* Nvidia: there is hardware acceleration for ``int``, ``unsigned int``

* AMD: there is hardware acceleration for ``int``, ``unsigned int``, and if the user defines the macro ``HIP_ENABLE_EXTRA_WARP_SYNC_TYPES``, then ``unsigned long long``, ``long long``, ``half``/``float``/``double`` precision floating point types will also receive hardware acceleration.

For bitwise-reduces: (``bit_and``, ``bit_or``, ``bit_xor``)

* Nvidia: ``unsigned int``

* AMD: ``unsigned int``, and if the user defines the macro ``HIP_ENABLE_EXTRA_WARP_SYNC_TYPES``, then ``int``, ``unsigned long long`` or ``long long`` are also hardware-accelerated.

inclusive_scan
-----------------

.. code-block:: cpp

  auto inclusive_scan(const TyGroup& group, TyVal&& val, Operation&& op)
  auto inclusive_scan(const TyGroup& group, TyVal&& val)

Defined in cooperative_groups/hip_scan.h. Performs an inclusive scan using the operation ``op`` on the specified group, contributing the value ``val``. Participation of all the threads belonging to the group is expected, with each thread contributing the same per-thread value. Behaviour is undefined if one of the threads of the group does not participate.

The parameters are described here: :ref:`cg_operations`

**Performance**

On AMD, when ``group`` is of the same size as the warp size and ``T`` a primitive type, DPP instructions are used, resulting in significantly faster execution than with other group sizes. The primitive types are:

For arithmetic scans (``plus``, ``less`` and ``greater``):

* Nvidia: there is hardware acceleration for ``int``, ``unsigned int``

* AMD: there is hardware acceleration for ``int``, ``unsigned int``, ``unsigned long long``, ``long long``, ``half``/``float``/``double`` 

For bitwise-scans: (``bit_and``, ``bit_or``, ``bit_xor``)

* Nvidia: ``unsigned int``

* AMD: ``unsigned int``, ``int``, ``unsigned long long``, ``long long``

exclusive_scan
-----------------
.. code-block:: cpp

  auto exclusive_scan(const TyGroup& group, TyVal&& val, Operation&& op)
  auto exclusive_scan(const TyGroup& group, TyVal&& val)

Defined in cooperative_groups/hip_scan.h. Performs an exclusive scan using the operation ``op`` on the specified group, contributing the value ``val``. Participation of all the threads belonging to the group is expected, with each thread contributing the same per-thread value. Behaviour is undefined if one of the threads of the group does not participate.

The parameters are described here: :ref:`cg_operations`

 The value returned for the first active lane is platform-dependent and may change in future releases. As a reference: 

* AMD - the "identity" value is returned, according to the operation:
  * ``plus``: always 0
  * ``less``:
    * for integer types: the maximum value for the type, i.e. ``std::numeric_limits<T>::max()``
    * for floating point types: ``inf``
  * ``greater``:
    * for integer types: the minimum value for the type, i.e. ``std::numeric_limits<T>::lowest()``
    * for floating point types: ``-inf``
  * ``bit_and``: a value with all the bits set to 1
  * ``bit_or``: always 0
  * ``bit_xor``: always 0

* NVIDIA - returns T {} (i.e. 0)

**Performance**

On AMD, when ``group`` is of the same size as the warp size and ``T`` a primitive type, DPP instructions are used, resulting in significantly faster execution than with other group sizes. The primitive types are:

For arithmetic scans (``plus``, ``less`` and ``greater``):

* Nvidia: there is hardware acceleration for ``int`` or ``unsigned int``

* AMD: there is hardware acceleration for ``int``, ``unsigned int``, ``unsigned long long``, ``long long``, ``half``/``float``/``double`` 

For bitwise-scans: (``bit_and``, ``bit_or``, ``bit_xor``)

* Nvidia: ``unsigned int``

* AMD: ``unsigned int``, ``int``, ``unsigned long long`` or ``long long``

memcpy_async
------------

To use ``memcpy_async``, include the optional header:

  .. code-block:: cpp

    #include <hip/cooperative_groups/memcpy_async.h>

Cooperatively copies memory between global and shared (LDS) memory, distributing
the work across all threads in the group. The copy is completed before the function
returns; no separate ``wait`` call is needed.

Two overloads are available:

.. code-block:: cpp

  // Byte-count overload
  void memcpy_async(const TyGroup& group,
                    TyElem* dst,
                    const TyElem* src,
                    const TySizeT& count);

  // Layout overload: copies min(dstLayout, srcLayout) elements
  void memcpy_async(const TyGroup& group,
                    TyElem* dst, const DstLayout& dstLayout,
                    const TyElem* src, const SrcLayout& srcLayout);

The supported group types are ``thread_block``, ``coalesced_group``, and
``thread_block_tile<N>``.

**Performance**

On devices that support hardware-accelerated asynchronous copies, ``memcpy_async``
uses dedicated global-to-LDS and LDS-to-global transfer paths in widths of 4, 8,
or 16 bytes. On other devices, the implementation falls back to a software copy
loop with equivalent semantics.

Unsupported NVIDIA CUDA features
================================

HIP doesn't support the following CUDA functions/operators in ``cooperative_groups`` namespace:

* ``synchronize``
* ``wait`` and ``wait_prior``
* ``invoke_one`` and ``invoke_one_broadcast``
* ``reduce_update_async`` and ``reduce_store_async``

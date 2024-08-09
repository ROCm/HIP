.. meta::
  :description: This chapter lists types and device API wrappers related to the Cooperative Group feature. Programmers can directly use these API features in their kernels.
  :keywords: AMD, ROCm, HIP, cooperative groups

.. _cooperative_groups_reference:

*******************************************************************************
HIP Cooperative groups API
*******************************************************************************

Cooperative kernel launches
===========================

The following host-side functions are used for cooperative kernel launches.

.. doxygenfunction:: hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream)

.. doxygenfunction:: hipLaunchCooperativeKernel(T f, dim3 gridDim, dim3 blockDim, void** kernelParams, unsigned int sharedMemBytes, hipStream_t stream)

.. doxygenfunction:: hipLaunchCooperativeKernelMultiDevice(hipLaunchParams* launchParamsList, int  numDevices, unsigned int  flags)

.. doxygenfunction:: hipModuleLaunchCooperativeKernel

.. doxygenfunction:: hipModuleLaunchCooperativeKernelMultiDevice

Cooperative groups classes
==========================

The following cooperative groups classes can be used on the device side.

.. _thread_group_ref:

.. doxygenclass:: cooperative_groups::thread_group
   :members:

.. _thread_block_ref:

.. doxygenclass:: cooperative_groups::thread_block
   :members:

.. _grid_group_ref:

.. doxygenclass:: cooperative_groups::grid_group
   :members:

.. _multi_grid_group_ref:

.. doxygenclass:: cooperative_groups::multi_grid_group
   :members:
 
.. _thread_block_tile_ref:

.. doxygenclass:: cooperative_groups::thread_block_tile
   :members:

.. _coalesced_group_ref:

.. doxygenclass:: cooperative_groups::coalesced_group
   :members:

Cooperative groups construct functions
======================================

The following functions are used to construct different group-type instances on the device side.

.. doxygenfunction:: cooperative_groups::this_multi_grid

.. doxygenfunction:: cooperative_groups::this_grid

.. doxygenfunction:: cooperative_groups::this_thread_block

.. doxygenfunction:: cooperative_groups::coalesced_threads

.. doxygenfunction:: cooperative_groups::tiled_partition(const ParentCGTy &g)

.. doxygenfunction:: cooperative_groups::tiled_partition(const thread_group &parent, unsigned int tile_size)

.. doxygenfunction:: cooperative_groups::binary_partition(const coalesced_group& cgrp, bool pred)

.. doxygenfunction:: cooperative_groups::binary_partition(const thread_block_tile<size, parent>& tgrp, bool pred)

Cooperative groups exposed API functions
========================================

The following functions are the exposed API for different group-type instances on the device side.

.. doxygenfunction:: cooperative_groups::group_size

.. doxygenfunction:: cooperative_groups::thread_rank

.. doxygenfunction:: cooperative_groups::is_valid

.. doxygenfunction:: cooperative_groups::sync
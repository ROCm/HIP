.. meta::
  :description: This chapter lists types and device API wrappers related to the
                Cooperative Group feature. Programmers can directly use these
                API features in their kernels.
  :keywords: AMD, ROCm, HIP, cooperative groups

.. _cooperative_groups_reference:

*******************************************************************************
Cooperative groups
*******************************************************************************

Cooperative kernel launches
===========================

The following host-side functions are used for cooperative kernel launches.

.. doxygengroup:: ModuleCooperativeG
   :content-only:

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

.. doxygengroup:: CooperativeGConstruct
   :content-only:

Cooperative groups exposed API functions
========================================

The following functions are the exposed API for different group-type instances on the device side.

.. doxygengroup:: CooperativeGAPI
   :content-only:

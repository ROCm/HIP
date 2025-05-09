.. meta::
  :description: HIP coherence control
                ecosystem ROCm software.
  :keywords: AMD, ROCm, HIP, host memory

.. _coherence_control:

*******************************************************************************
Coherence control
*******************************************************************************

Memory coherence describes how memory of a specific part of the system is
visible to the other parts of the system. For example, how GPU memory is visible
to the CPU and vice versa. In HIP, host and device memory can be allocated with
two different types of coherence:

* **Coarse-grained coherence:** The memory is considered up-to-date only after
  synchronization performed using :cpp:func:`hipDeviceSynchronize`,
  :cpp:func:`hipStreamSynchronize`, or any blocking operation that acts on the
  null stream such as :cpp:func:`hipMemcpy`. To avoid the cache from being
  accessed by a part of the system while simultaneously being written by
  another, the memory is made visible only after the caches have been flushed.

* **Fine-grained coherence:** The memory is coherent even while being modified
  by a part of the system. Fine-grained coherence ensures that up-to-date data
  is visible to others regardless of kernel boundaries. This can be useful if
  both host and device operate on the same data.

.. note::

  To achieve fine-grained coherence, many AMD GPUs use a limited cache policy,
  such as leaving these allocations uncached by the GPU or making them read-only.

Mi200 accelerator's hardware based floating point instructions work on
coarse-grained memory regions. Coarse-grained coherence is typically useful in
reducing host-device interconnect communication.

To check the availability of fine- and coarse-grained memory pools, use
``rocminfo``:

.. code-block:: sh

  $ rocminfo
  ...
  *******
  Agent 1
  *******
  Name:                    AMD EPYC 7742 64-Core Processor
  ...
  Pool Info:
  Pool 1
  Segment:                 GLOBAL; FLAGS: FINE GRAINED
  ...
  Pool 3
  Segment:                 GLOBAL; FLAGS: COARSE GRAINED
  ...
  *******
  Agent 9
  *******
  Name:                    gfx90a
  ...
  Pool Info:
  Pool 1
  Segment:                 GLOBAL; FLAGS: COARSE GRAINED
  ...

.. _hip-memory-coherence-table:

The APIs, flags and respective memory coherence control are listed in the
following table:

.. list-table:: Memory coherence control
    :widths: 25, 35, 20, 20
    :header-rows: 1
    :align: center

    * - API
      - Flag
      - :cpp:func:`hipMemAdvise` call with argument
      - Coherence
    * - ``hipHostMalloc`` :sup:`1`
      - ``hipHostMallocDefault``
      -
      - Fine-grained
    * - ``hipHostMalloc`` :sup:`1`
      - ``hipHostMallocNonCoherent``
      -
      - Coarse-grained
    * - ``hipExtMallocWithFlags``
      - ``hipDeviceMallocDefault``
      -
      - Coarse-grained
    * - ``hipExtMallocWithFlags``
      - ``hipDeviceMallocFinegrained``
      -
      - Fine-grained
    * - ``hipMallocManaged``
      -
      -
      - Fine-grained
    * - ``hipMallocManaged``
      -
      - ``hipMemAdviseSetCoarseGrain``
      - Coarse-grained
    * - ``malloc``
      -
      -
      - Fine-grained
    * - ``malloc``
      -
      - ``hipMemAdviseSetCoarseGrain``
      - Coarse-grained

:sup:`1` The :cpp:func:`hipHostMalloc` memory allocation coherence mode can be
affected by the ``HIP_HOST_COHERENT`` environment variable, if the
``hipHostMallocCoherent``, ``hipHostMallocNonCoherent``, and
``hipHostMallocMapped`` are unset. If neither these flags nor the
``HIP_HOST_COHERENT`` environment variable is set, or set as 0, the host memory
allocation is coarse-grained.

.. note::

  * When ``hipHostMallocMapped`` flag is set, the allocated host memory is
    fine-grained and the ``hipHostMallocNonCoherent`` flag is ignored.
  * Setting both the ``hipHostMallocCoherent`` and
    ``hipHostMallocNonCoherent`` flags leads to an illegal state.

Visibility of synchronization functions
================================================================================

The fine-grained coherence memory is visible at the synchronization points,
however the visibility of coarse-grained memory depends on the synchronization
function used. The effect and visibility of various synchronization functions on
fine- and coarse-grained memory types are listed here:

.. list-table:: HIP synchronize functions effect and visibility

    * - HIP API
      - :cpp:func:`hipStreamSynchronize`
      - :cpp:func:`hipDeviceSynchronize`
      - :cpp:func:`hipEventSynchronize`
      - :cpp:func:`hipStreamWaitEvent`
    * - Synchronization effect
      - Host waits for all commands in the specified stream to complete
      - Host waits for all commands in all streams on the specified device to complete
      - Host waits for the specified event to complete
      - Stream waits for the specified event to complete
    * - Fence
      - System-scope release
      - System-scope release
      - System-scope release
      - None
    * - Fine-grained host memory visibility
      - Yes
      - Yes
      - Yes
      - Yes
    * - Coarse-grained host memory visibility
      - Yes
      - Yes
      - Depends on the used event.
      - No

You can control the release scope for ``hipEvents``. By default, the GPU
performs a device-scope acquire and release operation with each recorded event.
This makes the host and device memory visible to other commands executing on the
same device.

:cpp:func:`hipEventCreateWithFlags`: You can specify a stronger system-level
fence by creating the event with ``hipEventCreateWithFlags``:

* ``hipEventReleaseToSystem``: Performs a system-scope release operation when
  the event is recorded. This makes both fine-grained and coarse-grained host
  memory visible to other agents in the system, which might also involve
  heavyweight operations such as cache flushing. Fine-grained memory typically
  uses lighter-weight in-kernel synchronization mechanisms such as an atomic
  operation and thus doesn't need to use  ``hipEventReleaseToSystem``.

* ``hipEventDisableTiming``: Events created with this flag don't record
  profiling data, which significantly improves synchronization performance.

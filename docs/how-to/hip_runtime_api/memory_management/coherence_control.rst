.. meta::
  :description: This chapter describes the coherence control of the HIP
                ecosystem ROCm software.
  :keywords: AMD, ROCm, HIP, host memory

.. _coherence_control:

*******************************************************************************
Coherence control
*******************************************************************************

Memory coherence describes how different parts of a system see the memory of a 
specific part of the system, e.g. how the CPU sees the GPUs memory or vice versa.
In HIP, host and device memory can be allocated with two different types of
coherence:

* **Coarse-grained coherence** means that memory is only considered up to date
  after synchronization, which can be enforced through :cpp:func:`hipDeviceSynchronize`,
  :cpp:func:`hipStreamSynchronize`, or any blocking operation that acts on the
  null stream (e.g. :cpp:func:`hipMemcpy`). One reason for this can be writes to
  caches, that the other part of the system can't access, so they are only
  visible once the caches have been flushed.
* **Fine-grained coherence** means the memory is coherent even while it is being
  modified by one of the parts of the system. Fine-grained coherence implies
  that up to date data is visible to others regardless of kernel boundaries.
  This can be useful if both host and device are operating on the same data.

.. note::

  In order to achieve this fine-grained coherence, many AMD GPUs use a limited
  cache policy, such as leaving these allocations uncached by the GPU, or making
  them read-only.

.. TODO: Is this still valid? What about Mi300?

Developers should use coarse-grained coherence where they can reduce host-device
interconnect communication and also Mi200 accelerators hardware based floating
point instructions are working on coarse grained memory regions.

The availability of fine- and coarse-grained memory pools can be checked with
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


The memory coherence control is described in the following table.

.. list-table:: Memory coherence control
    :widths: 25, 35, 20, 20
    :header-rows: 1
    :align: center

    * - API
      - Flag
      - :cpp:func:`hipMemAdvise` call with argument
      - Coherence
    * - ``hipHostMalloc``
      - ``hipHostMallocDefault``
      - 
      - Fine-grained
    * - ``hipHostMalloc``
      - ``hipHostMallocNonCoherent`` :sup:`1`
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
``hipHostMallocCoherent``, ``hipHostMallocNonCoherent``, ``hipHostMallocMapped``
are unset while one of the other flag is set. At this case, if the 
``HIP_HOST_COHERENT`` environment variable is not defined, or defined as 0, the
host memory allocation is coarse-grained.

.. note::

  * When ``hipHostMallocMapped`` flag is set, the allocated host memory is 
    fine-grained and the ``hipHostMallocNonCoherent`` flag is ignored.
  * It's an illegal state, if the ``hipHostMallocCoherent`` and
    ``hipHostMallocNonCoherent`` flags are set.

Visibility of synchronization functions
================================================================================

The fine-grained coherence memory is visible at synchronization points, however 
at coarse-grained coherence, it depends on the used synchronization function.
The synchronization functions effect and visibility on different coherence 
memory types collected in the following table.

.. list-table:: HIP synchronize functions effect and visibility

    * - HIP API
      - :cpp:func:`hipStreamSynchronize`
      - :cpp:func:`hipDeviceSynchronize`
      - :cpp:func:`hipEventSynchronize`
      - :cpp:func:`hipStreamWaitEvent`
    * - Synchronization Effect
      - host waits for all commands in the specified stream to complete
      - host waits for all commands in all streams on the specified device to complete
      - host waits for the specified event to complete
      - stream waits for the specified event to complete
    * - Fence
      - system-scope release
      - system-scope release
      - system-scope release
      - none
    * - Fine-grained host memory visibility
      - yes
      - yes
      - yes
      - yes
    * - Coarse-grained host memory visibility
      - yes
      - yes
      - depends on the used event.
      - no

Developers can control the release scope for :cpp:func:`hipEvents`:

* By default, the GPU performs a device-scope acquire and release operation
  with each recorded event.  This will make host and device memory visible to
  other commands executing on the same device.

A stronger system-level fence can be specified when the event is created with 
:cpp:func:`hipEventCreateWithFlags`:

* ``hipEventReleaseToSystem``: Perform a system-scope release operation
  when the event is recorded. This will make **both fine-grained and
  coarse-grained host memory visible to other agents in the system**, but may
  involve heavyweight operations such as cache flushing. Fine-grained memory
  will typically use lighter-weight in-kernel synchronization mechanisms such as
  an atomic operation and thus does not need to use.
  ``hipEventReleaseToSystem``.
* ``hipEventDisableTiming``: Events created with this flag will not
  record profiling data and provide the best performance if used for
  synchronization.

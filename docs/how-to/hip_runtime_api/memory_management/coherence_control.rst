.. meta::
  :description: This chapter describes the coherence control of the HIP
                ecosystem ROCm software.
  :keywords: AMD, ROCm, HIP, host memory

.. _coherence_control:

*******************************************************************************
Coherence control
*******************************************************************************

Memory coherence describes how different parts of a system see the memory of a specific part of the system, e.g. how the CPU sees the GPUs memory or vice versa.
In HIP, host and device memory can be allocated with two different types of coherence:

* **Coarse-grained coherence** means that memory is only considered up to date at 
  kernel boundaries, which can be enforced through hipDeviceSynchronize,
  hipStreamSynchronize, or any blocking operation that acts on the null
  stream (e.g. hipMemcpy). For example, cacheable memory is a type of
  coarse-grained memory where an up-to-date copy of the data can be stored
  elsewhere (e.g. in an L2 cache).
* **Fine-grained coherence** means the coherence is supported while a CPU/GPU 
  kernel is running. This can be useful if both host and device are operating on
  the same dataspace using system-scope atomic operations (e.g. updating an
  error code or flag to a buffer). Fine-grained memory implies that up-to-date
  data may be made visible to others regardless of kernel boundaries as
  discussed above.

.. note::

  In order to achieve this fine-grained coherence, many AMD GPUs use a limited
  cache policy, such as leaving these allocations uncached by the GPU, or making
  them read-only.

.. TODO: Is this still valid? What about Mi300?

Developers should use coarse-grained coherence where they can to reduce
host-device interconnect communication and also Mi200 accelerators hardware
based floating point instructions are working on coarse grained memory regions.

The availability of fine- and coarse-grained memory pools can be checked with
``rocminfo``.

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
``hipHostMallocCoherent=0``, ``hipHostMallocNonCoherent=0``,
``hipHostMallocMapped=0`` and one of the other flag is set to 1. At this case,
if the ``HIP_HOST_COHERENT`` is not defined, or defined as 0, the host memory
allocation is coarse-grained.

.. note::

  * At ``hipHostMallocMapped=1`` case the allocated host memory is 
    fine-grained and the ``hipHostMallocNonCoherent`` flag is ignored.
  * The ``hipHostMallocCoherent=1`` and ``hipHostMallocNonCoherent=1`` state is
    illegal. 

Visibility of synchronization functions
================================================================================

The fine-grained coherence memory is visible at synchronization points, however 
at coarse-grained coherence, it depends on the used synchronization function.
The synchronization functions effect and visibility on different coherence 
memory types collected in the following table.

.. list-table:: HIP API

    * - HIP API
      - ``hipStreamSynchronize``
      - ``hipDeviceSynchronize``
      - ``hipEventSynchronize``
      - ``hipStreamWaitEvent``
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

* :cpp:func:`hipEventReleaseToSystem`: Perform a system-scope release operation
  when the event is recorded. This will make **both fine-grained and
  coarse-grained host memory visible to other agents in the system**, but may
  involve heavyweight operations such as cache flushing. Fine-grained memory
  will typically use lighter-weight in-kernel synchronization mechanisms such as
  an atomic operation and thus does not need to use.
  :cpp:func:`hipEventReleaseToSystem`.
* :cpp:func:`hipEventDisableTiming`: Events created with this flag will not
  record profiling data and provide the best performance if used for
  synchronization.

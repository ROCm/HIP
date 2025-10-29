.. meta::
    :description: This chapter describes how to use multiple devices on one host.
    :keywords: ROCm, HIP, multi-device, multiple, GPUs, devices

.. _multi-device:

*******************************************************************************
Multi-device management
*******************************************************************************

Device enumeration
===============================================================================

Device enumeration involves identifying all the available GPUs connected to the
host system. A single host machine can have multiple GPUs, each with its own
unique identifier. By listing these devices, you can decide which GPU to use
for computation. The host queries the system to count and list all connected
GPUs that support the chosen ``HIP_PLATFORM``, ensuring that the application
can leverage the full computational power available. Typically, applications
list devices and their properties for deployment planning, and also make
dynamic selections during runtime to ensure optimal performance.

If the application does not define a specific GPU, device 0 is selected.

.. literalinclude:: ../../tools/example_codes/device_enumeration.cpp
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

.. _multi_device_selection:

Device selection
===============================================================================

Once you have enumerated the available GPUs, the next step is to select a
specific device for computation. This involves setting the active GPU that will
execute subsequent operations. This step is crucial in multi-GPU systems where
different GPUs might have different capabilities or workloads. By selecting the
appropriate device, you ensure that the computational tasks are directed to the
correct GPU, optimizing performance and resource utilization.

.. literalinclude:: ../../tools/example_codes/device_selection.hip
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

Stream and event behavior
===============================================================================

In a multi-device system, streams and events are essential for efficient
parallel computation and synchronization. Streams enable asynchronous task
execution, allowing multiple devices to process data concurrently without
blocking one another. Events provide a mechanism for synchronizing operations
across streams and devices, ensuring that tasks on one device are completed
before dependent tasks on another device begin. This coordination prevents race
conditions and optimizes data flow in multi-GPU systems. Together, streams and
events maximize performance by enabling parallel execution, load balancing, and
effective resource utilization across heterogeneous hardware.

.. literalinclude:: ../../tools/example_codes/multi_device_synchronization.hip
    :start-after: // [sphinx-start]
    :end-before: // [sphinx-end]
    :language: cpp

Peer-to-peer memory access
===============================================================================

In multi-GPU systems, peer-to-peer memory access enables one GPU to directly
read or write to the memory of another GPU. This capability reduces data
transfer times by allowing GPUs to communicate directly without involving the
host. Enabling peer-to-peer access can significantly improve the performance of
applications that require frequent data exchange between GPUs, as it eliminates
the need to transfer data through the host memory.

By adding peer-to-peer access to the example referenced in
:ref:`multi_device_selection`, data can be efficiently copied between devices.
If peer-to-peer access is not activated, the call to :cpp:func:`hipMemcpy`
still works but internally uses a staging buffer in host memory, which incurs a
performance penalty.

.. tab-set::

    .. tab-item:: with peer-to-peer

        .. literalinclude:: ../../tools/example_codes/p2p_memory_access.hip
            :start-after: // [sphinx-start]
            :end-before: // [sphinx-end]
            :emphasize-lines: 43-49, 63-67
            :language: cpp

    .. tab-item:: without peer-to-peer

        .. literalinclude:: ../../tools/example_codes/p2p_memory_access_host_staging.hip
            :start-after: // [sphinx-start]
            :end-before: // [sphinx-end]
            :emphasize-lines: 55-57
            :language: cpp

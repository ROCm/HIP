.. meta::
   :description: HIP provides an external resource interoperability API that
                 allows efficient data sharing between HIP's computing power and
                 OpenGL's graphics rendering.
   :keywords: AMD, ROCm, HIP, external, interop, interoperability

*******************************************************************************
External resource interoperability
*******************************************************************************

This feature allows HIP to work with resources -- like memory and semaphores --
created by other APIs. This means resources can be used from APIs like CUDA,
OpenCL and Vulkan within HIP, making it easier to integrate HIP into existing
projects.

To use external resources in HIP, you typically follow these steps:

- Import resources from other APIs using HIP provided functions
- Use external resources as if they were created in HIP
- Destroy the HIP resource object to clean up

Semaphore Functions
===============================================================================

Semaphore functions are essential for synchronization in parallel computing.
These functions facilitate communication and coordination between different
parts of a program or between different programs. By managing semaphores, tasks
are executed in the correct order, and resources are utilized effectively.
Semaphore functions ensure smooth operation, preventing conflicts and
maintaining the integrity of processes; upholding the integrity and performance
of concurrent processes.

External semaphore functions can be used in HIP as described in :ref:`external_resource_interoperability_reference`.

Memory Functions
===============================================================================

HIP external memory functions focus on the efficient sharing and management of
memory resources. These functions enable importing memory created by external
systems, enabling the HIP program to use this memory seamlessly. Memory
functions include mapping memory for effective use and ensuring proper cleanup
to prevent resource leaks. This is critical for performance, particularly in
applications handling large datasets or complex structures such as textures in
graphics. Proper memory management ensures stability and efficient resource
utilization.

Example
===============================================================================

ROCm examples include a
`HIP--Vulkan interoperation example <https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/vulkan_interop>`_
demonstrates how to perform interoperation between HIP and Vulkan.

In this example, a simple HIP kernel is used to compute a sine wave, which is
then rendered to a window as a graphical output using Vulkan. The process
requires several initialization steps, such as setting up a HIP context,
creating a Vulkan instance, and configuring the GPU device and queue. After
these initial steps, the kernel executes the sine wave computation, and Vulkan
continuously updates the window framebuffer to display the computed data until
the window is closed.

The following code converts a Vulkan memory handle to its equivalent HIP
handle. The input ``VkDeviceMemory`` and the created HIP memory represents the
same physical area of GPU memory, through the handles of each respective API.
Writing to the buffer in one API will allow us to read the results through the
other. Note that access to the buffer should be synchronized between the APIs,
for example using queue syncs or semaphores.

.. <!-- spellcheck-disable -->

.. literalinclude:: ../../tools/example_codes/external_interop.hip
   :start-after: // [Sphinx vulkan memory to hip start]
   :end-before: // [Sphinx vulkan memory to hip end]
   :language: cpp

.. <!-- spellcheck-enable -->

The Vulkan semaphore is converted to HIP semaphore shown in the following
example. Signaling on the semaphore in one API will allow the other API to wait
on it, which is how we can guarantee synchronized access to resources in a
cross-API manner.

.. <!-- spellcheck-disable -->

.. literalinclude:: ../../tools/example_codes/external_interop.hip
   :start-after: // [Sphinx semaphore import start]
   :end-before: // [Sphinx semaphore import end]
   :language: cpp

.. <!-- spellcheck-enable -->

When the HIP external memory is exported from Vulkan and imported to HIP, it is
not yet ready for use. The Vulkan handle is shared, allowing for memory sharing
rather than copying during the export process. To actually use the memory, we
need to map it to a pointer so that we may pass it to the kernel so that it can
be read from and written to. The external memory map to HIP in the following
example:

.. <!-- spellcheck-disable -->

.. literalinclude:: ../../tools/example_codes/external_interop.hip
   :start-after: // [Sphinx map external memory start]
   :end-before: // [Sphinx map external memory end]
   :language: cpp

.. <!-- spellcheck-enable -->

Wait for buffer is ready and not under modification at Vulkan side:

.. <!-- spellcheck-disable -->

.. literalinclude:: ../../tools/example_codes/external_interop.hip
   :start-after: // [Sphinx wait semaphore start]
   :end-before: // [Sphinx wait semaphore end]
   :language: cpp

.. <!-- spellcheck-enable -->

The sinewave kernel implementation:

.. <!-- spellcheck-disable -->

.. literalinclude:: ../../tools/example_codes/external_interop.hip
   :start-after: [Sphinx sinewave kernel start]
   :end-before: // [Sphinx sinewave kernel end]
   :language: cpp

.. <!-- spellcheck-enable -->

Signal to Vulkan that we are done with the buffer and that it can proceed with
rendering:

.. <!-- spellcheck-disable -->

.. literalinclude:: ../../tools/example_codes/external_interop.hip
   :start-after: // [Sphinx signal semaphore start]
   :end-before: // [Sphinx signal semaphore end]
   :language: cpp

.. <!-- spellcheck-enable -->
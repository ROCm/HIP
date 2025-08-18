.. meta::
   :description: HIP provides an OpenGL interoperability API that allows
                 efficient data sharing between HIP's computing power and
                 OpenGL's graphics rendering.
   :keywords: AMD, ROCm, HIP, OpenGL, interop, interoperability

*******************************************************************************
OpenGL interoperability
*******************************************************************************

The HIP--OpenGL interoperation involves mapping OpenGL resources, such as
buffers and textures, for HIP to interact with OpenGL. This mapping process
enables HIP to utilize these resources directly, bypassing the need for costly
data transfers between the CPU and GPU. This capability is useful in
applications that require both intensive GPU computation and real-time
visualization.

The graphics resources must be registered using functions like
:cpp:func:`hipGraphicsGLRegisterBuffer` or :cpp:func:`hipGraphicsGLRegisterImage`
then they can be mapped to HIP with :cpp:func:`hipGraphicsMapResources`
function.

After mapping, the :cpp:func:`hipGraphicsResourceGetMappedPointer` or
:cpp:func:`hipGraphicsSubResourceGetMappedArray` functions used to retrieve a
device pointer to the mapped resource, which can then be used in HIP kernels.

Unmapping resources with :cpp:func:`hipGraphicsUnmapResources` after
computations ensure proper resource management.

Example
===============================================================================

ROCm examples have a `HIP--OpenGL interoperation example <https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/opengl_interop>`_,
where a simple HIP kernel is used to simulate a sine wave and rendered to a
window as a grid of triangles using OpenGL. For a working example, there are
multiple initialization steps needed like creating and opening a window,
initializing OpenGL or selecting the OpenGL-capable device. After the
initialization in the example, the kernel simulates the sinewave and updates
the window's framebuffer in a cycle until the window is closed.

.. note::

   The more recent OpenGL functions are loaded with `OpenGL loader <https://github.com/ROCm/rocm-examples/tree/develop/External/glad>`_,
   as these are not loaded by default on all platforms. The use of a custom
   loader is shown in the following example

   .. <!-- spellcheck-disable -->

   .. literalinclude:: ../../tools/example_codes/opengl_interop.hip
      :start-after: // [Sphinx opengl functions load start]
      :end-before: // [Sphinx opengl functions load end]
      :language: cpp

   .. <!-- spellcheck-enable -->

The OpenGL buffer is imported to HIP in the following way:

.. <!-- spellcheck-disable -->

.. literalinclude:: ../../tools/example_codes/opengl_interop.hip
   :start-after: // [Sphinx buffer register and get start]
   :end-before: // [Sphinx buffer register and get end]
   :language: cpp

.. <!-- spellcheck-enable -->

The imported pointer is manipulated in the sinewave kernel as shown in the
following example:

.. <!-- spellcheck-disable -->

.. literalinclude:: ../../tools/example_codes/opengl_interop.hip
   :start-after: /// [Sphinx sinewave kernel start]
   :end-before: /// [Sphinx sinewave kernel end]
   :language: cpp

.. literalinclude:: ../../tools/example_codes/opengl_interop.hip
   :start-after: // [Sphinx buffer use in kernel start]
   :end-before: // [Sphinx buffer use in kernel end]
   :language: cpp

.. <!-- spellcheck-enable -->

The HIP graphics resource that is imported from the OpenGL buffer and is not
needed anymore should be unmapped and unregistered as shown in the following way:

.. <!-- spellcheck-disable -->

.. literalinclude:: ../../tools/example_codes/opengl_interop.hip
   :start-after: // [Sphinx unregister start]
   :end-before: // [Sphinx unregister end]
   :language: cpp

.. <!-- spellcheck-enable -->

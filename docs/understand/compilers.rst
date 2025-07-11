.. meta::
  :description: Compilation workflow of the HIP compilers.
  :keywords: AMD, ROCm, HIP, CUDA, HIP runtime API

.. _hip_compilers:

********************************************************************************
HIP compilers
********************************************************************************

ROCm provides the compiler tools used to compile HIP applications for use on AMD GPUs. 
The compilers set up the default libraries and include paths for the HIP and ROCm
libraries and some needed environment variables. For more information, see the
:doc:`ROCm compiler reference <llvm-project:reference/rocmcc>`.

Compilation workflow
================================================================================

HIP provides a flexible compilation workflow that supports both offline
compilation and runtime or just-in-time (JIT) compilation. Each approach has
advantages depending on the use case, target architecture, and performance
needs.

The offline compilation is ideal for production environments, where the
performance is critical and the target GPU architecture is known in advance.

The runtime compilation is useful in development environments or when
distributing software that must run on a wide range of hardware without the
knowledge of the GPU in advance. It provides flexibility at the cost of some
performance overhead.

Offline compilation
--------------------------------------------------------------------------------

Offline compilation is performed in two steps: host and  device code
compilation. 

- Host-code compilation: On the host side, ``amdclang++`` or ``hipcc`` can
  compile the host code in one step without other C++ compilers. 

- Device-code compilation: The compiled device code is embedded into the
  host object file. Depending on the platform, the device code can be compiled
  into assembly or binary. 

For an example on how to compile HIP from the command line, see :ref:`SAXPY
tutorial <compiling_on_the_command_line>` .

Runtime compilation
--------------------------------------------------------------------------------

HIP allows you to compile kernels at runtime using the ``hiprtc*`` API. Kernels
are stored as a text string, which is passed to HIPRTC alongside options to
guide the compilation.

For more information, see :doc:`HIP runtime compiler <../how-to/hip_rtc>`.

Static libraries
================================================================================

Both ``amdclang++`` and ``hipcc`` support generating two types of static libraries.

- The first type of static library only exports and launches host functions
  within the same library and not the device functions. This library type offers
  the ability to link with another compiler such as ``gcc``. Additionally,
  this library type contains host objects with device code embedded as fat
  binaries. This library type is generated using the flag ``--emit-static-lib``:

  .. code-block:: shell

    amdclang++ hipOptLibrary.cpp --emit-static-lib -fPIC -o libHipOptLibrary.a
    gcc test.cpp -L. -lhipOptLibrary -L/path/to/hip/lib -lamdhip64 -o test.out

- The second type of static library exports device functions to be linked by
  other code objects by using ``amdclang++`` or ``hipcc`` as the linker. This library type
  contains relocatable device objects and is generated using ``ar``:

  .. code-block:: shell

    hipcc hipDevice.cpp -c -fgpu-rdc -o hipDevice.o
    ar rcsD libHipDevice.a hipDevice.o
    hipcc libHipDevice.a test.cpp -fgpu-rdc -o test.out

Examples of this can be found in `rocm-examples <https://github.com/ROCm/rocm-examples>`_ under
`static host libraries <https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/static_host_library>`_
or `static device libraries <https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/static_device_library>`_.

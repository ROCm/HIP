.. meta::
  :description: Compilation workflow of the HIP compilers.
  :keywords: AMD, ROCm, HIP, CUDA, HIP runtime API

.. _hip_compilers:

********************************************************************************
HIP compilers
********************************************************************************

ROCm provides the compiler driver ``hipcc``, that can be used on AMD ROCm and
NVIDIA CUDA platforms.

On ROCm, ``hipcc`` takes care of the following:

- Setting the default library and include paths for HIP
- Setting some environment variables
- Invoking the appropriate compiler - ``amdclang++``

On NVIDIA CUDA platform, ``hipcc`` takes care of invoking compiler ``nvcc``.
``amdclang++`` is based on the ``clang++`` compiler. For more
details, see the :doc:`llvm project<llvm-project:index>`.

HIP compilation workflow
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

The HIP code compilation is performed in two stages: host and  device code
compilation stage.

- Device-code compilation stage: The compiled device code is embedded into the
  host object file. Depending on the platform, the device code can be compiled
  into assembly or binary. ``nvcc`` and ``amdclang++`` target different
  architectures and use different code object formats. ``nvcc`` uses the binary
  ``cubin`` or the assembly PTX files, while the ``amdclang++`` path is the
  binary ``hsaco`` format. On CUDA platforms, the driver compiles the PTX files
  to executable code during runtime.

- Host-code compilation stage: On the host side, ``hipcc`` or ``amdclang++`` can
  compile the host code in one step without other C++ compilers. On the other
  hand, ``nvcc`` only replaces the ``<<<...>>>`` kernel launch syntax with the
  appropriate CUDA runtime function call and the modified host code is passed to
  the default host compiler.

For an example on how to compile HIP from the command line, see :ref:`SAXPY
tutorial<compiling_on_the_command_line>` .

Runtime compilation
--------------------------------------------------------------------------------

HIP allows you to compile kernels at runtime using the ``hiprtc*`` API. Kernels
are stored as a text string, which is passed to HIPRTC alongside options to
guide the compilation.

For more details, see
:doc:`HIP runtime compiler <../how-to/hip_rtc>`.

Static libraries
================================================================================

``hipcc`` supports generating two types of static libraries.

- The first type of static library only exports and launches host functions
  within the same library and not the device functions. This library type offers
  the ability to link with a non-hipcc compiler such as ``gcc``. Additionally,
  this library type contains host objects with device code embedded as fat
  binaries. This library type is generated using the flag ``--emit-static-lib``:

  .. code-block:: shell

    hipcc hipOptLibrary.cpp --emit-static-lib -fPIC -o libHipOptLibrary.a
    gcc test.cpp -L. -lhipOptLibrary -L/path/to/hip/lib -lamdhip64 -o test.out

- The second type of static library exports device functions to be linked by
  other code objects by using ``hipcc`` as the linker. This library type
  contains relocatable device objects and is generated using ``ar``:

  .. code-block:: shell

    hipcc hipDevice.cpp -c -fgpu-rdc -o hipDevice.o
    ar rcsD libHipDevice.a hipDevice.o
    hipcc libHipDevice.a test.cpp -fgpu-rdc -o test.out

For more information, see `HIP samples host functions <https://github.com/ROCm/hip-tests/tree/develop/samples/2_Cookbook/15_static_library/host_functions>`_
and `device functions <https://github.com/ROCm/hip-tests/tree/develop/samples/2_Cookbook/15_static_library/device_functions>`_.

.. meta::
  :description: This chapter describes the compilation workflow of the HIP
                compilers.
  :keywords: AMD, ROCm, HIP, CUDA, HIP runtime API

.. _hip_compilers:

********************************************************************************
HIP compilers
********************************************************************************

ROCm provides the compiler driver ``hipcc``, that can be used on AMD and NVIDIA
platforms. ``hipcc`` takes care of setting the default library and include paths
for HIP, as well as some environment variables, and takes care of invoking the
appropriate compiler - ``amdclang++`` on AMD platforms and ``nvcc`` on NVIDIA
platforms. ``amdclang++`` is based on the ``clang++`` compiler. For further 
details, check :doc:`the llvm project<llvm-project:index>`.

HIP compilation workflow
================================================================================

Offline compilation
--------------------------------------------------------------------------------

The compilation of HIP code is separated into a host- and a device-code
compilation stage.

The compiled device code is embedded into the host object file. Depending on the
platform, the device code can be compiled into assembly or binary. ``nvcc`` and 
``amdclang++`` target different architectures and use different code object
formats: ``nvcc`` uses the binary ``cubin`` or the assembly ``PTX`` files, while
the ``amdclang++`` path is the binary ``hsaco`` format. On NVIDIA platforms the
driver takes care of compiling the PTX files to executable code during runtime.

On the host side ``nvcc`` only replaces the ``<<<...>>>`` kernel launch syntax
with the appropriate CUDA runtime function call and the modified host code is
passed to the default host compiler. ``hipcc`` or ``amdclang++`` can compile the
host code in one step without other C++ compilers.

An example for how to compile HIP from the command line can be found in the
:ref:`SAXPY tutorial<compiling_on_the_command_line>` .

Runtime compilation
--------------------------------------------------------------------------------

HIP lets you compile kernels at runtime with the ``hiprtc*`` API. Kernels are
stored as a text string that are then passed to HIPRTC alongside options to
guide the compilation.

For further details, check the
:doc:`how-to section for the HIP runtime compilation<../how-to/hip_rtc>`.

Static Libraries
================================================================================

``hipcc`` supports generating two types of static libraries. The first type of 
static library does not export device functions, and only exports and launches 
host functions within the same library. The advantage of this type is the 
ability to link with a non-hipcc compiler such as gcc. The second type exports
device functions to be linked by other code objects. However, this requires
using ``hipcc`` as the linker.

In addition, the first type of library contains host objects with device code
embedded as fat binaries. It is generated using the flag ``--emit-static-lib``.
The second type of library contains relocatable device objects and is generated
using ``ar``.

Here is an example to create and use static libraries:

* Type 1 using `--emit-static-lib`:

    .. code-block:: cpp
    
      hipcc hipOptLibrary.cpp --emit-static-lib -fPIC -o libHipOptLibrary.a
      gcc test.cpp -L. -lhipOptLibrary -L/path/to/hip/lib -lamdhip64 -o test.out

* Type 2 using system `ar`:

    .. code-block:: cpp
      
      hipcc hipDevice.cpp -c -fgpu-rdc -o hipDevice.o
      ar rcsD libHipDevice.a hipDevice.o
      hipcc libHipDevice.a test.cpp -fgpu-rdc -o test.out

For more information, please see `HIP samples host functions <https://github.com/ROCm/hip-tests/tree/develop/samples/2_Cookbook/15_static_library/host_functions>`_
and `device_functions <https://github.com/ROCm/hip-tests/tree/develop/samples/2_Cookbook/15_static_library/device_functions>`_.

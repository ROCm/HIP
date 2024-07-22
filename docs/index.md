# HIP documentation

The Heterogeneous-computing Interface for Portability (HIP) API is a C++ runtime
API and kernel language that lets developers create portable applications running 
in heterogeneous systems, using CPUs and AMD GPUs and NVIDIA GPUs from a single source code. 
HIP provides a simple marshalling language to access either the AMD ROCM back-end, 
or NVIDIA CUDA back-end, to build and run application kernels. For more information, 
see [Introduction to HIP](./understand/introduction_to_hip).

For HIP supported AMD GPUs on multiple operating systems, see:

* [Linux system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)
* [Microsoft Windows system requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html#windows-supported-gpus)

The HIP documentation is organized as follows:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Install

* [Installing HIP](./install/install)
* [Building HIP from source](./install/build)

:::

:::{grid-item-card} Conceptual

* {doc}`./understand/programming_model`
* {doc}`./understand/hardware_implementation`
* {doc}`./understand/amd_clr`

:::

:::{grid-item-card} How to

* [Programming Manual](./how-to/programming_manual)
* [HIP Porting Guide](./how-to/hip_porting_guide)
* [HIP Porting: Driver API Guide](./how-to/hip_porting_driver_api)
* {doc}`./how-to/hip_rtc`
* {doc}`./how-to/performance_guidelines`
* [Debugging with HIP](./how-to/debugging)
* {doc}`./how-to/logging`
* [Unified Memory](./how-to/unified_memory)
* {doc}`./how-to/faq`

:::

:::{grid-item-card} Reference

* {doc}`/doxygen/html/index`
* [C++ Language Extensions](./reference/cpp_language_extensions)
* [Comparing Syntax for Different APIs](./reference/terms)
* [HSA Runtime API for ROCm](./reference/virtual_rocr)
* [HIP Managed Memory Allocation API](./reference/unified_memory_reference)
* [List of deprecated APIs](./reference/deprecated_api_list)

:::

:::{grid-item-card} Tutorial

* [HIP basic examples](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic)
* [HIP examples](https://github.com/ROCm/HIP-Examples)
* [HIP test samples](https://github.com/ROCm/hip-tests/tree/develop/samples)
* [SAXPY tutorial](./tutorial/saxpy)

:::

::::

Known issues are listed on the [HIP GitHub repository](https://github.com/ROCm/HIP/issues).

To contribute features or functions to the HIP project, refer to [Contributing to HIP](https://github.com/ROCm/HIP/blob/develop/CONTRIBUTING.md).
To contribute to the documentation, refer to {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` page.

You can find licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.

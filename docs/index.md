# HIP documentation

The Heterogeneous-computing Interface for Portability (HIP) API is a C++ runtime
API and kernel language that lets developers create portable applications for AMD
and NVIDIA GPUs from single source code.

For HIP supported AMD GPUs on multiple operating systems, see:

* [Linux system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)
* [Microsoft Windows system requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html#windows-supported-gpus)

The CUDA enabled NVIDIA GPUs are supported by HIP. For more information, see [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

On the AMD ROCm platform, HIP provides header files and runtime library built on top of HIP-Clang compiler in the repository [Common Language Runtime (CLR)](./understand/amd_clr), which contains source codes for AMD's compute languages runtimes as follows,

On non-AMD platforms, like NVIDIA, HIP provides header files required to support non-AMD specific back-end implementation in the repository ['hipother'](https://github.com/ROCm/hipother), which translates from the HIP runtime APIs to CUDA runtime APIs.

## Overview

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Install

* [Installing HIP](./install/install)
* [Building HIP from source](./install/build)

:::

:::{grid-item-card} Conceptual

* {doc}`./understand/programming_model`
* {doc}`./understand/programming_model_reference`
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
* [FP8 numbers in HIP](./reference/fp8_numbers)

:::

:::{grid-item-card} Tutorial

* [HIP basic examples](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic)
* [HIP examples](https://github.com/ROCm/HIP-Examples)
* [HIP test samples](https://github.com/ROCm/hip-tests/tree/develop/samples)
* [SAXPY tutorial](./tutorial/saxpy)
* [Reduction tutorial](./tutorial/reduction)

:::

::::

Known issues are listed on the [HIP GitHub repository](https://github.com/ROCm/HIP/issues).

To contribute features or functions to the HIP project, refer to [Contributing to HIP](https://github.com/ROCm/HIP/blob/develop/CONTRIBUTING.md).
To contribute to the documentation, refer to {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` page.

You can find licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.

# HIP documentation

The Heterogeneous-computing Interface for Portability (HIP) is a C++ runtime API and kernel language that lets you create portable applications for AMD and NVIDIA GPUs from a single source code. For more information, see [What is HIP?](./what_is_hip)

Installation instructions are available from:

* [Installing HIP](./install/install)
* [Building HIP from source](./install/build)

HIP enabled GPUs:  

* [Supported AMD GPUs on Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-gpus)
* [Supported AMD GPUs on Windows](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html#windows-supported-gpus)
* [Supported NVIDIA GPUs](https://developer.nvidia.com/cuda-gpus)

The HIP documentation is organized into the following categories:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Conceptual

* {doc}`./understand/programming_model`
* {doc}`./understand/hardware_implementation`
* {doc}`./understand/amd_clr`
* {doc}`./understand/texture_fetching`

:::

:::{grid-item-card} How to

* [Programming manual](./how-to/programming_manual)
* [HIP porting guide](./how-to/hip_porting_guide)
* [HIP porting: driver API guide](./how-to/hip_porting_driver_api)
* {doc}`./how-to/hip_rtc`
* {doc}`./how-to/performance_guidelines`
* [Debugging with HIP](./how-to/debugging)
* {doc}`./how-to/logging`
* [Unified memory](./how-to/unified_memory)
* [Virtual memory](./how-to/virtual_memory)
* {doc}`./how-to/stream_ordered_allocator`
* [Cooperative groups](./how-to/cooperative_groups)
* [HIP graphs](./how-to/hipgraph)
* {doc}`./how-to/faq`

:::

:::{grid-item-card} Reference

* [HIP runtime API](./reference/hip_runtime_api_reference)
  * [Modules](./reference/hip_runtime_api/modules)
  * [Global defines, enums, structs and files](./reference/hip_runtime_api/global_defines_enums_structs_files)
* [HSA runtime API for ROCm](./reference/virtual_rocr)
* [C++ language extensions](./reference/cpp_language_extensions)
* [C++ language support](./reference/cpp_language_support)
* [HIP math API](./reference/math_api)
* [Comparing syntax for different APIs](./reference/terms)
* [List of deprecated APIs](./reference/deprecated_api_list)
* [FP8 numbers in HIP](./reference/fp8_numbers)

:::

:::{grid-item-card} Tutorial

* [HIP basic examples](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic)
* [HIP examples](https://github.com/ROCm/HIP-Examples)
* [HIP test samples](https://github.com/ROCm/hip-tests/tree/develop/samples)
* [SAXPY tutorial](./tutorial/saxpy)
* [Reduction tutorial](./tutorial/reduction)
* [Cooperative groups tutorial](./tutorial/cooperative_groups_tutorial)

:::

::::

Known issues are listed on the [HIP GitHub repository](https://github.com/ROCm/HIP/issues).

To contribute features or functions to the HIP project, refer to [Contributing to HIP](https://github.com/ROCm/HIP/blob/develop/CONTRIBUTING.md).
To contribute to the documentation, refer to {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` page.

You can find licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.

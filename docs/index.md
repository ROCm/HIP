<head>
  <meta charset="UTF-8">
  <meta name="description" content="HIP documentation and programming guide.">
  <meta name="keywords" content="HIP, Heterogeneous-computing Interface for Portability, HIP programming guide">
</head>

# HIP documentation

The Heterogeneous-computing Interface for Portability (HIP) is a C++ runtime API
and kernel language that lets you create portable applications for AMD and
NVIDIA GPUs from a single source code. For more information, see [What is HIP?](./what_is_hip)

```{note}
HIP API 7.0 introduces changes to make it align more closely with NVIDIA CUDA.
These changes are incompatible with prior releases, and might require recompiling
existing HIP applications for use with the ROCm 7.0 release. For more information,
see [HIP API 7.0 changes](./hip-7-changes).
```

Installation instructions are available from:

* [Installing HIP](./install/install)
* [Building HIP from source](./install/build)

The HIP documentation is organized into the following categories:

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Programming guide

* {doc}`./understand/programming_model`
* {doc}`./understand/hardware_implementation`
* {doc}`./understand/compilers`
* {doc}`./how-to/performance_guidelines`
* [Debugging with HIP](./how-to/debugging)
* {doc}`./how-to/logging`
* {doc}`./how-to/hip_runtime_api`
* {doc}`./how-to/hip_cpp_language_extensions`
* {doc}`./how-to/kernel_language_cpp_support`
* [HIP porting guide](./how-to/hip_porting_guide)
* [HIP porting: driver API guide](./how-to/hip_porting_driver_api)
* {doc}`./how-to/hip_rtc`
* {doc}`./understand/amd_clr`

:::

:::{grid-item-card} Reference

* [HIP runtime API](./reference/hip_runtime_api_reference)
* [HIP math API](./reference/math_api)
* [HIP complex math API](./reference/complex_math_api)
* [HIP environment variables](./reference/env_variables)
* [HIP error codes](./reference/error_codes)
* [CUDA to HIP API Function Comparison](./reference/api_syntax)
* [List of deprecated APIs](./reference/deprecated_api_list)
* [Low Precision Floating Point Types](./reference/low_fp_types)
* {doc}`./reference/hardware_features`

:::

:::{grid-item-card} Tutorial

* [HIP basic examples](https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic)
* [HIP examples](https://github.com/ROCm/rocm-examples)
* [SAXPY tutorial](./tutorial/saxpy)
* [Reduction tutorial](./tutorial/reduction)
* [Cooperative groups tutorial](./tutorial/cooperative_groups_tutorial)

:::

::::

Known issues are listed on the [HIP GitHub repository](https://github.com/ROCm/HIP/issues).

To contribute features or functions to the HIP project, refer to [Contributing to HIP](https://github.com/ROCm/HIP/blob/develop/CONTRIBUTING.md).
To contribute to the documentation, refer to {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` page.

You can find licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.

# HIP documentation

HIP is a C++ runtime API and kernel language that lets developers create
portable applications for AMD and NVIDIA GPUs from single source code.

## Overview

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Installation

* [Install HIP](./install/install)
* [Build HIP from source](./install/build)

:::

:::{grid-item-card} Conceptual

* {doc}`./understand/programming_model`
* {doc}`./understand/programming_model_reference`
* {doc}`./understand/hardware_implementation`

:::

:::{grid-item-card} How-to

* [Programming Manual](./how-to/programming_manual)
* [HIP Porting Guide](./how-to/hip_porting_guide)
* [HIP Porting: Driver API Guide](./how-to/hip_porting_driver_api)
* {doc}`./how-to/hip_rtc`
* {doc}`./how-to/performance_guidelines`
* [Debugging with HIP](./how-to/debugging)
* {doc}`./how-to/logging`
* {doc}`./how-to/faq`

:::

:::{grid-item-card} Reference

* {doc}`/doxygen/html/index`
* [C++ language extensions](./reference/kernel_language)
* [Comparing Syntax for different APIs](./reference/terms)
* [HSA Runtime API for ROCm](./reference/virtual_rocr)
* [List of deprecated APIs](./reference/deprecated_api_list)

:::

:::{grid-item-card} Tutorial

* [HIP examples](https://github.com/ROCm/HIP-Examples)
* [HIP test samples](https://github.com/ROCm/hip-tests/tree/develop/samples)

:::

::::

Known issues are listed on the [HIP GitHub repository](https://github.com/ROCm/HIP/issues).

To contribute features or functions to the HIP project, refer to [Contributing to HIP](https://github.com/ROCm/HIP/blob/develop/CONTRIBUTING.md).
To contribute to the documentation, refer to {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` page. 

You can find licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.

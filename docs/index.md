# HIP documentation

HIP is a C++ runtime API and kernel language that allows developers to create
portable applications for AMD and NVIDIA GPUs from single source code.

## Overview

::::{grid} 1 1 2 2
:gutter: 1

:::{grid-item-card} Installation

* [Install HIP](./install/install)
* [Build HIP from source](./install/build)

:::

:::{grid-item-card} Conceptual

* {doc}`./understand/programming_model`
* {doc}`./understand/programming_model_reference`
* [Glossary](./understand/glossary)

:::

:::{grid-item-card} How-to

* [Programming Manual](./how-to/hip-rtc)
* [HIP Porting Guide](./how-to/hip_porting_guide)
* [HIP Porting: Driver API Guide](./how-to/hip_porting_driver_api)
* {doc}`./how-to/hip_rtc`
* [Debugging with HIP](./how-to/debugging)
* {doc}`./how-to/logging`
* {doc}`./how-to/faq`

:::

:::{grid-item-card} Reference

* {doc}`/doxygen/html/index`
* [C++ kernel language](./reference/kernel_language)
* {doc}`./reference/math_api`
* [Comparing Syntax for different APIs](./reference/terms)
* [List of deprecated APIs](./reference/deprecated_api_list)

:::

::::

Known issues are listed on the [HIP GitHub repository](https://github.com/ROCm/HIP/issues).

To contribute features or functions to the HIP project, refer to the [Contributor guidelines](https://github.com/ROCm/HIP/CONTRIBUTING.md).
To contribute to the documentation, refer to {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` page. 

You can find licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.

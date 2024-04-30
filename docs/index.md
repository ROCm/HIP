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

:::{grid-item-card} Reference

* {doc}`/reference/programming_model`
* {doc}`/doxygen/html/index`
* [Deprecated APIs](./reference/deprecated_api_list)

:::

:::{grid-item-card} How-to

* [Debug with HIP](./how-to/debugging)
* [Generate logs](./how-to/logging)
* [Performance Guidelines](./how-to/performance_guidelines)

:::

:::{grid-item-card} Conceptual

* {doc}`/understand/programming_model`

:::

::::

## Legacy documentation

These documents have not yet been ported over to the Diátaxis framework.

::::{grid} 1 1 2 2
:gutter: 1

:::{grid-item-card} Reference

* [C++ kernel language](./old/reference/kernel_language)
* [Table Comparing Syntax for Different Compute APIs](./old/reference/terms)

:::

:::{grid-item-card} User Guide

* [HIP Porting Guide](./old/user_guide/hip_porting_guide)
* [HIP Porting Driver API Guide](./old/user_guide/hip_porting_driver_api)
* [HIP RTC Programming Guide](./old/user_guide/hip_rtc.md)
* [HIP Programming Manual](./old/user_guide/programming_manual.md)
* [Frequently asked questions](./old/user_guide/faq.md)

:::

::::

We welcome collaboration! If you’d like to contribute to our documentation, you can find instructions
on our {doc}`Contribute to ROCm docs <rocm:contribute/contributing>` page. Known issues are listed on
[GitHub](https://github.com/ROCm/HIP/issues).

If you want to contribute to the HIP project, refer to our [Contributor guidelines](https://github.com/ROCm/HIP/CONTRIBUTING.md).

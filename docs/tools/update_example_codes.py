#
# Copyright (C) Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE COPYRIGHT HOLDER(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
# AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import urllib.request

urllib.request.urlretrieve(
     "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Basic/opengl_interop/main.hip",
        "docs/tools/example_codes/opengl_interop.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Basic/vulkan_interop/main.hip",
    "docs/tools/example_codes/external_interop.hip"
)

# HIP-C%2B%2B-Language-Extensions
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/calling_global_functions/main.hip",
    "docs/tools/example_codes/calling_global_functions.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/extern_shared_memory/main.hip",
    "docs/tools/example_codes/extern_shared_memory.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/launch_bounds/main.hip",
    "docs/tools/example_codes/launch_bounds.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/set_constant_memory/main.hip",
    "docs/tools/example_codes/set_constant_memory.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/template_warp_size_reduction/main.hip",
    "docs/tools/example_codes/template_warp_size_reduction.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/timer/main.hip",
    "docs/tools/example_codes/timer.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/warp_size_reduction/main.hip",
    "docs/tools/example_codes/warp_size_reduction.hip"
)

# HIP-Porting-Guide
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-Porting-Guide/device_code_feature_identification/main.hip",
    "docs/tools/example_codes/device_code_feature_identification.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-Porting-Guide/host_code_feature_identification/main.cpp",
    "docs/tools/example_codes/host_code_feature_identification.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-Porting-Guide/identifying_compilation_target_platform/main.cpp",
    "docs/tools/example_codes/identifying_compilation_target_platform.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/HIP-Porting-Guide/identifying_host_device_compilation_pass/main.hip",
    "docs/tools/example_codes/identifying_host_device_compilation_pass.hip"
)

# Introduction-to-the-HIP-Programming-Model
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Introduction-to-the-HIP-Programming-Model/add_kernel/main.hip",
    "docs/tools/example_codes/add_kernel.hip"
)

# Porting-CUDA-Driver-API
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-Driver-API/load_module/main.cpp",
    "docs/tools/example_codes/load_module.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-Driver-API/load_module_ex/main.cpp",
    "docs/tools/example_codes/load_module_ex.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-Driver-API/load_module_ex_cuda/main.cpp",
    "docs/tools/example_codes/load_module_ex_cuda.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-Driver-API/per_thread_default_stream/main.cpp",
    "docs/tools/example_codes/per_thread_default_stream.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Porting-CUDA-Driver-API/pointer_memory_type/main.cpp",
    "docs/tools/example_codes/pointer_memory_type.cpp"
)

# Programming-for-HIP-Runtime-Compiler
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/compilation_apis/main.cpp",
    "docs/tools/example_codes/compilation_apis.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/linker_apis/main.cpp",
    "docs/tools/example_codes/linker_apis.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/linker_apis_file/main.cpp",
    "docs/tools/example_codes/linker_apis_file.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/linker_apis_options/main.cpp",
    "docs/tools/example_codes/linker_apis_options.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/lowered_names/main.cpp",
    "docs/tools/example_codes/lowered_names.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/rtc_error_handling/main.cpp",
    "docs/tools/example_codes/rtc_error_handling.cpp"
)

# Using-HIP-Runtime-API
# Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution/async_kernel_execution/main.hip",
    "docs/tools/example_codes/async_kernel_execution.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution/event_based_synchronization/main.hip",
    "docs/tools/example_codes/event_based_synchronization.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution/sequential_kernel_execution/main.hip",
    "docs/tools/example_codes/sequential_kernel_execution.hip"
)

# Using-HIP-Runtime-API / Call-Stack
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Call-Stack/call_stack_management/main.cpp",
    "docs/tools/example_codes/call_stack_management.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Call-Stack/device_recursion/main.hip",
    "docs/tools/example_codes/device_recursion.hip"
)

# Using-HIP-Runtime-API / Error-Handling
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Error-Handling/error_handling/main.hip",
    "docs/tools/example_codes/error_handling.hip"
)

# Using-HIP-Runtime-API / HIP-Graphs
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/HIP-Graphs/graph_capture/main.hip",
    "docs/tools/example_codes/graph_capture.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/HIP-Graphs/graph_creation/main.hip",
    "docs/tools/example_codes/graph_creation.hip"
)

# Using-HIP-Runtime-API / Initialization
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Initialization/simple_device_query/main.cpp",
    "docs/tools/example_codes/simple_device_query.cpp"
)

# Using-HIP-Runtime-API / Memory-Management / Device-Memory
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/constant_memory/main.hip",
    "docs/tools/example_codes/constant_memory_device.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/dynamic_shared_memory/main.hip",
    "docs/tools/example_codes/dynamic_shared_memory_device.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/explicit_copy/main.cpp",
    "docs/tools/example_codes/explicit_copy.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/kernel_memory_allocation/main.hip",
    "docs/tools/example_codes/kernel_memory_allocation.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/static_shared_memory/main.hip",
    "docs/tools/example_codes/static_shared_memory_device.hip"
)

# Using-HIP-Runtime-API / Memory-Management / Host-Memory
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Host-Memory/pageable_host_memory/main.cpp",
    "docs/tools/example_codes/pageable_host_memory.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Host-Memory/pinned_host_memory/main.cpp",
    "docs/tools/example_codes/pinned_host_memory.cpp"
)

# Using-HIP-Runtime-API / Memory-Management / SOMA
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/stream_ordered_memory_allocation/main.hip",
    "docs/tools/example_codes/stream_ordered_memory_allocation.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/ordinary_memory_allocation/main.hip",
    "docs/tools/example_codes/ordinary_memory_allocation.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool/main.hip",
    "docs/tools/example_codes/memory_pool.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool_resource_usage_statistics/main.cpp",
    "docs/tools/example_codes/memory_pool_resource_usage_statistics.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool_threshold/main.hip",
    "docs/tools/example_codes/memory_pool_threshold.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool_trim/main.cpp",
    "docs/tools/example_codes/memory_pool_trim.cpp"
)

# Using-HIP-Runtime-API / Memory-Management / Unified-Memory-Management
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/data_prefetching/main.hip",
    "docs/tools/example_codes/data_prefetching.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/dynamic_unified_memory/main.hip",
    "docs/tools/example_codes/dynamic_unified_memory.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/explicit_memory/main.hip",
    "docs/tools/example_codes/explicit_memory.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/memory_range_attributes/main.hip",
    "docs/tools/example_codes/memory_range_attributes.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/standard_unified_memory/main.hip",
    "docs/tools/example_codes/standard_unified_memory.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/static_unified_memory/main.hip",
    "docs/tools/example_codes/static_unified_memory.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/unified_memory_advice/main.hip",
    "docs/tools/example_codes/unified_memory_advice.hip"
)

# Using-HIP-Runtime-API / Multi-Device-Management
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/device_enumeration/main.cpp",
    "docs/tools/example_codes/device_enumeration.cpp"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/device_selection/main.hip",
    "docs/tools/example_codes/device_selection.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/multi_device_synchronization/main.hip",
    "docs/tools/example_codes/multi_device_synchronization.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/p2p_memory_access/main.hip",
    "docs/tools/example_codes/p2p_memory_access.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/p2p_memory_access_host_staging/main.hip",
    "docs/tools/example_codes/p2p_memory_access_host_staging.hip"
)

# Reference examples from HIP-Doc / Reference

# CUDA-to-HIP-API-Function-Comparison
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Reference/CUDA-to-HIP-API-Function-Comparison/block_reduction/main.cu",
    "docs/tools/example_codes/block_reduction.cu"
)

# HIP-Complex-Math-API
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Reference/HIP-Complex-Math-API/complex_math/main.hip",
    "docs/tools/example_codes/complex_math.hip"
)

# HIP-Math-API
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Reference/HIP-Math-API/math/main.hip",
    "docs/tools/example_codes/math.hip"
)

# Low-Precision-Floating-Point-Types
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Reference/Low-Precision-Floating-Point-Types/low_precision_float_fp8/main.hip",
    "docs/tools/example_codes/low_precision_float_fp8.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Reference/Low-Precision-Floating-Point-Types/low_precision_float_fp16/main.hip",
    "docs/tools/example_codes/low_precision_float_fp16.hip"
)

# Tutorial codes from HIP-Doc / Tutorials

# graph_api
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Tutorials/graph_api/src/main_streams.hip",
    "docs/tools/example_codes/graph_api_tutorial_main_streams.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Tutorials/graph_api/src/main_graph_capture.hip",
    "docs/tools/example_codes/graph_api_tutorial_main_graph_capture.hip"
)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ROCm/rocm-examples/amd-staging/HIP-Doc/Tutorials/graph_api/src/main_graph_creation.hip",
    "docs/tools/example_codes/graph_api_tutorial_main_graph_creation.hip"
)
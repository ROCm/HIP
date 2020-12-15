# hipInfo

Simple tool that prints properties for each device (from hipGetDeviceProperties), and compiler info.
    Properties includes all of the architectural feature flags for each device.

Also demonstrates how to use platform-specific compilation path (testing `__HIP_PLATFORM_AMD__` or `__HIP_PLATFORM_NVIDIA__`)

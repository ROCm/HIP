# HIP applications that test hostcall #

The tests for hostcall are arranged in the following directories:

  * **basic:** Exercise the basic hostcall service by directly invoking
    `__ockl_hostcall_preview()` from the device library.

  * **device:** Exercise `__ockl_hostcall_preview()` from the device
    library, but launch their own host-side consumer. This is useful
    for testing the device implementaiton without relying on the host
    implementation. The tests perform the following non-standard
    actions, which are not supported in a HIP application:
    - Create their own custom hostcall buffer and pass it as an explicit
      argument to the kernel.
    - Launch their own hostcall consumer thread which responds to
      packets in the custom buffer.

    Note that the HIP runtime still launches its own hostcall consumer
    thread, but it remains unused.

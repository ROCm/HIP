.. meta::
   :description: HIP provides a logging mechanism that allows you to trace HIP API and runtime codes
                 when running a HIP application.
   :keywords: AMD, ROCm, HIP, logging

**********************************************************
Logging HIP activity
**********************************************************

HIP provides a logging mechanism that allows you to trace HIP API and runtime codes when running a
HIP application. In addition to being useful to our users/developers, the HIP development team uses
these logs to improve the HIP runtime.

By adjusting the logging settings and logging mask, you can get different types of information for
different functionalities, such as HIP APIs, executed kernels, queue commands, and queue contents.
Refer to the following sections for examples.

.. tip::

  Logging works for the release and debug versions of HIP. If you want to save logging output in a file,
  define the file when running the application via command line. For example:

  .. code-block:: bash

    user@user-test:~/hip/bin$ ./hipinfo > ~/hip_log.txt

Logging level
======================================

HIP logging is disabled by default. You can enable it via the ``AMD_LOG_LEVEL`` environment variable.
The value of this variable controls your logging level. Levels are defined as follows:

.. code-block:: cpp

  enum LogLevel {
    LOG_NONE           = 0,
    LOG_ERROR          = 1,
    LOG_WARNING        = 2,
    LOG_INFO           = 3,
    LOG_DEBUG          = 4,
    LOG_EXTRA_DEBUG    = 5
  };

.. tip::

  You can call a logging function with different logging levels. All information under the value set for
  ``AMD_LOG_LEVEL`` is printed.

Logging mask
======================================

The logging mask is designed to print functionality types when you're running a HIP application.
Once you set ``AMD_LOG_LEVEL``, the logging mask is set as the default value (``0x7FFFFFFF``). You can
change this to any of the valid values:

.. code-block:: cpp

  enum LogMask {
    LOG_API       = 1,      //!< (0x1)     API call
    LOG_CMD       = 2,      //!< (0x2)     Kernel and Copy Commands and Barriers
    LOG_WAIT      = 4,      //!< (0x4)     Synchronization and waiting for commands to finish
    LOG_AQL       = 8,      //!< (0x8)     Decode and display AQL packets
    LOG_QUEUE     = 16,     //!< (0x10)    Queue commands and queue contents
    LOG_SIG       = 32,     //!< (0x20)    Signal creation, allocation, pool
    LOG_LOCK      = 64,     //!< (0x40)    Locks and thread-safety code.
    LOG_KERN      = 128,    //!< (0x80)    Kernel creations and arguments, etc.
    LOG_COPY      = 256,    //!< (0x100)   Copy debug
    LOG_COPY2     = 512,    //!< (0x200)   Detailed copy debug
    LOG_RESOURCE  = 1024,   //!< (0x400)   Resource allocation, performance-impacting events.
    LOG_INIT      = 2048,   //!< (0x800)   Initialization and shutdown
    LOG_MISC      = 4096,   //!< (0x1000)  Misc debug, not yet classified
    LOG_AQL2      = 8192,   //!< (0x2000)  Show raw bytes of AQL packet
    LOG_CODE      = 16384,  //!< (0x4000)  Show code creation debug
    LOG_CMD2      = 32768,  //!< (0x8000)  More detailed command info, including barrier commands
    LOG_LOCATION  = 65536,  //!< (0x10000) Log message location
    LOG_MEM       = 131072, //!< (0x20000) Memory allocation
    LOG_MEM_POOL  = 262144, //!< (0x40000) Memory pool allocation, including memory in graphs
    LOG_TS        = 524288, //!< (0x80000) Timestamp details
    LOG_ALWAYS    = -1      //!< (0xFFFFFFFF) Log always even mask flag is zero
  };

You can also define the logging mask via the ``AMD_LOG_MASK`` environment variable.

Logging command
======================================

You can use the following code to print HIP logging information:

.. code-block:: cpp

  #define ClPrint(level, mask, format, ...)                                       \
    do {                                                                          \
      if (AMD_LOG_LEVEL >= level) {                                               \
        if (AMD_LOG_MASK & mask || mask == amd::LOG_ALWAYS) {                     \
          if (AMD_LOG_MASK & amd::LOG_LOCATION) {                                 \
            amd::log_printf(level, __FILENAME__, __LINE__, format, ##__VA_ARGS__);\
          } else {                                                                \
            amd::log_printf(level, "", 0, format, ##__VA_ARGS__);                 \
          }                                                                       \
        }                                                                         \
      }                                                                           \
    } while (false)


Using HIP code, call the ``ClPrint()`` function with the desired input variables. For example:

.. code-block:: cpp

  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Initializing HSA stack.");


Logging examples
======================================

On **Linux**, you can enable HIP logging and retrieve logging information when you run ``hipinfo``.

.. code-block:: console

  user@user-test:~/hip/bin$ export AMD_LOG_LEVEL=4
  user@user-test:~/hip/bin$ ./hipinfo

  :3:rocdevice.cpp            :453 : 23647210092: Initializing HSA stack.
  :3:comgrctx.cpp             :33  : 23647639336: Loading COMGR library.
  :3:rocdevice.cpp            :203 : 23647687108: Numa select cpu agent[0]=0x13407c0(fine=0x13409a0,coarse=0x1340ad0) for gpu agent=0x1346150
  :4:runtime.cpp              :82  : 23647698669: init
  :3:hip_device_runtime.cpp   :473 : 23647698869: 5617 : [7fad295dd840] hipGetDeviceCount: Returned hipSuccess
  :3:hip_device_runtime.cpp   :502 : 23647698990: 5617 : [7fad295dd840] hipSetDevice ( 0 )
  :3:hip_device_runtime.cpp   :507 : 23647699042: 5617 : [7fad295dd840] hipSetDevice: Returned hipSuccess
  --------------------------------------------------------------------------------
  device#                           0
  :3:hip_device.cpp           :150 : 23647699276: 5617 : [7fad295dd840] hipGetDeviceProperties ( 0x7ffdbe7db730, 0 )
  :3:hip_device.cpp           :237 : 23647699335: 5617 : [7fad295dd840] hipGetDeviceProperties: Returned hipSuccess
  Name:                             Device 7341
  pciBusID:                         3
  pciDeviceID:                      0
  pciDomainID:                      0
  multiProcessorCount:              11
  maxThreadsPerMultiProcessor:      2560
  isMultiGpuBoard:                  0
  clockRate:                        1900 Mhz
  memoryClockRate:                  875 Mhz
  memoryBusWidth:                   0
  clockInstructionRate:             1000 Mhz
  totalGlobalMem:                   7.98 GB
  maxSharedMemoryPerMultiProcessor: 64.00 KB
  totalConstMem:                    8573157376
  sharedMemPerBlock:                64.00 KB
  canMapHostMemory:                 1
  regsPerBlock:                     0
  warpSize:                         32
  l2CacheSize:                      0
  computeMode:                      0
  maxThreadsPerBlock:               1024
  maxThreadsDim.x:                  1024
  maxThreadsDim.y:                  1024
  maxThreadsDim.z:                  1024
  maxGridSize.x:                    2147483647
  maxGridSize.y:                    2147483647
  maxGridSize.z:                    2147483647
  major:                            10
  minor:                            12
  concurrentKernels:                1
  cooperativeLaunch:                0
  cooperativeMultiDeviceLaunch:     0
  arch.hasGlobalInt32Atomics:       1
  ...
  gcnArch:                          1012
  isIntegrated:                     0
  maxTexture1D:                     65536
  maxTexture2D.width:               16384
  maxTexture2D.height:              16384
  maxTexture3D.width:               2048
  maxTexture3D.height:              2048
  maxTexture3D.depth:               2048
  isLargeBar:                       0
  :3:hip_device_runtime.cpp   :471 : 23647701557: 5617 : [7fad295dd840] hipGetDeviceCount ( 0x7ffdbe7db714 )
  :3:hip_device_runtime.cpp   :473 : 23647701608: 5617 : [7fad295dd840] hipGetDeviceCount: Returned hipSuccess
  :3:hip_peer.cpp             :76  : 23647701731: 5617 : [7fad295dd840] hipDeviceCanAccessPeer ( 0x7ffdbe7db728, 0, 0 )
  :3:hip_peer.cpp             :60  : 23647701784: 5617 : [7fad295dd840] canAccessPeer: Returned hipSuccess
  :3:hip_peer.cpp             :77  : 23647701831: 5617 : [7fad295dd840] hipDeviceCanAccessPeer: Returned hipSuccess
  peers:
  :3:hip_peer.cpp             :76  : 23647701921: 5617 : [7fad295dd840] hipDeviceCanAccessPeer ( 0x7ffdbe7db728, 0, 0 )
  :3:hip_peer.cpp             :60  : 23647701965: 5617 : [7fad295dd840] canAccessPeer: Returned hipSuccess
  :3:hip_peer.cpp             :77  : 23647701998: 5617 : [7fad295dd840] hipDeviceCanAccessPeer: Returned hipSuccess
  non-peers:                        device#0

  :3:hip_memory.cpp           :345 : 23647702191: 5617 : [7fad295dd840] hipMemGetInfo ( 0x7ffdbe7db718, 0x7ffdbe7db720 )
  :3:hip_memory.cpp           :360 : 23647702243: 5617 : [7fad295dd840] hipMemGetInfo: Returned hipSuccess
  memInfo.total:                    7.98 GB
  memInfo.free:                     7.98 GB (100%)


On **Windows**, you can set ``AMD_LOG_LEVEL`` via environment variable from the advanced system
settings or the command prompt (when run as administrator). The following example shows debug log
information when calling the backend runtime.

.. code-block:: bash

  C:\hip\bin>set AMD_LOG_LEVEL=4
  C:\hip\bin>hipinfo
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\comgrctx.cpp:33  : 605413686305 us: 29864: [tid:0x9298] Loading COMGR library.
  :4:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\platform\runtime.cpp:83  : 605413869411 us: 29864: [tid:0x9298] init
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_context.cpp:47  : 605413869502 us: 29864: [tid:0x9298] Direct Dispatch: 0
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.cpp:543 : 605413870553 us: 29864: [tid:0x9298] hipGetDeviceCount: Returned hipSuccess :
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.cpp:556 : 605413870631 us: 29864: [tid:0x9298] ←[32m hipSetDevice ( 0 ) ←[0m
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.cpp:561 : 605413870848 us: 29864: [tid:0x9298] hipSetDevice: Returned hipSuccess :
  --------------------------------------------------------------------------------
  device#                           0
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device.cpp:346 : 605413871623 us: 29864: [tid:0x9298] ←[32m hipGetDeviceProperties ( 0000008AEBEFF8C8, 0 ) ←[0m
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device.cpp:348 : 605413871695 us: 29864: [tid:0x9298] hipGetDeviceProperties: Returned hipSuccess :
  Name:                             AMD Radeon(TM) Graphics
  pciBusID:                         3
  pciDeviceID:                      0
  pciDomainID:                      0
  multiProcessorCount:              7
  maxThreadsPerMultiProcessor:      2560
  isMultiGpuBoard:                  0
  clockRate:                        1600 Mhz
  memoryClockRate:                  1333 Mhz
  memoryBusWidth:                   0
  totalGlobalMem:                   12.06 GB
  totalConstMem:                    2147483647
  sharedMemPerBlock:                64.00 KB
  ...
  gcnArchName:                      gfx90c:xnack-
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.cpp:541 : 605413924779 us: 29864: [tid:0x9298] ←[32m hipGetDeviceCount ( 0000008AEBEFF8A4 ) ←[0m
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_device_runtime.cpp:543 : 605413925075 us: 29864: [tid:0x9298] hipGetDeviceCount: Returned hipSuccess :
  peers:                            :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_peer.cpp:176 : 605413928643 us: 29864: [tid:0x9298] ←[32m hipDeviceCanAccessPeer ( 0000008AEBEFF890, 0, 0 ) ←[0m
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_peer.cpp:177 : 605413928743 us: 29864: [tid:0x9298] hipDeviceCanAccessPeer: Returned hipSuccess :
  non-peers:                        :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_peer.cpp:176 : 605413930830 us: 29864: [tid:0x9298] ←[32m hipDeviceCanAccessPeer ( 0000008AEBEFF890, 0, 0 ) ←[0m
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_peer.cpp:177 : 605413930882 us: 29864: [tid:0x9298] hipDeviceCanAccessPeer: Returned hipSuccess :
  device#0
  ...
  :4:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\pal\palmemory.cpp:430 : 605414517802 us: 29864: [tid:0x9298] Free-:     8000 bytes, VM[ 3007c8000,  3007d0000]
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\devprogram.cpp:2979: 605414517893 us: 29864: [tid:0x9298] For Init/Fini: Kernel Name: __amd_rocclr_copyBufferToImage
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\devprogram.cpp:2979: 605414518259 us: 29864: [tid:0x9298] For Init/Fini: Kernel Name: __amd_rocclr_copyBuffer
  ...
  :4:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\pal\palmemory.cpp:206 : 605414523422 us: 29864: [tid:0x9298] Alloc: 100000 bytes, ptr[00000003008D0000-00000003009D0000], obj[00000003007D0000-00000003047D0000]
  :4:C:\constructicon\builds\gfx\two\22.40\drivers\compute\vdi\device\pal\palmemory.cpp:206 : 605414523767 us: 29864: [tid:0x9298] Alloc: 100000 bytes, ptr[00000003009D0000-0000000300AD0000], obj[00000003007D0000-00000003047D0000]
  :3:C:\constructicon\builds\gfx\two\22.40\drivers\compute\hipamd\src\hip_memory.cpp:681 : 605414524092 us: 29864: [tid:0x9298] hipMemGetInfo: Returned hipSuccess :
  memInfo.total:                    12.06 GB
  memInfo.free:                     11.93 GB (99%)

Logging hipcc commands
================================================================================

To see the detailed commands that hipcc issues, set the environment variable
``HIPCC_VERBOSE``. Doing so will print the HIP-clang (or NVCC) commands that
hipcc generates to ``stderr``.

.. code-block:: shell

  export HIPCC_VERBOSE=1
  hipcc main.cpp
  hipcc-cmd: /opt/rocm/lib/llvm/bin/clang++ --offload-arch=gfx90a --driver-mode=g++ -O3 --hip-link -x hip main.cpp

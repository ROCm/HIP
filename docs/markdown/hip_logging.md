## What is HIP logging for? ###

HIP provides a logging mechanism, which is a convinient way of printing important information so as to trace HIP API and runtime codes during the execution of HIP application.
It assists HIP development team in the development of HIP runtime, and is useful for HIP application developers as well.
Depending on the setting of logging level and logging mask, HIP logging will print different kinds of information, for different types of functionalities such as HIP APIs, executed kernels, queue commands and queue contents, etc.

## HIP Logging Level:

By Default, HIP logging is disabled, it can be enabled via environment setting,
  - AMD_LOG_LEVEL

The value of the setting controls different logging level,

```
enum LogLevel {
LOG_NONE = 0,
LOG_ERROR = 1,
LOG_WARNING = 2,
LOG_INFO = 3,
LOG_DEBUG = 4
};
```

## HIP Logging Mask:

Logging mask is designed to print types of functionalities during the execution of HIP application.
It can be set as one of the following values,

```
enum LogMask {
  LOG_API       = 0x00000001, //!< API call
  LOG_CMD       = 0x00000002, //!< Kernel and Copy Commands and Barriers
  LOG_WAIT      = 0x00000004, //!< Synchronization and waiting for commands to finish
  LOG_AQL       = 0x00000008, //!< Decode and display AQL packets
  LOG_QUEUE     = 0x00000010, //!< Queue commands and queue contents
  LOG_SIG       = 0x00000020, //!< Signal creation, allocation, pool
  LOG_LOCK      = 0x00000040, //!< Locks and thread-safety code.
  LOG_KERN      = 0x00000080, //!< kernel creations and arguments, etc.
  LOG_COPY      = 0x00000100, //!< Copy debug
  LOG_COPY2     = 0x00000200, //!< Detailed copy debug
  LOG_RESOURCE  = 0x00000400, //!< Resource allocation, performance-impacting events.
  LOG_INIT      = 0x00000800, //!< Initialization and shutdown
  LOG_MISC      = 0x00001000, //!< misc debug, not yet classified
  LOG_AQL2      = 0x00002000, //!< Show raw bytes of AQL packet
  LOG_CODE      = 0x00004000, //!< Show code creation debug
  LOG_CMD2      = 0x00008000, //!< More detailed command info, including barrier commands
  LOG_LOCATION  = 0x00010000, //!< Log message location
  LOG_ALWAYS    = 0xFFFFFFFF, //!< Log always even mask flag is zero
};
```

Once AMD_LOG_LEVEL is set, logging mask is set as default with the value 0x7FFFFFFF.
However, for different pupose of logging functionalities, logging mask can be defined as well via environment variable,

  - AMD_LOG_MASK

## HIP Logging command:

To pring HIP logging information, the function is defined as
```
#define ClPrint(level, mask, format, ...)
  do {
    if (AMD_LOG_LEVEL >= level) {
      if (AMD_LOG_MASK & mask || mask == amd::LOG_ALWAYS) {
        if (AMD_LOG_MASK & amd::LOG_LOCATION) {
          amd::log_printf(level, __FILENAME__, __LINE__, format, ##__VA_ARGS__);
        } else {
          amd::log_printf(level, "", 0, format, ##__VA_ARGS__);
        }
      }
    }
  } while (false)
```

So in HIP code, call ClPrint() function with proper input varibles as needed, for example,
```
ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Initializing HSA stack.");
```

## HIP Logging Example:

Below is an example to enable HIP logging and get logging information during execution of hipinfo,

```
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
arch.hasGlobalFloatAtomicExch:    1
arch.hasSharedInt32Atomics:       1
arch.hasSharedFloatAtomicExch:    1
arch.hasFloatAtomicAdd:           1
arch.hasGlobalInt64Atomics:       1
arch.hasSharedInt64Atomics:       1
arch.hasDoubles:                  1
arch.hasWarpVote:                 1
arch.hasWarpBallot:               1
arch.hasWarpShuffle:              1
arch.hasFunnelShift:              0
arch.hasThreadFenceSystem:        1
arch.hasSyncThreadsExt:           0
arch.hasSurfaceFuncs:             0
arch.has3dGrid:                   1
arch.hasDynamicParallelism:       0
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
```

## HIP Logging Tips:

- HIP logging works for both release and debug version of HIP application.

- Logging function with different logging level can be called in the code as needed.

- Information with logging level less than AMD_LOG_LEVEL will be printed.

- If need to save the HIP logging output information in a file, just define the file at the command when run the application at the terminal, for example,

```
user@user-test:~/hip/bin$ ./hipinfo > ~/hip_log.txt
```


# Logging Mechanisms

HIP provides a logging mechanism, which is a convenient way of printing
important information so as to trace HIP API and runtime codes during the
execution of HIP application.
It assists HIP development team in the development of HIP runtime, and is useful
for HIP application developers as well.
Depending on the setting of logging level and logging mask, HIP logging will
print different kinds of information, for different types of functionalities
such as HIP APIs, executed kernels, queue commands and queue contents, etc.

## HIP Logging Level:

By default, HIP logging is disabled, it can be enabled via the `AMD_LOG_LEVEL`
environment variable.
The value controls the logging level. The levels are defined as:

```cpp
enum LogLevel {
  LOG_NONE    = 0,
  LOG_ERROR   = 1,
  LOG_WARNING = 2,
  LOG_INFO    = 3,
  LOG_DEBUG   = 4
};
```

## HIP Logging Mask:

Logging mask is designed to print types of functionalities during the execution
of HIP application.
It can be set as one of the following values,

```cpp
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
  LOG_ALWAYS    = -1      //!< (0xFFFFFFFF) Log always even mask flag is zero
};
```

Once `AMD_LOG_LEVEL` is set, logging mask is set as default with the value
`-1`.
However, for different purpose of logging functionalities, logging mask can be
defined as well via environment variable `AMD_LOG_MASK`

## HIP Logging command:

To pring HIP logging information, the function is defined as
```cpp
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
```

So in HIP code, call `ClPrint()` function with proper input varibles as needed,
for example,
```cpp
ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Initializing HSA stack.");
```

## HIP Logging Example:

Below is an example to enable HIP logging and get logging information during execution of hipinfo on Linux,

```console
user@user-test:~/hip/bin$ export AMD_LOG_LEVEL=4
user@user-test:~/hip/bin$ ./hipinfo

:3:rocdevice.cpp            :444 : 115921848303 us: [pid:177158 tid:0x7f941a0d5a80] Initializing HSA stack.
:3:comgrctx.cpp             :33  : 115921854454 us: [pid:177158 tid:0x7f941a0d5a80] Loading COMGR library.
:3:rocdevice.cpp            :210 : 115921854490 us: [pid:177158 tid:0x7f941a0d5a80] Numa selects cpu agent[0]=0xcc4ef0(fine=0xbbffe0,coarse=0xcc53d0) for gpu agent=0xcc59c0 CPU<->GPU XGMI=0
:3:rocdevice.cpp            :1680: 115921854758 us: [pid:177158 tid:0x7f941a0d5a80] Gfx Major/Minor/Stepping: 10/3/1
:3:rocdevice.cpp            :1682: 115921854764 us: [pid:177158 tid:0x7f941a0d5a80] HMM support: 1, XNACK: 0, Direct host access: 0
:3:rocdevice.cpp            :1684: 115921854766 us: [pid:177158 tid:0x7f941a0d5a80] Max SDMA Read Mask: 0x3, Max SDMA Write Mask: 0x3
:4:rocdevice.cpp            :2063: 115921854812 us: [pid:177158 tid:0x7f941a0d5a80] Allocate hsa host memory 0x7f941a179000, size 0x38
:4:rocdevice.cpp            :2063: 115921855244 us: [pid:177158 tid:0x7f941a0d5a80] Allocate hsa host memory 0x7f930c400000, size 0x101000
:4:rocdevice.cpp            :2063: 115921856057 us: [pid:177158 tid:0x7f941a0d5a80] Allocate hsa host memory 0x7f930c200000, size 0x101000
:4:runtime.cpp              :83  : 115921856451 us: [pid:177158 tid:0x7f941a0d5a80] init
:3:hip_context.cpp          :48  : 115921856457 us: [pid:177158 tid:0x7f941a0d5a80] Direct Dispatch: 1
:3:hip_device_runtime.cpp   :546 : 115921856476 us: [pid:177158 tid:0x7f941a0d5a80]  hipGetDeviceCount ( 0x7ffc69af52e4 ) 
:3:hip_device_runtime.cpp   :548 : 115921856479 us: [pid:177158 tid:0x7f941a0d5a80] hipGetDeviceCount: Returned hipSuccess : 
:3:hip_device_runtime.cpp   :561 : 115921856484 us: [pid:177158 tid:0x7f941a0d5a80]  hipSetDevice ( 0 ) 
:3:hip_device_runtime.cpp   :565 : 115921856488 us: [pid:177158 tid:0x7f941a0d5a80] hipSetDevice: Returned hipSuccess : 
--------------------------------------------------------------------------------
device#                           0
:3:hip_device.cpp           :381 : 115921856498 us: [pid:177158 tid:0x7f941a0d5a80]  hipGetDeviceProperties ( 0x7ffc69af4fa0, 0 ) 
:3:hip_device.cpp           :383 : 115921856502 us: [pid:177158 tid:0x7f941a0d5a80] hipGetDeviceProperties: Returned hipSuccess : 
Name:                             AMD Radeon RX 6700 XT
pciBusID:                         3
pciDeviceID:                      0
pciDomainID:                      0
multiProcessorCount:              20
maxThreadsPerMultiProcessor:      2048
isMultiGpuBoard:                  0
clockRate:                        2855 Mhz
memoryClockRate:                  1000 Mhz
memoryBusWidth:                   192
totalGlobalMem:                   11.98 GB
totalConstMem:                    2147483647
sharedMemPerBlock:                64.00 KB
canMapHostMemory:                 1
regsPerBlock:                     65536
warpSize:                         32
l2CacheSize:                      3145728
computeMode:                      0
maxThreadsPerBlock:               1024
maxThreadsDim.x:                  1024
maxThreadsDim.y:                  1024
maxThreadsDim.z:                  1024
maxGridSize.x:                    2147483647
maxGridSize.y:                    65536
maxGridSize.z:                    65536
major:                            10
minor:                            3
concurrentKernels:                1
cooperativeLaunch:                1
cooperativeMultiDeviceLaunch:     1
isIntegrated:                     0
maxTexture1D:                     16384
maxTexture2D.width:               16384
maxTexture2D.height:              16384
maxTexture3D.width:               16384
maxTexture3D.height:              16384
maxTexture3D.depth:               8192
isLargeBar:                       0
asicRevision:                     0
maxSharedMemoryPerMultiProcessor: 64.00 KB
clockInstructionRate:             1000.00 Mhz
...
gcnArchName:                      gfx1031
:3:hip_device_runtime.cpp   :546 : 115921856613 us: [pid:177158 tid:0x7f941a0d5a80]  hipGetDeviceCount ( 0x7ffc69af4f8c ) 
:3:hip_device_runtime.cpp   :548 : 115921856616 us: [pid:177158 tid:0x7f941a0d5a80] hipGetDeviceCount: Returned hipSuccess : 
:3:hip_peer.cpp             :176 : 115921856625 us: [pid:177158 tid:0x7f941a0d5a80]  hipDeviceCanAccessPeer ( 0x7ffc69af4f90, 0, 0 ) 
:3:hip_peer.cpp             :177 : 115921856628 us: [pid:177158 tid:0x7f941a0d5a80] hipDeviceCanAccessPeer: Returned hipSuccess : 
peers:
:3:hip_peer.cpp             :176 : 115921856633 us: [pid:177158 tid:0x7f941a0d5a80]  hipDeviceCanAccessPeer ( 0x7ffc69af4f90, 0, 0 ) 
:3:hip_peer.cpp             :177 : 115921856636 us: [pid:177158 tid:0x7f941a0d5a80] hipDeviceCanAccessPeer: Returned hipSuccess : 
non-peers:                        device#0 

:3:hip_memory.cpp           :764 : 115921856649 us: [pid:177158 tid:0x7f941a0d5a80]  hipMemGetInfo ( 0x7ffc69af4f90, 0x7ffc69af4f98 ) 
:3:hip_memory.cpp           :788 : 115921856654 us: [pid:177158 tid:0x7f941a0d5a80] hipMemGetInfo: Returned hipSuccess : 
memInfo.total:                    11.98 GB
memInfo.free:                     11.86 GB (99%)
```

On Windows, AMD_LOG_LEVEL can be set via environment variable from advanced system setting, or from Command prompt run as administrator, as shown below as an example, which shows some debug log information calling backend runtime on Windows.
```
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
```

## HIP Logging Tips:

- HIP logging works for both release and debug version of HIP application.
- Logging function with different logging level can be called in the code as
  needed.
- Information with logging level less than AMD_LOG_LEVEL will be printed.
- If need to save the HIP logging output information in a file, just define the
  file at the command when run the application at the terminal, for example,

```console
user@user-test:~/hip/bin$ ./hipinfo > ~/hip_log.txt
```


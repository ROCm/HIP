.. meta::
  :description: HIP runtime compiler (RTC)
  :keywords: AMD, ROCm, HIP, CUDA, RTC, HIP runtime compiler

.. _hip_runtime_compiler_how-to:

*******************************************************************************
Programming for HIP runtime compiler (RTC)
*******************************************************************************

HIP supports the kernels compilation at runtime with the ``hiprtc*`` APIs.
Kernels can be stored as a text string and can be passed to HIPRTC APIs
alongside options to guide the compilation.

.. note::

  * Device code compilation via HIPRTC uses the ``__hip_internal`` namespace instead
    of the ``std`` namespace to avoid namespace collision. 
  * This library can be used for compilation on systems without AMD GPU drivers
    installed (offline compilation). However, running the compiled code still
    requires both the HIP runtime library and GPU drivers on the target system.
  * Developers can bundle this library with their application.
  * HIPRTC leverages AMD's Code Object Manager API (``Comgr``) internally, which
    is designed to simplify linking, compiling, and inspecting code objects. For
    more information, see the `llvm-project/amd/comgr/README <https://github.com/ROCm/llvm-project/blob/amd-staging/amd/comgr/README.md>`_.
  * Comgr may cache HIPRTC compilations. To force full recompilation for each HIPRTC API invocation, set AMD_COMGR_CACHE=0.

    - When viewing the *README* in the Comgr GitHub repository you should look at a
      specific branch of interest, such as ``docs/6.3.0`` or ``docs/6.4.1``, rather than the default branch.

Compilation APIs
===============================================================================

To use HIPRTC functionality the header needs to be included:

.. code-block:: cpp

  #include <hip/hiprtc.h>

.. note::

  Prior to the 7.0 release, the HIP runtime included the hipRTC library. With the 7.0
  release, the library is separate and must be specifically included as shown above. 
  
Kernels can be stored in a string:

.. code-block:: cpp

  static constexpr auto kernel_source {
  R"(
      extern "C"
      __global__ void vector_add(float* output, float* input1, float* input2, size_t size) {
        int i = threadIdx.x;
        if (i < size) {
          output[i] = input1[i] + input2[i];
        }
      }
  )"};

To compile this kernel, it needs to be associated with
:cpp:struct:`hiprtcProgram` type, which is done by declaring :code:`hiprtcProgram prog;`
and associating the string of kernel with this program:

.. code-block:: cpp

  hiprtcCreateProgram(&prog,                 // HIPRTC program handle
                      kernel_source,         // HIP kernel source string
                      "vector_add.cpp",      // Name of the HIP program, can be null or an empty string
                      0,                     // Number of headers
                      NULL,                  // Header sources
                      NULL);                 // Name of header files

:cpp:func:`hiprtcCreateProgram` API also allows you to add headers which can be
included in your RTC program. For online compilation, the compiler pre-defines
HIP device API functions, HIP specific types and macros for device compilation,
but doesn't include standard C/C++ headers by default. Users can only include
header files provided to :cpp:func:`hiprtcCreateProgram`.

After associating the kernel string with :cpp:struct:`hiprtcProgram`, you can
now compile this program using:

.. code-block:: cpp

  hiprtcCompileProgram(prog,     // hiprtcProgram
                      0,         // Number of options
                      options);  // Clang Options [Supported Clang Options](clang_options.md)

:cpp:func:`hiprtcCompileProgram` returns a status value which can be converted
to string via :cpp:func:`hiprtcGetErrorString`. If compilation is successful,
:cpp:func:`hiprtcCompileProgram` will return ``HIPRTC_SUCCESS``.

if the compilation fails or produces warnings, you can look up the logs via:

.. code-block:: cpp

  size_t logSize;
  hiprtcGetProgramLogSize(prog, &logSize);

  if (logSize) {
    string log(logSize, '\0');
    hiprtcGetProgramLog(prog, &log[0]);
    // Corrective action with logs
  }

If the compilation is successful, you can load the compiled binary in a local
variable.

.. code-block:: cpp

  size_t codeSize;
  hiprtcGetCodeSize(prog, &codeSize);

  vector<char> kernel_binary(codeSize);
  hiprtcGetCode(prog, kernel_binary.data());

After loading the binary, :cpp:struct:`hiprtcProgram` can be destroyed.
:code:`hiprtcDestroyProgram(&prog);`

The binary present in ``kernel_binary`` can now be loaded via
:cpp:func:`hipModuleLoadData` API.

.. code-block:: cpp

  hipModule_t module;
  hipFunction_t kernel;

  hipModuleLoadData(&module, kernel_binary.data());
  hipModuleGetFunction(&kernel, module, "vector_add");

And now this kernel can be launched via ``hipModule`` APIs.

The full example is below:

.. code-block:: cpp

  #include <hip/hip_runtime.h>
  #include <hip/hiprtc.h>

  #include <iostream>
  #include <string>
  #include <vector>

  #define CHECK_RET_CODE(call, ret_code)                                                             \
    {                                                                                                \
      if ((call) != ret_code) {                                                                      \
        std::cout << "Failed in call: " << #call << std::endl;                                       \
        std::abort();                                                                                \
      }                                                                                              \
    }
  #define HIP_CHECK(call) CHECK_RET_CODE(call, hipSuccess)
  #define HIPRTC_CHECK(call) CHECK_RET_CODE(call, HIPRTC_SUCCESS)

  // source code for hiprtc
  static constexpr auto kernel_source{
      R"(
      extern "C"
      __global__ void vector_add(float* output, float* input1, float* input2, size_t size) {
        int i = threadIdx.x;
        if (i < size) {
          output[i] = input1[i] + input2[i];
        }
      }
  )"};

  int main() {
    hiprtcProgram prog;
    auto rtc_ret_code = hiprtcCreateProgram(&prog,            // HIPRTC program handle
                                            kernel_source,    // kernel source string
                                            "vector_add.cpp", // Name of the file
                                            0,                // Number of headers
                                            NULL,             // Header sources
                                            NULL);            // Name of header file

    if (rtc_ret_code != HIPRTC_SUCCESS) {
      std::cout << "Failed to create program" << std::endl;
      std::abort();
    }

    hipDeviceProp_t props;
    int device = 0;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    std::string sarg = std::string("--gpu-architecture=") +
        props.gcnArchName;  // device for which binary is to be generated

    const char* options[] = {sarg.c_str()};

    rtc_ret_code = hiprtcCompileProgram(prog,      // hiprtcProgram
                                        0,         // Number of options
                                        options);  // Clang Options
    if (rtc_ret_code != HIPRTC_SUCCESS) {
      std::cout << "Failed to create program" << std::endl;
      std::abort();
    }

    size_t logSize;
    HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));

    if (logSize) {
      std::string log(logSize, '\0');
      HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
      std::cout << "Compilation failed or produced warnings: " << log << std::endl;
      std::abort();
    }

    size_t codeSize;
    HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

    std::vector<char> kernel_binary(codeSize);
    HIPRTC_CHECK(hiprtcGetCode(prog, kernel_binary.data()));

    HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

    hipModule_t module;
    hipFunction_t kernel;

    HIP_CHECK(hipModuleLoadData(&module, kernel_binary.data()));
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "vector_add"));

    constexpr size_t ele_size = 256;  // total number of items to add
    std::vector<float> hinput, output;
    hinput.reserve(ele_size);
    output.reserve(ele_size);
    for (size_t i = 0; i < ele_size; i++) {
      hinput.push_back(static_cast<float>(i + 1));
      output.push_back(0.0f);
    }

    float *dinput1, *dinput2, *doutput;
    HIP_CHECK(hipMalloc(&dinput1, sizeof(float) * ele_size));
    HIP_CHECK(hipMalloc(&dinput2, sizeof(float) * ele_size));
    HIP_CHECK(hipMalloc(&doutput, sizeof(float) * ele_size));

    HIP_CHECK(hipMemcpy(dinput1, hinput.data(), sizeof(float) * ele_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dinput2, hinput.data(), sizeof(float) * ele_size, hipMemcpyHostToDevice));

    struct {
      float* output;
      float* input1;
      float* input2;
      size_t size;
    } args{doutput, dinput1, dinput2, ele_size};

    auto size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(kernel, 1, 1, 1, ele_size, 1, 1, 0, nullptr, nullptr, config));

    HIP_CHECK(hipMemcpy(output.data(), doutput, sizeof(float) * ele_size, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < ele_size; i++) {
      if ((hinput[i] + hinput[i]) != output[i]) {
        std::cout << "Failed in validation: " << (hinput[i] + hinput[i]) << " - " << output[i]
                  << std::endl;
        std::abort();
      }
    }
    std::cout << "Passed" << std::endl;

    HIP_CHECK(hipFree(dinput1));
    HIP_CHECK(hipFree(dinput2));
    HIP_CHECK(hipFree(doutput));
  }

.. note::

  Some applications define datatypes such as ``int64_t``, ``uint64_t``, ``int32_t``, and ``uint32_t``
  that could lead to conflicts when integrating with ``hipRTC``. To resolve these conflicts, these
  datatypes are replaced with HIP-specific internal datatypes prefixed with ``__hip``. For example,
  ``int64_t`` is replaced by ``__hip_int64_t``.

HIPRTC specific options
===============================================================================

HIPRTC provides a few HIPRTC specific flags:

* ``--gpu-architecture`` : This flag can guide the code object generation for a
  specific GPU architecture. Example:
  ``--gpu-architecture=gfx906:sramecc+:xnack-``, its equivalent to
  ``--offload-arch``.

  * This option is compulsory if compilation is done on a system without AMD
    GPUs supported by HIP runtime.

  * Otherwise, HIPRTC will load the hip runtime and gather the current device
    and its architecture info and use it as option.

* ``-fgpu-rdc`` : This flag when provided during the
  :cpp:func:`hiprtcCreateProgram` generates the bitcode (HIPRTC doesn't convert
  this bitcode into ISA and binary). This bitcode can later be fetched using
  :cpp:func:`hiprtcGetBitcode` and :cpp:func:`hiprtcGetBitcodeSize` APIs.

Bitcode
-------------------------------------------------------------------------------

In the usual scenario, the kernel associated with :cpp:struct:`hiprtcProgram` is
compiled into the binary which can be loaded and run. However, if ``-fgpu-rdc``
option is provided in the compile options, HIPRTC calls comgr and generates only
the LLVM bitcode. It doesn't convert this bitcode to ISA and generate the final
binary.

.. code-block:: cpp

  std::string sarg = std::string("-fgpu-rdc");
  const char* options[] = {
      sarg.c_str() };
  hiprtcCompileProgram(prog, // hiprtcProgram
                       1,    // Number of options
                       options);

If the compilation is successful, one can load the bitcode in a local variable
using the bitcode APIs provided by HIPRTC.

.. code-block:: cpp

  size_t bitCodeSize;
  hiprtcGetBitcodeSize(prog, &bitCodeSize);

  vector<char> kernel_bitcode(bitCodeSize);
  hiprtcGetBitcode(prog, kernel_bitcode.data());

CU mode vs WGP mode
-------------------------------------------------------------------------------

All :doc:`supported AMD GPUs <rocm-install-on-linux:reference/system-requirements>` are built around a data-parallel
processor (DPP) array.

On CDNA GPUs, the DPP is organized as a set of compute unit (CU) pipelines, with each CU containing a single SIMD64
unit. Each CU has its own low-latency memory space called local data share (LDS), which threads from a warp running on
the CU can access.

On RDNA GPUs, the DPP is organized as a set of workgroup processor (WGP) pipelines. Each WGP contains two CUs, and each
CU contains two SIMD32 units. The LDS is attached to the WGP, so threads from different warps can access the same LDS if
they run on CUs within the same WGP.

.. note::
  
  Because CDNA GPUs do not use workgroup processors and have a different CU layout, the following information applies
  only to RDNA GPUs.

Warps are dispatched in one of two modes. These control whether warps are distributed across two SIMD32s (**CU mode**)
or across all four SIMD32s within a WGP (**WGP mode**).

CU mode executes two warps per block on a single CU and provides only half the LDS to those warps. Independence between
CUs can improve performance for workloads avoiding inter-warp communication, but LDS capacity per CU is limited.

WGP mode executes four warps per block on a WGP with a shared LDS. It can increase occupancy and improve performance
for workloads without heavy inter-warp communication, but it can degrade performance for programs relying on atomics or
extensive inter-warp communication.

For more information on the differences between CU and WGP modes, please refer to the appropriate ISA reference under
`AMD RDNA architecture <https://gpuopen.com/amd-gpu-architecture-programming-documentation/>`__.

.. note::

  HIPRTC assumes **WGP mode by default** for RDNA GPUs. This can be overridden by passing ``-mcumode`` as a compile
  option in :cpp:func:`hiprtcCompileProgram`.

Linker APIs
===============================================================================

The bitcode generated using the HIPRTC Bitcode APIs can be loaded using
``hipModule`` APIs and also can be linked with other generated bitcodes with
appropriate linker flags using the HIPRTC linker APIs. This also provides more
flexibility and optimizations to the applications who want to generate the
binary dynamically according to their needs. The input bitcodes can be generated
only for a specific architecture or it can be a bundled bitcode which is
generated for multiple architectures.

Example
-------------------------------------------------------------------------------

Firstly, HIPRTC link instance or a pending linker invocation must be created
using :cpp:func:`hiprtcLinkCreate`, with the appropriate linker options
provided.

.. code-block:: cpp

  hiprtcLinkCreate( num_options,           // number of options
                    options,               // Array of options
                    option_vals,           // Array of option values cast to void*
                    &rtc_link_state );     // HIPRTC link state created upon success

Following which, the bitcode data can be added to this link instance via
:cpp:func:`hiprtcLinkAddData` (if the data is present as a string) or
:cpp:func:`hiprtcLinkAddFile` (if the data is present as a file) with the
appropriate input type according to the data or the bitcode used.

.. code-block:: cpp

  hiprtcLinkAddData(rtc_link_state,        // HIPRTC link state
                    input_type,            // type of the input data or bitcode
                    bit_code_ptr,          // input data which is null terminated
                    bit_code_size,         // size of the input data
                    "a",                   // optional name for this input
                    0,                     // size of the options
                    0,                     // Array of options applied to this input
                    0);                    // Array of option values cast to void*

.. code-block:: cpp

  hiprtcLinkAddFile(rtc_link_state,        // HIPRTC link state
                    input_type,            // type of the input data or bitcode
                    bc_file_path.c_str(),  // path to the input file where bitcode is present
                    0,                     // size of the options
                    0,                     // Array of options applied to this input
                    0);                    // Array of option values cast to void*

Once the bitcodes for multiple architectures are added to the link instance, the
linking of the device code must be completed using :cpp:func:`hiprtcLinkComplete`
which generates the final binary.

.. code-block:: cpp

  hiprtcLinkComplete(rtc_link_state,       // HIPRTC link state
                     &binary,              // upon success, points to the output binary
                     &binarySize);         // size of the binary is stored (optional)

If the :cpp:func:`hiprtcLinkComplete` returns successfully, the generated binary
can be loaded and run using the ``hipModule*`` APIs.

.. code-block:: cpp

  hipModuleLoadData(&module, binary);

.. note::

  * The compiled binary must be loaded before HIPRTC link instance is destroyed
    using the :cpp:func:`hiprtcLinkDestroy` API.

    .. code-block:: cpp

      hiprtcLinkDestroy(rtc_link_state);

  * The correct sequence of calls is : :cpp:func:`hiprtcLinkCreate`,
    :cpp:func:`hiprtcLinkAddData` or :cpp:func:`hiprtcLinkAddFile`,
    :cpp:func:`hiprtcLinkComplete`, :cpp:func:`hipModuleLoadData`,
    :cpp:func:`hiprtcLinkDestroy`.

Input Types
-------------------------------------------------------------------------------

HIPRTC provides ``hiprtcJITInputType`` enumeration type which defines the input
types accepted by the Linker APIs. Here are the ``enum`` values of
``hiprtcJITInputType``. However only the input types
``HIPRTC_JIT_INPUT_LLVM_BITCODE``, ``HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE`` and
``HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE`` are supported currently.

``HIPRTC_JIT_INPUT_LLVM_BITCODE`` can be used to load both LLVM bitcode or LLVM
IR assembly code. However, ``HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE`` and
``HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE`` are only for bundled
bitcode and archive of bundled bitcode.

.. code-block:: cpp

  HIPRTC_JIT_INPUT_CUBIN = 0,
  HIPRTC_JIT_INPUT_PTX,
  HIPRTC_JIT_INPUT_FATBINARY,
  HIPRTC_JIT_INPUT_OBJECT,
  HIPRTC_JIT_INPUT_LIBRARY,
  HIPRTC_JIT_INPUT_NVVM,
  HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES,
  HIPRTC_JIT_INPUT_LLVM_BITCODE = 100,
  HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE = 101,
  HIPRTC_JIT_INPUT_LLVM_ARCHIVES_OF_BUNDLED_BITCODE = 102,
  HIPRTC_JIT_NUM_INPUT_TYPES = (HIPRTC_JIT_NUM_LEGACY_INPUT_TYPES + 3)

Backward Compatibility of LLVM Bitcode/IR
-------------------------------------------------------------------------------

For HIP applications utilizing HIPRTC to compile LLVM bitcode/IR, compatibility
is assured only when the ROCm or HIP SDK version used for generating the LLVM
bitcode/IR matches the version used during the runtime compilation. When an
application requires the ingestion of bitcode/IR not derived from the currently
installed AMD compiler, it must run with HIPRTC and comgr dynamic libraries that
are compatible with the version of the bitcode/IR.

`Comgr <https://github.com/ROCm/llvm-project/tree/amd-staging/amd/comgr/README.md>`_ is a
shared library that incorporates the LLVM/Clang compiler that HIPRTC relies on.
To identify the bitcode/IR version that comgr is compatible with, one can
execute "clang -v" using the clang binary from the same ROCm or HIP SDK package.
For instance, if compiling bitcode/IR version 14, the HIPRTC and comgr libraries
released by AMD around mid 2022 would be the best choice, assuming the
LLVM/Clang version included in the package is also version 14.

.. note:: 
  When viewing the *README* in the Comgr GitHub repository you should look at a
  specific branch of interest, such as ``docs/6.3.0`` or ``docs/6.4.1``, rather than the default branch.

To ensure smooth operation and compatibility, an application may choose to ship
the specific versions of HIPRTC and comgr dynamic libraries, or it may opt to
clearly specify the version requirements and dependencies. This approach
guarantees that the application can correctly compile the specified version of
bitcode/IR.

Link Options
-------------------------------------------------------------------------------

* ``HIPRTC_JIT_IR_TO_ISA_OPT_EXT`` - AMD Only. Options to be passed on to link
  step of compiler by :cpp:func:`hiprtcLinkCreate`.

* ``HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT`` - AMD Only. Count of options passed on
  to link step of compiler.

Example:

.. code-block:: cpp

  const char* isaopts[] = {"-mllvm", "-inline-threshold=1", "-mllvm", "-inlinehint-threshold=1"};
  std::vector<hiprtcJIT_option> jit_options = {HIPRTC_JIT_IR_TO_ISA_OPT_EXT,
                                              HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
  size_t isaoptssize = 4;
  const void* lopts[] = {(void*)isaopts, (void*)(isaoptssize)};
  hiprtcLinkState linkstate;
  hiprtcLinkCreate(2, jit_options.data(), (void**)lopts, &linkstate);

Error Handling
===============================================================================

HIPRTC defines the ``hiprtcResult`` enumeration type and a function
:cpp:func:`hiprtcGetErrorString` for API call error handling. ``hiprtcResult``
``enum`` defines the API result codes. HIPRTC APIs return ``hiprtcResult`` to
indicate the call result. :cpp:func:`hiprtcGetErrorString` function returns a
string describing the given ``hiprtcResult`` code, for example HIPRTC_SUCCESS to
"HIPRTC_SUCCESS". For unrecognized enumeration values, it returns
"Invalid HIPRTC error code".

``hiprtcResult`` ``enum`` supported values and the
:cpp:func:`hiprtcGetErrorString` usage are mentioned below.

.. code-block:: cpp

  HIPRTC_SUCCESS = 0,
  HIPRTC_ERROR_OUT_OF_MEMORY = 1,
  HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  HIPRTC_ERROR_INVALID_INPUT = 3,
  HIPRTC_ERROR_INVALID_PROGRAM = 4,
  HIPRTC_ERROR_INVALID_OPTION = 5,
  HIPRTC_ERROR_COMPILATION = 6,
  HIPRTC_ERROR_LINKING = 7,
  HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 8,
  HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 9,
  HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 10,
  HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 11,
  HIPRTC_ERROR_INTERNAL_ERROR = 12

.. code-block:: cpp

  hiprtcResult result;
  result = hiprtcCompileProgram(prog, 1, opts);
  if (result != HIPRTC_SUCCESS) {
  std::cout << "hiprtcCompileProgram fails with error " << hiprtcGetErrorString(result);
  }

HIPRTC General APIs
===============================================================================

HIPRTC provides ``hiprtcVersion(int* major, int* minor)`` for querying the
version. This sets the output parameters major and minor with the HIP Runtime
compilation major version and minor version number respectively.

Currently, it returns hardcoded values. This should be implemented to return HIP
runtime major and minor version in the future releases.

Lowered Names (Mangled Names)
===============================================================================

HIPRTC mangles the ``__global__`` function names and names of ``__device__`` and
``__constant__`` variables. If the generated binary is being loaded using the
HIP Runtime API, the kernel function or ``__device__/__constant__`` variable
must be looked up by name, but this is very hard when the name has been mangled.
To overcome this, HIPRTC provides API functions that map ``__global__`` function
or ``__device__/__constant__`` variable names in the source to the mangled names
present in the generated binary.

The two APIs :cpp:func:`hiprtcAddNameExpression` and
:cpp:func:`hiprtcGetLoweredName` provide this functionality. First, a 'name
expression' string denoting the address for the ``__global__`` function or
``__device__/__constant__`` variable is provided to
:cpp:func:`hiprtcAddNameExpression`. Then, the program is compiled with
:cpp:func:`hiprtcCreateProgram`. During compilation, HIPRTC will parse the name
expression string as a C++ constant expression at the end of the user program.
Finally, the function :cpp:func:`hiprtcGetLoweredName` is called with the
original name expression and it returns a pointer to the lowered name. The
lowered name can be used to refer to the kernel or variable in the HIP Runtime
API.

.. note::

  * The identical name expression string must be provided on a subsequent call
    to :cpp:func:`hiprtcGetLoweredName` to extract the lowered name.

  * The correct sequence of calls is : :cpp:func:`hiprtcAddNameExpression`,
    :cpp:func:`hiprtcCreateProgram`, :cpp:func:`hiprtcGetLoweredName`,
    :cpp:func:`hiprtcDestroyProgram`.

  * The lowered names must be fetched using :cpp:func:`hiprtcGetLoweredName`
    only after the HIPRTC program has been compiled, and before it has been
    destroyed.

Example
-------------------------------------------------------------------------------

Kernel containing various definitions ``__global__`` functions/function
templates and ``__device__/__constant__`` variables can be stored in a string.

.. code-block:: cpp

  static constexpr const char gpu_program[] {
  R"(
  __device__ int V1; // set from host code
  static __global__ void f1(int *result) { *result = V1 + 10; }
  namespace N1 {
  namespace N2 {
  __constant__ int V2; // set from host code
  __global__ void f2(int *result) { *result = V2 + 20; }
  }
  }
  template<typename T>
  __global__ void f3(int *result) { *result = sizeof(T); }
  )"};

:cpp:func:`hiprtcAddNameExpression` is called with various name expressions
referring to the address of ``__global__`` functions and
``__device__/__constant__`` variables.

.. code-block:: cpp

  kernel_name_vec.push_back("&f1");
  kernel_name_vec.push_back("N1::N2::f2");
  kernel_name_vec.push_back("f3<int>");
  for (auto&& x : kernel_name_vec) hiprtcAddNameExpression(prog, x.c_str());
  variable_name_vec.push_back("&V1");
  variable_name_vec.push_back("&N1::N2::V2");
  for (auto&& x : variable_name_vec) hiprtcAddNameExpression(prog, x.c_str());

After which, the program is compiled using :cpp:func:`hiprtcCompileProgram`, the
generated binary is loaded using :cpp:func:`hipModuleLoadData`, and the mangled
names can be fetched using :cpp:func:`hirtcGetLoweredName`.

.. code-block:: cpp

  for (decltype(variable_name_vec.size()) i = 0; i != variable_name_vec.size(); ++i) {
    const char* name;
    hiprtcGetLoweredName(prog, variable_name_vec[i].c_str(), &name);
  }

.. code-block:: cpp

  for (decltype(kernel_name_vec.size()) i = 0; i != kernel_name_vec.size(); ++i) {
    const char* name;
    hiprtcGetLoweredName(prog, kernel_name_vec[i].c_str(), &name);
  }

The mangled name of the variables are used to look up the variable in the module
and update its value.

.. code-block:: cpp

  hipDeviceptr_t variable_addr;
  size_t bytes{};
  hipModuleGetGlobal(&variable_addr, &bytes, module, name);
  hipMemcpyHtoD(variable_addr, &initial_value, sizeof(initial_value));


Finally, the mangled name of the kernel is used to launch it using the
``hipModule`` APIs.

.. code-block:: cpp

  hipFunction_t kernel;
  hipModuleGetFunction(&kernel, module, name);
  hipModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, config);

Versioning
===============================================================================

HIPRTC uses the following versioning:

* Linux

  * HIPRTC follows the same versioning as HIP runtime library.
  * The ``so`` name field for the shared library is set to MAJOR version. For
    example, for HIP 5.3 the ``so`` name is set to 5 (``hiprtc.so.5``).

* Windows

  * HIPRTC dll is named as ``hiprtcXXYY.dll`` where ``XX`` is MAJOR version and
    ``YY`` is MINOR version. For example, for HIP 5.3 the name is
    ``hiprtc0503.dll``.

HIP header support
===============================================================================

Added HIPRTC support for all the hip common header files such as
``library_types.h``, ``hip_math_constants.h``, ``hip_complex.h``,
``math_functions.h``, ``surface_types.h`` etc. from 6.1. HIPRTC users need not
include any HIP macros or constants explicitly in their header files. All of
these should get included via HIPRTC builtins when the app links to HIPRTC
library.

Deprecation notice
===============================================================================

* Currently HIPRTC APIs are separated from HIP APIs and HIPRTC is available as a
  separate library ``libhiprtc.so``/ ``libhiprtc.dll``. But on Linux, HIPRTC
  symbols are also present in ``libamdhip64.so`` in order to support the
  existing applications. Gradually, these symbols will be removed from HIP
  library and applications using HIPRTC will be required to explicitly link to
  HIPRTC library. However, on Windows ``hiprtc.dll`` must be used as the
  ``amdhip64.dll`` doesn't contain the HIPRTC symbols.

* Data types such as ``uint32_t``, ``uint64_t``, ``int32_t``, ``int64_t``
  defined in std namespace in HIPRTC are deprecated earlier and are being
  removed from ROCm release 6.1 since these can conflict with the standard
  C++ data types. These data types are now prefixed with ``__hip__``, for example
  ``__hip_uint32_t``. Applications previously using ``std::uint32_t`` or similar
  types can use ``__hip_`` prefixed types to avoid conflicts with standard std
  namespace or application can have their own definitions for these types. Also,
  type_traits templates previously defined in std namespace are moved to
  ``__hip_internal`` namespace as implementation details.

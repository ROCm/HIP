# Support of Clang options
 Clang versiion: clang version 9.0.0 (https://github.com/llvm-mirror/clang.git 50988b3ea26e892a74a2848e5a06b96f2472a05c) (https://github.com/RadeonOpenCompute/llvm.git 6483275d3d7603784c30e08927f2b315dfb95639)

|Option|Support|Description|
|-------|------|-------|
|`-###`|Supported|`Print (but do not run) the commands to run for this compilation`|
|`--analyzer-output <value>`|Supported|`Static analyzer report output format (html\|plist\|plist-multi-file\|plist-html\|text).`|
|`--analyze`|Supported|`Run the static analyzer`|
|`-arcmt-migrate-emit-errors`|Unsupported|`Emit ARC errors even if the migrator can fix them`|
|`-arcmt-migrate-report-output <value>`|Unsupported|`Output path for the plist report`|
|`-B <dir>`|Supported|`Add <dir> to search path for binaries and object files used implicitly`|
|`-CC`|Supported|`Include comments from within macros in preprocessed output`|
|`-cfguard`|Supported|`Emit tables required for Windows Control Flow Guard.`|
|`-cl-denorms-are-zero`|Supported|`OpenCL only. Allow denormals to be flushed to zero.`|
|`-cl-fast-relaxed-math`|Supported|`OpenCL only. Sets -cl-finite-math-only and -cl-unsafe-math-optimizations, and defines __FAST_RELAXED_MATH__.`|
|`-cl-finite-math-only`|Supported|`OpenCL only. Allow floating-point optimizations that assume arguments and results are not NaNs or +-Inf.`|
|`-cl-fp32-correctly-rounded-divide-sqrt`|Supported|`OpenCL only. Specify that single precision floating-point divide and sqrt used in the program source are correctly rounded.`|
|`-cl-kernel-arg-info`|Supported|`OpenCL only. Generate kernel argument metadata.`|
|`-cl-mad-enable`|Supported|`OpenCL only. Allow use of less precise MAD computations in the generated binary.`|
|`-cl-no-signed-zeros`|Supported|`OpenCL only. Allow use of less precise no signed zeros computations in the generated binary.`|
|`-cl-opt-disable`|Supported|`OpenCL only. This option disables all optimizations. By default optimizations are enabled.`|
|`-cl-single-precision-constant`|Supported|`OpenCL only. Treat double precision floating-point constant as single precision constant.`|
|`-cl-std=<value>`|Supported|`OpenCL language standard to compile for.`|
|`-cl-strict-aliasing`|Supported|`OpenCL only. This option is added for compatibility with OpenCL 1.0.`|
|`-cl-uniform-work-group-size`|Supported|`OpenCL only. Defines that the global work-size be a multiple of the work-group size specified to clEnqueueNDRangeKernel`|
|`-cl-unsafe-math-optimizations`|Supported|`OpenCL only. Allow unsafe floating-point optimizations. Also implies -cl-no-signed-zeros and -cl-mad-enable.`|
|`--config <value>`|Supported|`Specifies configuration file`|
|`--cuda-compile-host-device`|Supported|`Compile CUDA code for both host and device (default). Has no effect on non-CUDA compilations.`|
|`--cuda-device-only`|Supported|`Compile CUDA code for device only`|
|`--cuda-gpu-arch=<value>`|Supported|`CUDA GPU architecture (e.g. sm_35). May be specified more than once.`|
|`--cuda-host-only`|Supported|`Compile CUDA code for host only. Has no effect on non-CUDA compilations.`|
|`--cuda-include-ptx=<value>`|Unsupported|`Include PTX for the following GPU architecture (e.g. sm_35) or 'all'. May be specified more than once.`|
|`--cuda-noopt-device-debug`|Unsupported|`Enable device-side debug info generation. Disables ptxas optimizations.`|
|`--cuda-path-ignore-env`|Unsupported|`Ignore environment variables to detect CUDA installation`|
|`--cuda-path=<value>`|Unsupported|`CUDA installation path`|
|`-cxx-isystem <directory>`|Supported|`Add directory to the C++ SYSTEM include search path`|
|`-C`|Supported|`Include comments in preprocessed output`|
|`-c`|Supported|`Only run preprocess, compile, and assemble steps`|
|`-dD`|Supported|`Print macro definitions in -E mode in addition to normal output`|
|`-dependency-dot <value>`|Supported|`Filename to write DOT-formatted header dependencies to`|
|`-dependency-file <value>`|Supported|`Filename (or -) to write dependency output to`|
|`-dI`|Supported|`Print include directives in -E mode in addition to normal output`|
|`-dM`|Supported|`Print macro definitions in -E mode instead of normal output`|
|`-D <macro>`|Supported|`=<value> Define <macro> to <value> (or 1 if <value> omitted)`|
|`-emit-ast`|Supported|`Emit Clang AST files for source inputs`|
|`-emit-llvm`|Supported|`Use the LLVM representation for assembler and object files`|
|`-enable-trivial-auto-var-init-zero-knowing-it-will-be-removed-from-clang<value>`|Supported|`Trivial automatic variable initialization to zero is only here for benchmarks, it'll eventually be removed, and I'm OK with that because I'm only using it to benchmark`|
|`-E`||`Only run the preprocessor`|
|`-faddrsig`|Supported|`Emit an address-significance table`|
|`-faligned-allocation`|Supported|`Enable C++17 aligned allocation functions`|
|`-fallow-editor-placeholders`|Supported|`Treat editor placeholders as valid source code`|
|`-fansi-escape-codes`|Supported|`Use ANSI escape codes for diagnostics`|
|`-fapple-kext`|Unsupported|`Use Apple's kernel extensions ABI`|
|`-fapple-pragma-pack`|Unsupported|`Enable Apple gcc-compatible #pragma pack handling`|
|`-fapplication-extension`|Unsupported|`Restrict code to those available for App Extensions`|
|`-fblocks`|Supported|`Enable the 'blocks' language feature`|
|`-fborland-extensions`|Unsupported|`Accept non-standard constructs supported by the Borland compiler`|
|`-fbuild-session-file=<file>`|Supported|`Use the last modification time of <file> as the build session timestamp`|
|`-fbuild-session-timestamp=<time since Epoch in seconds>`|Supported|`Time when the current build session started`|
|`-fbuiltin-module-map`|Unsupported|`Load the clang builtins module map file.`|
|`-fc++-static-destructors`|Supported|`Enable C++ static destructor registration (the default)`|
|`-fcall-saved-x10`|Unsupported|`Make the x10 register call-saved (AArch64 only)`|
|`-fcall-saved-x11`|Unsupported|`Make the x11 register call-saved (AArch64 only)`|
|`-fcall-saved-x12`|Unsupported|`Make the x12 register call-saved (AArch64 only)`|
|`-fcall-saved-x13`|Unsupported|`Make the x13 register call-saved (AArch64 only)`|
|`-fcall-saved-x14`|Unsupported|`Make the x14 register call-saved (AArch64 only)`|
|`-fcall-saved-x15`|Unsupported|`Make the x15 register call-saved (AArch64 only)`|
|`-fcall-saved-x18`|Unsupported|`Make the x18 register call-saved (AArch64 only)`|
|`-fcall-saved-x8`|Unsupported|`Make the x8 register call-saved (AArch64 only)`|
|`-fcall-saved-x9`|Unsupported|`Make the x9 register call-saved (AArch64 only)`|
|`-fcf-protection=<value>`|Unsupported|`Instrument control-flow architecture protection. Options: return, branch, full, none.`|
|`-fcf-protection`|Unsupported|`Enable cf-protection in 'full' mode`|
|`-fchar8_t`|Supported|`Enable C++ builtin type char8_t`|
|`-fclang-abi-compat=<version>`|Supported|`Attempt to match the ABI of Clang <version>`|
|`-fcolor-diagnostics`|Supported|`Use colors in diagnostics`|
|`-fcomment-block-commands=<arg>`|Supported|`Treat each comma separated argument in <arg> as a documentation comment block command`|
|`-fcomplete-member-pointers`|Supported|`Require member pointer base types to be complete if they would be significant under the Microsoft ABI`|
|`-fcoroutines-ts`|Supported|`Enable support for the C++ Coroutines TS`|
|`-fcoverage-mapping`|Unsupported|`Generate coverage mapping to enable code coverage analysis`|
|`-fcuda-approx-transcendentals`|Unsupported|`Use approximate transcendental functions`|
|`-fcuda-flush-denormals-to-zero`|Supported|`Flush denormal floating point values to zero in CUDA device mode.`|
|`-fcuda-short-ptr`|Unsupported|`Use 32-bit pointers for accessing const/local/shared address spaces.`|
|`-fcxx-exceptions`|Supported|`Enable C++ exceptions`|
|`-fdata-sections`|Supported|`Place each data in its own section (ELF Only)`|
|`-fdebug-info-for-profiling`|Supported|`Emit extra debug info to make sample profile more accurate.`|
|`-fdebug-macro`|Supported|`Emit macro debug information`|
|`-fdebug-prefix-map=<value>`|Supported|`remap file source paths in debug info`|
|`-fdebug-ranges-base-address`|Supported|`Use DWARF base address selection entries in debug_ranges`|
|`-fdebug-types-section`|Supported|`Place debug types in their own section (ELF Only)`|
|`-fdeclspec`|Supported|`Allow __declspec as a keyword`|
|`-fdelayed-template-parsing`|Supported|`Parse templated function definitions at the end of the translation unit`|
|`-fdelete-null-pointer-checks`|Supported|`Treat usage of null pointers as undefined behavior.`|
|`-fdiagnostics-absolute-paths`|Supported|`Print absolute paths in diagnostics`|
|`-fdiagnostics-hotness-threshold=<number>`|Unsupported|`Prevent optimization remarks from being output if they do not have at least this profile count`|
|`-fdiagnostics-parseable-fixits`|Supported|`Print fix-its in machine parseable form`|
|`-fdiagnostics-print-source-range-info`|Supported|`Print source range spans in numeric form`|
|`-fdiagnostics-show-hotness`|Unsupported|`Enable profile hotness information in diagnostic line`|
|`-fdiagnostics-show-note-include-stack`|Supported|`Display include stacks for diagnostic notes`|
|`-fdiagnostics-show-option`|Supported|`Print option name with mappable diagnostics`|
|`-fdiagnostics-show-template-tree`|Supported|`Print a template comparison tree for differing templates`|
|`-fdigraphs`|Supported|`Enable alternative token representations '<:', ':>', '<%', '%>', '%:', '%:%:' (default)`|
|`-fdiscard-value-names`|Supported|`Discard value names in LLVM IR`|
|`-fdollars-in-identifiers`|Supported|`Allow '$' in identifiers`|
|`-fdouble-square-bracket-attributes`|Supported|`Enable '[[]]' attributes in all C and C++ language modes`|
|`-fdwarf-exceptions`|Unsupported|`Use DWARF style exceptions`|
|`-fembed-bitcode-marker`|Supported|`Embed placeholder LLVM IR data as a marker`|
|`-fembed-bitcode=<option>`|Supported|`Embed LLVM bitcode (option: off, all, bitcode, marker)`|
|`-fembed-bitcode`|Supported|`Embed LLVM IR bitcode as data`|
|`-femit-all-decls`|Supported|`Emit all declarations, even if unused`|
|`-femulated-tls`|Supported|`Use emutls functions to access thread_local variables`|
|`-fexceptions`|Supported|`Enable support for exception handling`|
|`-fexperimental-isel`|Supported|`Enables the experimental global instruction selector`|
|`-fexperimental-new-pass-manager`|Supported|`Enables an experimental new pass manager in LLVM.`|
|`-ffast-math`|Supported|`Allow aggressive, lossy floating-point optimizations`|
|`-ffine-grained-bitfield-accesses`|Supported|`Use separate accesses for consecutive bitfield runs with legal widths and alignments.`|
|`-ffixed-point`|Supported|`Enable fixed point types`|
|`-ffixed-r19`|Unsupported|`Reserve register r19 (Hexagon only)`|
|`-ffixed-r9`|Unsupported|`Reserve the r9 register (ARM only)`|
|`-ffixed-x10`|Unsupported|`Reserve the 10 register (AArch64 only)`|
|`-ffixed-x11`|Unsupported|`Reserve the 11 register (AArch64 only)`|
|`-ffixed-x12`|Unsupported|`Reserve the 12 register (AArch64 only)`|
|`-ffixed-x13`|Unsupported|`Reserve the 13 register (AArch64 only)`|
|`-ffixed-x14`|Unsupported|`Reserve the 14 register (AArch64 only)`|
|`-ffixed-x15`|Unsupported|`Reserve the 15 register (AArch64 only)`|
|`-ffixed-x18`|Unsupported|`Reserve the 18 register (AArch64 only)`|
|`-ffixed-x1`|Unsupported|`Reserve the 1 register (AArch64 only)`|
|`-ffixed-x20`|Unsupported|`Reserve the 20 register (AArch64 only)`|
|`-ffixed-x21`|Unsupported|`Reserve the 21 register (AArch64 only)`|
|`-ffixed-x22`|Unsupported|`Reserve the 22 register (AArch64 only)`|
|`-ffixed-x23`|Unsupported|`Reserve the 23 register (AArch64 only)`|
|`-ffixed-x24`|Unsupported|`Reserve the 24 register (AArch64 only)`|
|`-ffixed-x25`|Unsupported|`Reserve the 25 register (AArch64 only)`|
|`-ffixed-x26`|Unsupported|`Reserve the 26 register (AArch64 only)`|
|`-ffixed-x27`|Unsupported|`Reserve the 27 register (AArch64 only)`|
|`-ffixed-x28`|Unsupported|`Reserve the 28 register (AArch64 only)`|
|`-ffixed-x2`|Unsupported|`Reserve the 2 register (AArch64 only)`|
|`-ffixed-x3`|Unsupported|`Reserve the 3 register (AArch64 only)`|
|`-ffixed-x4`|Unsupported|`Reserve the 4 register (AArch64 only)`|
|`-ffixed-x5`|Unsupported|`Reserve the 5 register (AArch64 only)`|
|`-ffixed-x6`|Unsupported|`Reserve the 6 register (AArch64 only)`|
|`-ffixed-x7`|Unsupported|`Reserve the 7 register (AArch64 only)`|
|`-ffixed-x9`|Unsupported|`Reserve the 9 register (AArch64 only)`|
|`-fforce-emit-vtables`|Supported|`Emits more virtual tables to improve devirtualization`|
|`-fforce-enable-int128`|Supported|`Enable support for int128_t type`|
|`-ffp-contract=<value>`|Supported|`Form fused FP ops (e.g. FMAs): fast (everywhere) \| on (according to FP_CONTRACT pragma, default) \| off (never fuse)`|
|`-ffreestanding`|Supported|`Assert that the compilation takes place in a freestanding environment`|
|`-ffunction-sections`|Supported|`Place each function in its own section (ELF Only)`|
|`-fgnu-keywords`|Supported|`Allow GNU-extension keywords regardless of language standard`|
|`-fgnu-runtime`|Unsupported|`Generate output compatible with the standard GNU Objective-C runtime`|
|`-fgnu89-inline`|Unsupported|`Use the gnu89 inline semantics`|
|`-fgpu-rdc`|Supported|`Generate relocatable device code, also known as separate compilation mode.`|
|`-fimplicit-module-maps`|Unsupported|`Implicitly search the file system for module map files.`|
|`-finline-functions`|Supported|`Inline suitable functions`|
|`-finline-hint-functions`|Supported|`Inline functions which are (explicitly or implicitly) marked inline`|
|`-finstrument-function-entry-bare`|Unsupported|`Instrument function entry only, after inlining, without arguments to the instrumentation call`|
|`-finstrument-functions-after-inlining`|Unsupported|`Like -finstrument-functions, but insert the calls after inlining`|
|`-finstrument-functions`|Unsupported|`Generate calls to instrument function entry and exit`|
|`-fintegrated-as`|Supported|`Enable the integrated assembler`|
|`-fkeep-static-consts`|Supported|`Keep static const variables even if unused`|
|`-flto-jobs=<value>`|Unsupported|`Controls the backend parallelism of -flto=thin (default of 0 means the number of threads will be derived from the number of CPUs detected)`|
|`-flto=<value>`|Unsupported|`Set LTO mode to either 'full' or 'thin'`|
|`-flto`|Unsupported|`Enable LTO in 'full' mode`|
|`-fmath-errno`|Supported|`Require math functions to indicate errors by setting errno`|
|`-fmax-type-align=<value>`|Supported|`Specify the maximum alignment to enforce on pointers lacking an explicit alignment`|
|`-fmerge-all-constants`|Supported|`Allow merging of constants`|
|`-fmodule-file=[<name>=]<file>`|Unsupported|`Specify the mapping of module name to precompiled module file, or load a module file if name is omitted.`|
|`-fmodule-map-file=<file>`|Unsupported|`Load this module map file`|
|`-fmodule-name=<name>`|Unsupported|`Specify the name of the module to build`|
|`-fmodules-cache-path=<directory>`|Unsupported|`Specify the module cache path`|
|`-fmodules-decluse`|Unsupported|`Require declaration of modules used within a module`|
|`-fmodules-disable-diagnostic-validation`|Unsupported|`Disable validation of the diagnostic options when loading the module`|
|`-fmodules-ignore-macro=<value>`|Unsupported|`Ignore the definition of the given macro when building and loading modules`|
|`-fmodules-prune-after=<seconds>`|Unsupported|`Specify the interval (in seconds) after which a module file will be considered unused`|
|`-fmodules-prune-interval=<seconds>`|Unsupported|`Specify the interval (in seconds) between attempts to prune the module cache`|
|`-fmodules-search-all`|Unsupported|`Search even non-imported modules to resolve references`|
|`-fmodules-strict-decluse`|Unsupported|`Like -fmodules-decluse but requires all headers to be in modules`|
|`-fmodules-ts`|Unsupported|`Enable support for the C++ Modules TS`|
|`-fmodules-user-build-path <directory>`|Unsupported|`Specify the module user build path`|
|`-fmodules-validate-once-per-build-session`|Unsupported|`Don't verify input files for the modules if the module has been successfully validated or loaded during this build session`|
|`-fmodules-validate-system-headers`|Supported|`Validate the system headers that a module depends on when loading the module`|
|`-fmodules`|Unsupported|`Enable the 'modules' language feature`|
|`-fms-compatibility-version=<value>`|Supported|`Dot-separated value representing the Microsoft compiler version number to report in _MSC_VER (0 = don't define it (default))`|
|`-fms-compatibility`|Supported|`Enable full Microsoft Visual C++ compatibility`|
|`-fms-extensions`|Supported|`Accept some non-standard constructs supported by the Microsoft compiler`|
|`-fmsc-version=<value>`|Supported|`Microsoft compiler version number to report in _MSC_VER (0 = don't define it (default))`|
|`-fnew-alignment=<align>`|Supported|`Specifies the largest alignment guaranteed by '::operator new(size_t)'`|
|`-fno-access-control`|Supported|`Disable C++ access control`|
|`-fno-addrsig`|Supported|`Don't emit an address-significance table`|
|`-fno-assume-sane-operator-new`|Supported|`Don't assume that C++'s global operator new can't alias any pointer`|
|`-fno-autolink`|Supported|`Disable generation of linker directives for automatic library linking`|
|`-fno-builtin-<value>`|Supported|`Disable implicit builtin knowledge of a specific function`|
|`-fno-builtin`|Supported|`Disable implicit builtin knowledge of functions`|
|`-fno-c++-static-destructors`|Supported|`Disable C++ static destructor registration`|
|`-fno-char8_t`|Supported|`Disable C++ builtin type char8_t`|
|`-fno-common`|Supported|`Compile common globals like normal definitions`|
|`-fno-complete-member-pointers`|Supported|`Do not require member pointer base types to be complete if they would be significant under the Microsoft ABI`|
|`-fno-constant-cfstrings`|Supported|`Disable creation of CodeFoundation-type constant strings`|
|`-fno-coverage-mapping`|Supported|`Disable code coverage analysis`|
|`-fno-crash-diagnostics`|Supported|`Disable auto-generation of preprocessed source files and a script for reproduction during a clang crash`|
|`-fno-debug-info-for-profiling`|Supported|`Do not emit extra debug info for sample profiler.`|
|`-fno-debug-macro`|Supported|`Do not emit macro debug information`|
|`-fno-declspec`|Unsupported|`Disallow __declspec as a keyword`|
|`-fno-delayed-template-parsing`|Supported|`Disable delayed template parsing`|
|`-fno-delete-null-pointer-checks`|Supported|`Do not treat usage of null pointers as undefined behavior.`|
|`-fno-diagnostics-fixit-info`|Supported|`Do not include fixit information in diagnostics`|
|`-fno-digraphs`|Supported|`Disallow alternative token representations '<:', ':>', '<%', '%>', '%:', '%:%:'`|
|`-fno-discard-value-names`|Supported|`Do not discard value names in LLVM IR`|
|`-fno-dollars-in-identifiers`|Supported|`Disallow '$' in identifiers`|
|`-fno-double-square-bracket-attributes`|Supported|`Disable '[[]]' attributes in all C and C++ language modes`|
|`-fno-elide-constructors`|Supported|`Disable C++ copy constructor elision`|
|`-fno-elide-type`|Supported|`Do not elide types when printing diagnostics`|
|`-fno-experimental-isel`|Supported|`Disables the experimental global instruction selector`|
|`-fno-experimental-new-pass-manager`|Supported|`Disables an experimental new pass manager in LLVM.`|
|`-fno-fine-grained-bitfield-accesses`|Supported|`Use large-integer access for consecutive bitfield runs.`|
|`-fno-fixed-point`|Supported|`Disable fixed point types`|
|`-fno-force-enable-int128`|Supported|`Disable support for int128_t type`|
|`-fno-gnu-inline-asm`|Supported|`Disable GNU style inline asm`|
|`-fno-integrated-as`|Supported|`Disable the integrated assembler`|
|`-fno-jump-tables`|Supported|`Do not use jump tables for lowering switches`|
|`-fno-lax-vector-conversions`|Supported|`Disallow implicit conversions between vectors with a different number of elements or different element types`|
|`-fno-lto`|Supported|`Disable LTO mode (default)`|
|`-fno-merge-all-constants`|Supported|`Disallow merging of constants`|
|`-fno-objc-infer-related-result-type`|Supported|`do not infer Objective-C related result type based on method family`|
|`-fno-operator-names`|Supported|`Do not treat C++ operator name keywords as synonyms for operators`|
|`-fno-plt`|Supported|`Do not use the PLT to make function calls`|
|`-fno-preserve-as-comments`|Supported|`Do not preserve comments in inline assembly`|
|`-fno-profile-generate`|Supported|`Disable generation of profile instrumentation.`|
|`-fno-profile-instr-generate`|Supported|`Disable generation of profile instrumentation.`|
|`-fno-profile-instr-use`|Supported|`Disable using instrumentation data for profile-guided optimization`|
|`-fno-register-global-dtors-with-atexit`|Supported|`Don't use atexit or __cxa_atexit to register global destructors`|
|`-fno-reroll-loops`|Supported|`Turn off loop reroller`|
|`-fno-rtlib-add-rpath`|Supported|`Do not add -rpath with architecture-specific resource directory to the linker flags`|
|`-fno-rtti-data`|Supported|`Control emission of RTTI data`|
|`-fno-rtti`|Supported|`Disable generation of rtti information`|
|`-fno-sanitize-address-poison-custom-array-cookie`|Supported|`Disable poisoning array cookies when using custom operator new[] in AddressSanitizer`|
|`-fno-sanitize-address-use-after-scope`|Supported|`Disable use-after-scope detection in AddressSanitizer`|
|`-fno-sanitize-address-use-odr-indicator`|Supported|`Disable ODR indicator globals`|
|`-fno-sanitize-blacklist`|Supported|`Don't use blacklist file for sanitizers`|
|`-fno-sanitize-cfi-cross-dso`|Supported|`Disable control flow integrity (CFI) checks for cross-DSO calls.`|
|`-fno-sanitize-coverage=<value>`|Supported|`Disable specified features of coverage instrumentation for Sanitizers`|
|`-fno-sanitize-memory-track-origins`|Supported|`Disable origins tracking in MemorySanitizer`|
|`-fno-sanitize-memory-use-after-dtor`|Supported|`Disable use-after-destroy detection in MemorySanitizer`|
|`-fno-sanitize-recover=<value>`|Supported|`Disable recovery for specified sanitizers`|
|`-fno-sanitize-stats`|Supported|`Disable sanitizer statistics gathering.`|
|`-fno-sanitize-thread-atomics`|Supported|`Disable atomic operations instrumentation in ThreadSanitizer`|
|`-fno-sanitize-thread-func-entry-exit`|Supported|`Disable function entry/exit instrumentation in ThreadSanitizer`|
|`-fno-sanitize-thread-memory-access`|Supported|`Disable memory access instrumentation in ThreadSanitizer`|
|`-fno-sanitize-trap=<value>`|Supported|`Disable trapping for specified sanitizers`|
|`-fno-short-wchar`|Supported|`Force wchar_t to be an unsigned int`|
|`-fno-show-column`|Supported|`Do not include column number on diagnostics`|
|`-fno-show-source-location`|Supported|`Do not include source location information with diagnostics`|
|`-fno-signed-char`|Supported|`Char is unsigned`|
|`-fno-signed-zeros`|Supported|`Allow optimizations that ignore the sign of floating point zeros`|
|`-fno-spell-checking`|Supported|`Disable spell-checking`|
|`-fno-stack-protector`|Supported|`Disable the use of stack protectors`|
|`-fno-stack-size-section`|Supported|`Don't emit section containing metadata on function stack sizes`|
|`-fno-standalone-debug`|Supported|`Limit debug information produced to reduce size of debug binary`|
|`-fno-strict-float-cast-overflow`|Supported|`Relax language rules and try to match the behavior of the target's native float-to-int conversion instructions`|
|`-fno-threadsafe-statics`|Supported|`Do not emit code to make initialization of local statics thread safe`|
|`-fno-trigraphs`|Supported|`Do not process trigraph sequences`|
|`-fno-unroll-loops`|Supported|`Turn off loop unroller`|
|`-fno-use-cxa-atexit`|Supported|`Don't use __cxa_atexit for calling destructors`|
|`-fno-use-init-array`|Supported|`Don't use .init_array instead of .ctors`|
|`-fobjc-arc-exceptions`|Unsupported|`Use EH-safe code when synthesizing retains and releases in -fobjc-arc`|
|`-fobjc-arc`|Unsupported|`Synthesize retain and release calls for Objective-C pointers`|
|`-fobjc-exceptions`|Unsupported|`Enable Objective-C exceptions`|
|`-fobjc-runtime=<value>`|Unsupported|`Specify the target Objective-C runtime kind and version`|
|`-fobjc-weak`|Unsupported|`Enable ARC-style weak references in Objective-C`|
|`-fopenmp-simd`|Unsupported|`Emit OpenMP code only for SIMD-based constructs.`|
|`-fopenmp-targets=<value>`|Unsupported|`Specify comma-separated list of triples OpenMP offloading targets to be supported`|
|`-fopenmp`|Unsupported|`Parse OpenMP pragmas and generate parallel code.`|
|`-foptimization-record-file=<value>`|Supported|`Specify the file name of any generated YAML optimization record`|
|`-fpack-struct=<value>`|Unsupported|`Specify the default maximum struct packing alignment`|
|`-fpascal-strings`|Supported|`Recognize and construct Pascal-style string literals`|
|`-fpass-plugin=<dsopath>`|Supported|`Load pass plugin from a dynamic shared object file (only with new pass manager).`|
|`-fpcc-struct-return`|Unsupported|`Override the default ABI to return all structs on the stack`|
|`-fplt`|Supported|`Use the PLT to make function calls`|
|`-fplugin=<dsopath>`|Supported|`Load the named plugin (dynamic shared object)`|
|`-fprebuilt-module-path=<directory>`|Unsupported|`Specify the prebuilt module path`|
|`-fprofile-exclude-files=<value>`|Unsupported|`Instrument only functions from files where names don't match all the regexes separated by a semi-colon`|
|`-fprofile-filter-files=<value>`|Unsupported|`Instrument only functions from files where names match any regex separated by a semi-colon`|
|`-fprofile-generate=<directory>`|Unsupported|`Generate instrumented code to collect execution counts into <directory>/default.profraw (overridden by LLVM_PROFILE_FILE env var)`|
|`-fprofile-generate`|Unsupported|`Generate instrumented code to collect execution counts into default.profraw (overridden by LLVM_PROFILE_FILE env var)`|
|`-fprofile-instr-generate=<file>`|Unsupported|`Generate instrumented code to collect execution counts into <file> (overridden by LLVM_PROFILE_FILE env var)`|
|`-fprofile-instr-generate`|Unsupported|`Generate instrumented code to collect execution counts into default.profraw file (overridden by '=' form of option or LLVM_PROFILE_FILE env var)`|
|`-fprofile-instr-use=<value>`|Unsupported|`Use instrumentation data for profile-guided optimization`|
|`-fprofile-remapping-file=<file>`|Unsupported|`Use the remappings described in <file> to match the profile data against names in the program`|
|`-fprofile-sample-accurate`|Unsupported|`Specifies that the sample profile is accurate`|
|`-fprofile-sample-use=<value>`|Unsupported|`Enable sample-based profile guided optimizations`|
|`-fprofile-use=<pathname>`|Unsupported|`Use instrumentation data for profile-guided optimization. If pathname is a directory, it reads from <pathname>/default.profdata. Otherwise, it reads from file <pathname>.`|
|`-freciprocal-math`|Supported|`Allow division operations to be reassociated`|
|`-freg-struct-return`|Unsupported|`Override the default ABI to return small structs in registers`|
|`-fregister-global-dtors-with-atexit`|Supported|`Use atexit or __cxa_atexit to register global destructors`|
|`-frelaxed-template-template-args`|Supported|`Enable C++17 relaxed template template argument matching`|
|`-freroll-loops`|Supported|`Turn on loop reroller`|
|`-frtlib-add-rpath`|Supported|`Add -rpath with architecture-specific resource directory to the linker flags`|
|`-fsanitize-address-field-padding=<value>`|Unsupported|`Level of field padding for AddressSanitizer`|
|`-fsanitize-address-globals-dead-stripping`|Unsupported|`Enable linker dead stripping of globals in AddressSanitizer`|
|`-fsanitize-address-poison-custom-array-cookie`|Unsupported|`Enable poisoning array cookies when using custom operator new[] in AddressSanitizer`|
|`-fsanitize-address-use-after-scope`|Unsupported|`Enable use-after-scope detection in AddressSanitizer`|
|`-fsanitize-address-use-odr-indicator`|Unsupported|`Enable ODR indicator globals to avoid false ODR violation reports in partially sanitized programs at the cost of an increase in binary size`|
|`-fsanitize-blacklist=<value>`|Unsupported|`Path to blacklist file for sanitizers`|
|`-fsanitize-cfi-cross-dso`|Unsupported|`Enable control flow integrity (CFI) checks for cross-DSO calls.`|
|`-fsanitize-cfi-icall-generalize-pointers`|Unsupported|`Generalize pointers in CFI indirect call type signature checks`|
|`-fsanitize-coverage=<value>`|Unsupported|`Specify the type of coverage instrumentation for Sanitizers`|
|`-fsanitize-hwaddress-abi=<value>`|Unsupported|`Select the HWAddressSanitizer ABI to target (interceptor or platform, default interceptor)`|
|`-fsanitize-memory-track-origins=<value>`|Unsupported|`Enable origins tracking in MemorySanitizer`|
|`-fsanitize-memory-track-origins`|Unsupported|`Enable origins tracking in MemorySanitizer`|
|`-fsanitize-memory-use-after-dtor`|Unsupported|`Enable use-after-destroy detection in MemorySanitizer`|
|`-fsanitize-recover=<value>`|Unsupported|`Enable recovery for specified sanitizers`|
|`-fsanitize-stats`|Unsupported|`Enable sanitizer statistics gathering.`|
|`-fsanitize-thread-atomics`|Unsupported|`Enable atomic operations instrumentation in ThreadSanitizer (default)`|
|`-fsanitize-thread-func-entry-exit`|Unsupported|`Enable function entry/exit instrumentation in ThreadSanitizer (default)`|
|`-fsanitize-thread-memory-access`|Unsupported|`Enable memory access instrumentation in ThreadSanitizer (default)`|
|`-fsanitize-trap=<value>`|Unsupported|`Enable trapping for specified sanitizers`|
|`-fsanitize-undefined-strip-path-components=<number>`|Unsupported|`Strip (or keep only, if negative) a given number of path components when emitting check metadata.`|
|`-fsanitize=<check>`|Unsupported|`Turn on runtime checks for various forms of undefined or suspicious behavior. See user manual for available checks`|
|`-fsave-optimization-record`|Supported|`Generate a YAML optimization record file`|
|`-fseh-exceptions`|Supported|`Use SEH style exceptions`|
|`-fshort-enums`|Supported|`Allocate to an enum type only as many bytes as it needs for the declared range of possible values`|
|`-fshort-wchar`|Unsupported|`Force wchar_t to be a short unsigned int`|
|`-fshow-overloads=<value>`|Supported|`Which overload candidates to show when overload resolution fails: best\|all; defaults to all`|
|`-fsized-deallocation`|Supported|`Enable C++14 sized global deallocation functions`|
|`-fsjlj-exceptions`|Supported|`Use SjLj style exceptions`|
|`-fslp-vectorize`|Supported|`Enable the superword-level parallelism vectorization passes`|
|`-fsplit-dwarf-inlining`|Unsupported|`Provide minimal debug info in the object/executable to facilitate online symbolication/stack traces in the absence of .dwo/.dwp files when using Split DWARF`|
|`-fsplit-lto-unit`|Unsupported|`Enables splitting of the LTO unit.`|
|`-fstack-protector-all`|Unsupported|`Enable stack protectors for all functions`|
|`-fstack-protector-strong`|Unsupported|`Enable stack protectors for some functions vulnerable to stack smashing. Compared to -fstack-protector, this uses a stronger heuristic that includes functions containing arrays of any size (and any type), as well as any calls to alloca or the taking of an address from a local variable`|
|`-fstack-protector`|Unsupported|`Enable stack protectors for some functions vulnerable to stack smashing. This uses a loose heuristic which considers functions vulnerable if they contain a char (or 8bit integer) array or constant sized calls to alloca, which are of greater size than ssp-buffer-size (default: 8 bytes). All variable sized calls to alloca are considered vulnerable`|
|`-fstack-size-section`|Supported|`Emit section containing metadata on function stack sizes`|
|`-fstandalone-debug`|Supported|`Emit full debug info for all types used by the program`|
|`-fstrict-enums`|Supported|`Enable optimizations based on the strict definition of an enum's value range`|
|`-fstrict-float-cast-overflow`|Supported|`Assume that overflowing float-to-int casts are undefined (default)`|
|`-fstrict-return`|Supported|`Always treat control flow paths that fall off the end of a non-void function as unreachable`|
|`-fstrict-vtable-pointers`|Supported|`Enable optimizations based on the strict rules for overwriting polymorphic C++ objects`|
|`-fthinlto-index=<value>`|Unsupported|`Perform ThinLTO importing using provided function summary index`|
|`-ftrap-function=<value>`|Unsupported|`Issue call to specified function rather than a trap instruction`|
|`-ftrapv-handler=<function name>`|Unsupported|`Specify the function to be called on overflow`|
|`-ftrapv`|Unsupported|`Trap on integer overflow`|
|`-ftrigraphs`|Supported|`Process trigraph sequences`|
|`-ftrivial-auto-var-init=<value>`|Supported|`Initialize trivial automatic stack variables: uninitialized (default) \| pattern`|
|`-funique-section-names`|Supported|`Use unique names for text and data sections (ELF Only)`|
|`-funroll-loops`|Supported|`Turn on loop unroller`|
|`-fuse-init-array`|Supported|`Use .init_array instead of .ctors`|
|`-fveclib=<value>`|Unsupported|`Use the given vector functions library`|
|`-fvectorize`|Unsupported|`Enable the loop vectorization passes`|
|`-fvisibility-global-new-delete-hidden`|Supported|`Give global C++ operator new and delete declarations hidden visibility`|
|`-fvisibility-inlines-hidden`|Supported|`Give inline C++ member functions hidden visibility by default`|
|`-fvisibility-ms-compat`|Supported|`Give global types 'default' visibility and global functions and variables 'hidden' visibility by default`|
|`-fvisibility=<value>`|Supported|`Set the default symbol visibility for all global declarations`|
|`-fwhole-program-vtables`|Unsupported|`Enables whole-program vtable optimization. Requires -flto`|
|`-fwrapv`|Supported|`Treat signed integer overflow as two's complement`|
|`-fwritable-strings`|Supported|`Store string literals as writable data`|
|`-fxray-always-emit-customevents`|Unsupported|`Determine whether to always emit __xray_customevent(...) calls even if the function it appears in is not always instrumented.`|
|`-fxray-always-emit-typedevents`|Unsupported|`Determine whether to always emit __xray_typedevent(...) calls even if the function it appears in is not always instrumented.`|
|`-fxray-always-instrument= <value>`|Unsupported|`DEPRECATED: Filename defining the whitelist for imbuing the 'always instrument' XRay attribute.`|
|`-fxray-attr-list= <value>`|Unsupported|`Filename defining the list of functions/types for imbuing XRay attributes.`|
|`-fxray-instruction-threshold= <value>`|Unsupported|`Sets the minimum function size to instrument with XRay`|
|`-fxray-instrumentation-bundle= <value>`|Unsupported|`Select which XRay instrumentation points to emit. Options: all, none, function, custom. Default is 'all'.`|
|`-fxray-instrument`|Unsupported|`Generate XRay instrumentation sleds on function entry and exit`|
|`-fxray-link-deps`|Unsupported|`Tells clang to add the link dependencies for XRay.`|
|`-fxray-modes= <value>`|Unsupported|`List of modes to link in by default into XRay instrumented binaries.`|
|`-fxray-never-instrument= <value>`|Unsupported|`DEPRECATED: Filename defining the whitelist for imbuing the 'never instrument' XRay attribute.`|
|`-fzvector`|Supported|`Enable System z vector language extension`|
|`-F <value>`|Unsupported|`Add directory to framework include search path`|
|`--gcc-toolchain=<value>`|Supported|`Use the gcc toolchain at the given directory`|
|`-gcodeview-ghash`|Supported|`Emit type record hashes in a .debug$H section`|
|`-gcodeview`|Supported|`Generate CodeView debug information`|
|`-gdwarf-2`|Supported|`Generate source-level debug information with dwarf version 2`|
|`-gdwarf-3`|Supported|`Generate source-level debug information with dwarf version 3`|
|`-gdwarf-4`|Supported|`Generate source-level debug information with dwarf version 4`|
|`-gdwarf-5`|Supported|`Generate source-level debug information with dwarf version 5`|
|`-gembed-source`|Supported|`Embed source text in DWARF debug sections`|
|`-gline-directives-only`|Supported|`Emit debug line info directives only`|
|`-gline-tables-only`|Supported|`Emit debug line number tables only`|
|`-gmodules`|Supported|`Generate debug info with external references to clang modules or precompiled headers`|
|`-gno-embed-source`|Supported|`Restore the default behavior of not embedding source text in DWARF debug sections`|
|`-gsplit-dwarf=<value>`|Supported|`Set DWARF fission mode to either 'split' or 'single'`|
|`-gz=<value>`|Supported|`DWARF debug sections compression type`|
|`-gz`|Supported|`DWARF debug sections compression type`|
|`-G <size>`|Unsupported|`Put objects of at most <size> bytes into small data section (MIPS / Hexagon)`|
|`-g`|Supported|`Generate source-level debug information`|
|`--help-hidden`|Supported|`Display help for hidden options`|
|`-help`|Supported|`Display available options`|
|`--hip-device-lib-path=<value>`|Supported|`HIP device library path`|
|`--hip-device-lib=<value>`|Supported|`HIP device library`|
|`--hip-link`|Supported|`Link clang-offload-bundler bundles for HIP`|
|`-H`|Supported|`Show header includes and nesting depth`|
|`-I-`|Supported|`Restrict all prior -I flags to double-quoted inclusion and remove current directory from include path`|
|`-idirafter <value>`|Supported|`Add directory to AFTER include search path`|
|`-iframeworkwithsysroot <directory>`|Unsupported|`Add directory to SYSTEM framework search path, absolute paths are relative to -isysroot`|
|`-iframework <value>`|Unsupported|`Add directory to SYSTEM framework search path`|
|`-imacros <file>`|Supported|`Include macros from file before parsing`|
|`-include-pch <file>`|Supported|`Include precompiled header file`|
|`-include <file>`|Supported|`Include file before parsing`|
|`-index-header-map`|Supported|`Make the next included directory (-I or -F) an indexer header map`|
|`-iprefix <dir>`|Supported|`Set the -iwithprefix/-iwithprefixbefore prefix`|
|`-iquote <directory>`|Supported|`Add directory to QUOTE include search path`|
|`-isysroot <dir>`|Supported|`Set the system root directory (usually /)`|
|`-isystem-after <directory>`|Supported|`Add directory to end of the SYSTEM include search path`|
|`-isystem <directory>`|Supported|`Add directory to SYSTEM include search path`|
|`-ivfsoverlay <value>`|Supported|`Overlay the virtual filesystem described by file over the real file system`|
|`-iwithprefixbefore <dir>`|Supported|`Set directory to include search path with prefix`|
|`-iwithprefix <dir>`|Supported|`Set directory to SYSTEM include search path with prefix`|
|`-iwithsysroot <directory>`|Supported|`Add directory to SYSTEM include search path, absolute paths are relative to -isysroot`|
|`-I <dir>`|Supported|`Add directory to include search path`|
|`--libomptarget-nvptx-path=<value>`|Unsupported|`Path to libomptarget-nvptx libraries`|
|`-L <dir>`|Supported|`Add directory to library search path`|
|`-mabicalls`|Unsupported|`Enable SVR4-style position-independent code (Mips only)`|
|`-malign-double`|Supported|`Align doubles to two words in structs (x86 only)`|
|`-mbackchain`|Unsupported|`Link stack frames through backchain on System Z`|
|`-mbranch-protection=<value>`|Unsupported|`Enforce targets of indirect branches and function returns`|
|`-mcode-object-v3`|Unsupported|`Enable code object v3 (AMDGPU only)`|
|`-mcrc`|Unsupported|`Allow use of CRC instructions (ARM/Mips only)`|
|`-MD`|Supported|`Write a depfile containing user and system headers`|
|`-meabi <value>`|Supported|`Set EABI type, e.g. 4, 5 or gnu (default depends on triple)`|
|`-membedded-data`|Unsupported|`Place constants in the .rodata section instead of the .sdata section even if they meet the -G <size> threshold (MIPS)`|
|`-mexecute-only`|Unsupported|`Disallow generation of data access to code sections (ARM only)`|
|`-mextern-sdata`|Unsupported|`Assume that externally defined data is in the small data if it meets the -G <size> threshold (MIPS)`|
|`-mfentry`|Supported|`Insert calls to fentry at function entry (x86 only)`|
|`-mfix-cortex-a53-835769`|Unsupported|`Workaround Cortex-A53 erratum 835769 (AArch64 only)`|
|`-mfp32`|Unsupported|`Use 32-bit floating point registers (MIPS only)`|
|`-mfp64`|Unsupported|`Use 64-bit floating point registers (MIPS only)`|
|`-MF <file>`|Supported|`Write depfile output from -MMD, -MD, -MM, or -M to <file>`|
|`-mgeneral-regs-only`|Unsupported|`Generate code which only uses the general purpose registers (AArch64 only)`|
|`-mglobal-merge`|Supported|`Enable merging of globals`|
|`-mgpopt`|Unsupported|`Use GP relative accesses for symbols known to be in a small data section (MIPS)`|
|`-MG`|Supported|`Add missing headers to depfile`|
|`-mhvx-length=<value>`|Unsupported|`Set Hexagon Vector Length`|
|`-mhvx=<value>`|Unsupported|`Enable Hexagon Vector eXtensions`|
|`-mhvx`|Unsupported|`Enable Hexagon Vector eXtensions`|
|`-miamcu`|Unsupported|`Use Intel MCU ABI`|
|`--migrate`|Unsupported|`Run the migrator`|
|`-mincremental-linker-compatible`|Supported|`(integrated-as) Emit an object file which can be used with an incremental linker`|
|`-mindirect-jump=<value>`|Unsupported|`Change indirect jump instructions to inhibit speculation`|
|`-mios-version-min=<value>`|Unsupported|`Set iOS deployment target`|
|`-MJ <value>`|Unsupported|`Write a compilation database entry per input`|
|`-mllvm <value>`|Supported|`Additional arguments to forward to LLVM's option processing`|
|`-mlocal-sdata`|Unsupported|`Extend the -G behaviour to object local data (MIPS)`|
|`-mlong-calls`|Supported|`Generate branches with extended addressability, usually via indirect jumps.`|
|`-mmacosx-version-min=<value>`|Unsupported|`Set Mac OS X deployment target`|
|`-mmadd4`|Supported|`Enable the generation of 4-operand madd.s, madd.d and related instructions.`|
|`-MMD`|Supported|`Write a depfile containing user headers`|
|`-mmemops`|Supported|`Enable generation of memop instructions`|
|`-mms-bitfields`|Unsupported|`Set the default structure layout to be compatible with the Microsoft compiler standard`|
|`-mmsa`|Unsupported|`Enable MSA ASE (MIPS only)`|
|`-mmt`|Unsupported|`Enable MT ASE (MIPS only)`|
|`-MM`|Supported|`Like -MMD, but also implies -E and writes to stdout by default`|
|`-mno-abicalls`|Unsupported|`Disable SVR4-style position-independent code (Mips only)`|
|`-mno-code-object-v3`|Supported|`Disable code object v3 (AMDGPU only)`|
|`-mno-crc`|Unsupported|`Disallow use of CRC instructions (Mips only)`|
|`-mno-embedded-data`|Unsupported|`Do not place constants in the .rodata section instead of the .sdata if they meet the -G <size> threshold (MIPS)`|
|`-mno-execute-only`|Unsupported|`Allow generation of data access to code sections (ARM only)`|
|`-mno-extern-sdata`|Unsupported|`Do not assume that externally defined data is in the small data if it meets the -G <size> threshold (MIPS)`|
|`-mno-fix-cortex-a53-835769`|Unsupported|`Don't workaround Cortex-A53 erratum 835769 (AArch64 only)`|
|`-mno-global-merge`|Supported|`Disable merging of globals`|
|`-mno-gpopt`|Unsupported|`Do not use GP relative accesses for symbols known to be in a small data section (MIPS)`|
|`-mno-hvx`|Unsupported|`Disable Hexagon Vector eXtensions`|
|`-mno-implicit-float`|Supported|`Don't generate implicit floating point instructions`|
|`-mno-incremental-linker-compatible`|Supported|`(integrated-as) Emit an object file which cannot be used with an incremental linker`|
|`-mno-local-sdata`|Unsupported|`Do not extend the -G behaviour to object local data (MIPS)`|
|`-mno-long-calls`|Supported|`Restore the default behaviour of not generating long calls`|
|`-mno-madd4`|Supported|`Disable the generation of 4-operand madd.s, madd.d and related instructions.`|
|`-mno-memops`|Supported|`Disable generation of memop instructions`|
|`-mno-movt`|Supported|`Disallow use of movt/movw pairs (ARM only)`|
|`-mno-ms-bitfields`|Supported|`Do not set the default structure layout to be compatible with the Microsoft compiler standard`|
|`-mno-msa`|Unsupported|`Disable MSA ASE (MIPS only)`|
|`-mno-mt`|Unsupported|`Disable MT ASE (MIPS only)`|
|`-mno-neg-immediates`|Supported|`Disallow converting instructions with negative immediates to their negation or inversion.`|
|`-mno-nvj`|Supported|`Disable generation of new-value jumps`|
|`-mno-nvs`|Supported|`Disable generation of new-value stores`|
|`-mno-outline`|Unsupported|`Disable function outlining (AArch64 only)`|
|`-mno-packets`|Supported|`Disable generation of instruction packets`|
|`-mno-relax`|Supported|`Disable linker relaxation`|
|`-mno-restrict-it`|Unsupported|`Allow generation of deprecated IT blocks for ARMv8. It is off by default for ARMv8 Thumb mode`|
|`-mno-sram-ecc`|Supported|`Disable SRAM ECC (AMDGPU only)`|
|`-mno-stack-arg-probe`|Supported|`Disable stack probes which are enabled by default`|
|`-mno-tls-direct-seg-refs`|Supported|`Disable direct TLS access through segment registers`|
|`-mno-unaligned-access`|Unsupported|`Force all memory accesses to be aligned (AArch32/AArch64 only)`|
|`-mno-xnack`|Supported|`Disable XNACK (AMDGPU only)`|
|`-mnocrc`|Unsupported|`Disallow use of CRC instructions (ARM only)`|
|`-mnvj`|Supported|`Enable generation of new-value jumps`|
|`-mnvs`|Supported|`Enable generation of new-value stores`|
|`-module-dependency-dir <value>`|Unsupported|`Directory to dump module dependencies to`|
|`-module-file-info`|Unsupported|`Provide information about a particular module file`|
|`-momit-leaf-frame-pointer`|Supported|`Omit frame pointer setup for leaf functions`|
|`-moutline`|Unsupported|`Enable function outlining (AArch64 only)`|
|`-mpackets`|Supported|`Enable generation of instruction packets`|
|`-mpie-copy-relocations`|Supported|`Use copy relocations support for PIE builds`|
|`-mprefer-vector-width=<value>`|Unsupported|`Specifies preferred vector width for auto-vectorization. Defaults to 'none' which allows target specific decisions.`|
|`-MP`|Supported|`Create phony target for each dependency (other than main file)`|
|`-mqdsp6-compat`|Unsupported|`Enable hexagon-qdsp6 backward compatibility`|
|`-MQ <value>`|Supported|`Specify name of main file output to quote in depfile`|
|`-mrelax-all`|Supported|`(integrated-as) Relax all machine instructions`|
|`-mrelax`|Supported|`Enable linker relaxation`|
|`-mrestrict-it`|Unsupported|`Disallow generation of deprecated IT blocks for ARMv8. It is on by default for ARMv8 Thumb mode.`|
|`-mrtd`|Unsupported|`Make StdCall calling convention the default`|
|`-msign-return-address=<value>`|Unsupported|`Select return address signing scope`|
|`-msoft-float`|Supported|`Use software floating point`|
|`-msram-ecc`|Supported|`Enable SRAM ECC (AMDGPU only)`|
|`-mstack-alignment=<value>`|Unsupported|`Set the stack alignment`|
|`-mstack-arg-probe`|Unsupported|`Enable stack probes`|
|`-mstack-probe-size=<value>`|Unsupported|`Set the stack probe size`|
|`-mstackrealign`|Unsupported|`Force realign the stack at entry to every function`|
|`-mthread-model <value>`|Supported|`The thread model to use, e.g. posix, single (posix by default)`|
|`-mtls-direct-seg-refs`|Supported|`Enable direct TLS access through segment registers (default)`|
|`-mtp=<value>`|Unsupported|`Read thread pointer from coprocessor register (ARM only)`|
|`-MT <value>`|Unsupported|`Specify name of main file output in depfile`|
|`-munaligned-access`|Unsupported|`Allow memory accesses to be unaligned (AArch32/AArch64 only)`|
|`-MV`|Supported|`Use NMake/Jom format for the depfile`|
|`-mxnack`|Supported|`Enable XNACK (AMDGPU only)`|
|`-M`|Supported|`Like -MD, but also implies -E and writes to stdout by default`|
|`--no-cuda-gpu-arch=<value>`|Supported|`Remove GPU architecture (e.g. sm_35) from the list of GPUs to compile for. 'all' resets the list to its default value.`|
|`--no-cuda-include-ptx=<value>`|Supported|`Do not include PTX for the following GPU architecture (e.g. sm_35) or 'all'. May be specified more than once.`|
|`--no-cuda-version-check`|Supported|`Don't error out if the detected version of the CUDA install is too low for the requested CUDA gpu architecture.`|
|`--no-system-header-prefix=<prefix>`|Supported|`Treat all #include paths starting with <prefix> as not including a system header.`|
|`-nobuiltininc`|Supported|`Disable builtin #include directories`|
|`-nostdinc++`|Unsupported|`Disable standard #include directories for the C++ standard library`|
|`-ObjC++`|Unsupported|`Treat source input files as Objective-C++ inputs`|
|`-objcmt-atomic-property`|Unsupported|`Make migration to 'atomic' properties`|
|`-objcmt-migrate-all`|Unsupported|`Enable migration to modern ObjC`|
|`-objcmt-migrate-annotation`|Unsupported|`Enable migration to property and method annotations`|
|`-objcmt-migrate-designated-init`|Unsupported|`Enable migration to infer NS_DESIGNATED_INITIALIZER for initializer methods`|
|`-objcmt-migrate-instancetype`|Unsupported|`Enable migration to infer instancetype for method result type`|
|`-objcmt-migrate-literals`|Unsupported|`Enable migration to modern ObjC literals`|
|`-objcmt-migrate-ns-macros`|Unsupported|`Enable migration to NS_ENUM/NS_OPTIONS macros`|
|`-objcmt-migrate-property-dot-syntax`|Unsupported|`Enable migration of setter/getter messages to property-dot syntax`|
|`-objcmt-migrate-property`|Unsupported|`Enable migration to modern ObjC property`|
|`-objcmt-migrate-protocol-conformance`|Unsupported|`Enable migration to add protocol conformance on classes`|
|`-objcmt-migrate-readonly-property`|Unsupported|`Enable migration to modern ObjC readonly property`|
|`-objcmt-migrate-readwrite-property`|Unsupported|`Enable migration to modern ObjC readwrite property`|
|`-objcmt-migrate-subscripting`|Unsupported|`Enable migration to modern ObjC subscripting`|
|`-objcmt-ns-nonatomic-iosonly`|Unsupported|`Enable migration to use NS_NONATOMIC_IOSONLY macro for setting property's 'atomic' attribute`|
|`-objcmt-returns-innerpointer-property`|Unsupported|`Enable migration to annotate property with NS_RETURNS_INNER_POINTER`|
|`-objcmt-whitelist-dir-path=<value>`|Unsupported|`Only modify files with a filename contained in the provided directory path`|
|`-ObjC`|Unsupported|`Treat source input files as Objective-C inputs`|
|`-o <file>`|Supported|`Write output to <file>`|
|`-pg`|Supported|`Enable mcount instrumentation`|
|`-pipe`|Supported|`Use pipes between commands, when possible`|
|`--precompile`|Supported|`Only precompile the input`|
|`-print-effective-triple`|Supported|`Print the effective target triple`|
|`-print-file-name=<file>`|Supported|`Print the full library path of <file>`|
|`-print-ivar-layout`|Unsupported|`Enable Objective-C Ivar layout bitmap print trace`|
|`-print-libgcc-file-name`|Supported|`Print the library path for the currently used compiler runtime library ("libgcc.a" or "libclang_rt.builtins.*.a")`|
|`-print-prog-name=<name>`|Supported|`Print the full program path of <name>`|
|`-print-resource-dir`|Supported|`Print the resource directory pathname`|
|`-print-search-dirs`|Supported|`Print the paths used for finding libraries and programs`|
|`-print-target-triple`|Supported|`Print the normalized target triple`|
|`-pthread`|Supported|`Support POSIX threads in generated code`|
|`--ptxas-path=<value>`|Unsupported|`Path to ptxas (used for compiling CUDA code)`|
|`-P`|Supported|`Disable linemarker output in -E mode`|
|`-Qn`|Supported|`Do not emit metadata containing compiler name and version`|
|`-Qunused-arguments`|Supported|`Don't emit warning for unused driver arguments`|
|`-Qy`|Supported|`Emit metadata containing compiler name and version`|
|`-relocatable-pch`|Supported|`Whether to build a relocatable precompiled header`|
|`-rewrite-legacy-objc`|Unsupported|`Rewrite Legacy Objective-C source to C++`|
|`-rewrite-objc`|Unsupported|`Rewrite Objective-C source to C++`|
|`-Rpass-analysis=<value>`|Supported|`Report transformation analysis from optimization passes whose name matches the given POSIX regular expression`|
|`-Rpass-missed=<value>`|Supported|`Report missed transformations by optimization passes whose name matches the given POSIX regular expression`|
|`-Rpass=<value>`|Supported|`Report transformations performed by optimization passes whose name matches the given POSIX regular expression`|
|`-rtlib=<value>`|Unsupported|`Compiler runtime library to use`|
|`-R<remark>`|Unsupported|`Enable the specified remark`|
|`-save-stats=<value>`|Supported|`Save llvm statistics.`|
|`-save-stats`|Supported|`Save llvm statistics.`|
|`-save-temps=<value>`|Supported|`Save intermediate compilation results.`|
|`-save-temps`|Supported|`Save intermediate compilation results`|
|`-serialize-diagnostics <value>`|Supported|`Serialize compiler diagnostics to a file`|
|`-shared-libsan`|Unsupported|`Dynamically link the sanitizer runtime`|
|`-static-libsan`|Unsupported|`Statically link the sanitizer runtime`|
|`-std=<value>`|Supported|`Language standard to compile for`|
|`-stdlib=<value>`|Supported|`C++ standard library to use`|
|`--system-header-prefix=<prefix>`|Supported|`Treat all #include paths starting with <prefix> as including a system header.`|
|`-S`|Supported|`Only run preprocess and compilation steps`|
|`--target=<value>`|Supported|`Generate code for the given target`|
|`-Tbss <addr>`|Supported|`Set starting address of BSS to <addr>`|
|`-Tdata <addr>`|Supported|`Set starting address of DATA to <addr>`|
|`-time`|Supported|`Time individual commands`|
|`-traditional-cpp`|Unsupported|`Enable some traditional CPP emulation`|
|`-trigraphs`|Supported|`Process trigraph sequences`|
|`-Ttext <addr>`|Supported|`Set starting address of TEXT to <addr>`|
|`-T <script>`|Unsupported|`Specify <script> as linker script`|
|`-undef`|Supported|`undef all system defines`|
|`-U <macro>`|Supported|`Undefine macro <macro>`|
|`--verify-debug-info`|Supported|`Verify the binary representation of debug output`|
|`-verify-pch`|Unsupported|`Load and verify that a pre-compiled header file is not stale`|
|`--version`|Supported|`Print version information`|
|`-v`|Supported|`Show commands to run and use verbose output`|
|`-Wa,<arg>`|Supported|`Pass the comma separated arguments in <arg> to the assembler`|
|`-Wdeprecated`|Supported|`Enable warnings for deprecated constructs and define __DEPRECATED`|
|`-Wl,<arg>`|Supported|`Pass the comma separated arguments in <arg> to the linker`|
|`-working-directory <value>`|Supported|`Resolve file paths relative to the specified directory`|
|`-Wp,<arg>`|Supported|`Pass the comma separated arguments in <arg> to the preprocessor`|
|`-W<warning>`|Supported|`Enable the specified warning`|
|`-w`|Supported|`Suppress all warnings`|
|`-Xanalyzer <arg>`|Supported|`Pass <arg> to the static analyzer`|
|`-Xassembler <arg>`|Supported|`Pass <arg> to the assembler`|
|`-Xclang <arg>`|Supported|`Pass <arg> to the clang compiler`|
|`-Xcuda-fatbinary <arg>`|Supported|`Pass <arg> to fatbinary invocation`|
|`-Xcuda-ptxas <arg>`|Supported|`Pass <arg> to the ptxas assembler`|
|`-Xlinker <arg>`|Supported|`Pass <arg> to the linker`|
|`-Xopenmp-target=<triple> <arg>`|Supported|`Pass <arg> to the target offloading toolchain identified by <triple>.`|
|`-Xopenmp-target <arg>`|Supported|`Pass <arg> to the target offloading toolchain.`|
|`-Xpreprocessor <arg>`|Supported|`Pass <arg> to the preprocessor`|
|`-x <language>`|Supported|`Treat subsequent input files as having type <language>`|
|`-z <arg>`|Supported|`Pass -z <arg> to the linker`|

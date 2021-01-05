# HIP testing environment.

This document explains how to use the HIP CMAKE testing environment.
We make use of the HIT Integrated Tester (HIT) framework to automatically find and add test cases to the CMAKE testing environment.

### Quickstart

HIP unit tests are integrated into the top-level cmake project. The tests depend upon the installed version of HIP.
Typical usage (paths relative to top of the HIP repo):
```
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_INSTALL_PREFIX=$PWD/install
$ make
$ make install
$ make build_tests
$ make test
```

### How to add a new test

The test infrastructure use a hierarchy of folders. So add the new test to the appropriate folder. 
The tests/src/runtimeApi/memory/hipMemset.cpp file contains a simple unit test and is a good starting point for other tests.
Copy this to a new test name and modify it.


### HIP Integrated Tester (HIT)

The HIT framework automatically finds and adds test cases to the CMAKE testing environment. It achives this by parsing all files in the tests/src folder.
The parser looks for a code block similar to the one below.
```
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * TEST: %t
 * //Small copy
 * TEST: %t -N 10    --memsetval 0x42
 * // Oddball size
 * TEST: %t -N 10013 --memsetval 0x5a
 * // Big copy
 * TEST: %t -N 256M  --memsetval 0xa6
 * HIT_END
 */
```
In the above, BUILD commands provide instructions on how to build the test case while TEST commands provide instructions on how to execute the test case.

#### BUILD command

The supported syntax for the BUILD command is:
```
BUILD: %t %s HIPCC_OPTIONS <hipcc_specific_options> CLANG_OPTIONS <clang_specific_options> NVCC_OPTIONS <nvcc_specific_options> EXCLUDE_HIP_PLATFORM <amd|nvidia|all> EXCLUDE_HIP_RUNTIME <rocclr> EXCLUDE_HIP_COMPILER <clang> DEPENDS EXCLUDE_HIP_LIB_TYPE <static|shared> <dependencies>
```
%s: refers to current source file name. Additional source files needed for the test can be specified by name (including relative path).
%t: refers to target executable named derived by removing the extension from the current source file. Alternatively a target executable name can be specified.
HIPCC_OPTIONS: All options specified after this delimiter are passed to hipcc on both amd and nvidia platforms.
CLANG_OPTIONS: All options specified after this delimiter are passed to hipcc on HIP-Clang compiler only.
NVCC_OPTIONS: All options specified after this delimiter are passed to hipcc on nvidia platform only.
EXCLUDE_HIP_PLATFORM: This can be used to exclude a test case from amd, nvidia or both platforms.
EXCLUDE_HIP_RUNTIME: This can be used to exclude a test case from rocclr runtime.
EXCLUDE_HIP_COMPILER: This can be used to exclude a test case from clang compiler.
EXCLUDE_HIP_RUNTIME AND EXCLUDE_HIP_COMPILER: when both options are specified it excludes test case from particular runtime and compiler.
EXCLUDE_HIP_LIB_TYPE: This can be used to exclude a test case from static or shared libs.
DEPENDS: This can be used to specify dependencies that need to be built before building the current target.


#### BUILD_CMD command

The supported syntax for the BUILD_CMD command is:
```
BUILD_CMD: <targetname> <build_command> EXCLUDE_HIP_PLATFORM <amd|nvidia|all> EXCLUDE_HIP_RUNTIME <rocclr> EXCLUDE_HIP_COMPILER <clang> EXCLUDE_HIP_LIB_TYPE <static|shared> DEPENDS <dependencies>
```
%s: refers to current source file name. Additional source files needed for the test can be specified by name (including relative path).
%t: refers to target executable named derived by removing the extension from the current source file. Alternatively a target executable name can be specified.
%hc: refers to hipcc pointed to by $CMAKE_INSTALL_PREFIX/bin/hipcc.
%hip-path: refers to hip installed location pointed to by $CMAKE_INSTALL_PREFIX
%cc: refers to system c compiler pointed to by /usr/bin/cc.
%cxx: refers to system c compiler pointed to by /usr/bin/c++.
%S: refers to path to current source file.
%T: refers to path to current build target.
EXCLUDE_HIP_PLATFORM: This can be used to exclude a test case from amd, nvidia or both platforms.
EXCLUDE_HIP_RUNTIME: This can be used to exclude a test case from rocclr runtime.
EXCLUDE_HIP_COMPILER: This can be used to exclude a test case from clang compiler.
EXCLUDE_HIP_RUNTIME AND EXCLUDE_HIP_COMPILER: when both options are specified it excludes test from particular runtime and compiler.
EXCLUDE_HIP_LIB_TYPE: This can be used to exclude a test case from static or shared libs.
DEPENDS: This can be used to specify dependencies that need to be built before building the current target.


#### TEST command

The supported syntax for the TEST command is:
```
TEST: %t <arguments_to_test_executable> EXCLUDE_HIP_PLATFORM <amd|nvidia|all> EXCLUDE_HIP_RUNTIME <rocclr> EXCLUDE_HIP_COMPILER <clang> EXCLUDE_HIP_LIB_TYPE <static|shared>
```
%t: refers to target executable named derived by removing the extension from the current source file. Alternatively a target executable name can be specified.
EXCLUDE_HIP_PLATFORM: This can be used to exclude a test case from amd, nvidia or both platforms. 
EXCLUDE_HIP_RUNTIME: This can be used to exclude a test case from rocclr runtime.
EXCLUDE_HIP_COMPILER: This can be used to exclude a test case from clang compiler.
EXCLUDE_HIP_RUNTIME AND EXCLUDE_HIP_COMPILER: when both options are specified it excludes test from particular runtime and compiler.
EXCLUDE_HIP_LIB_TYPE: This can be used to exclude a test case from static or shared libs.

Note that if the test has been excluded for a specific platform/runtime/compiler in the BUILD command, it is automatically excluded from the TEST command as well for the sameplatform.

#### TEST_NAMED command

When using the TEST command, HIT will squash and append the arguments specified to the test executable name to generate the CMAKE test name. Sometimes we might want to specify a more descriptive name. The TEST_NAMED command is used for that. The supported syntax for the TEST_NAMED command is:
```
TEST: %t CMAKE_TEST_NAME <arguments_to_test_executable> EXCLUDE_HIP_PLATFORM <amd|nvidia|all> EXCLUDE_HIP_RUNTIME <rocclr> EXCLUDE_HIP_COMPILER <clang> EXCLUDE_HIP_LIB_TYPE <static|shared>
```


### Running tests:
```
ctest
```

### Run subsets of all tests:
```
# Run one test on the commandline
./directed_tests/runtime/memory/hipMemset

# Run all the hipMemcpy tests:
ctest -R Memcpy

# Run all tests in a specific folder:
ctest -R memory
```

### Performance tests:
```
Above tests are direct tests which are majorly used for function verification.
We also provide performance tests under tests/performance folder.

# Build all performance tests after running "make install" under build folder:
make build_perf

Then all performance test applications will be built into ./performance_tests folder.

# Run all performance tests:
make perf

# Run individual performance test:
For example,
performance_tests/memory/hipPerfMemMallocCpyFree

# Run a specific test set:
For example,
/usr/bin/ctest -C performance -R performance_tests/perfDispatch --verbose
Here "-C performance" indicate the "performance" configuration of ctest.
```

### If a test fails - how to debug a test

Find the test and commandline that fail:

(From the build directory, perhaps hip/build)
grep -IR hipMemcpy-modes -IR ../tests/
../tests/src/runtimeApi/memory/hipMemcpy.cpp: * TEST_NAMED: %t hipMemcpy-modes --tests 0x1

# Guidelines for adding new tests

- Prefer to enhance an existing test as opposed to writing a new one. Tests have overhead to start and many small tests spend precious test time on startup and initialization issues.
- Make the test run standalone without requirement for command-line arguments.  THis makes it easier to debug since the name of the test is shown in the test report and if you know the name of the test you can the run the test.
- For long-running tests or tests with multiple phases, consider using the --tests option as an optional mechanism to allow debuggers to start with the failing subset of the test.


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

The HIT framework sutomatically finds and adds test cases to the CMAKE testing environment. It achives this by parsing all files in the tests/src folder.
The parser looks for a code block similar to the one below.
```
/* HIT_START
 * BUILD: %t %s ../../test_common.cpp
 * RUN: %t
 * //Small copy
 * RUN: %t -N 10    --memsetval 0x42
 * // Oddball size
 * RUN: %t -N 10013 --memsetval 0x5a
 * // Big copy
 * RUN: %t -N 256M  --memsetval 0xa6
 * HIT_END
 */
```
In the above, BUILD commands provide instructions on how to build the test case while RUN commands provide instructions on how to execute the test case.

#### BUILD command

The supported syntax for the BUILD command is:
```
BUILD: %t %s HIPCC_OPTIONS <hipcc_specific_options> HCC_OPTIONS <hcc_specific_options> NVCC_OPTIONS <nvcc_specific_options> EXCLUDE_HIP_PLATFORM <hcc|nvcc|all>
```
%s: refers to current source file name. Additional source files needed for the test can be specified by name (including relative path).
%t: refers to target executable named derived by removing the extension from the current source file. Alternatively a target executable name can be specified.
HIPCC_OPTIONS: All options specified after this delimiter are passed to hipcc on both HCC and NVCC platforms.
HCC_OPTIONS: All options specified after this delimiter are passed to hipcc on HCC platform only.
NVCC_OPTIONS: All options specified after this delimiter are passed to hipcc on NVCC platform only.
EXCLUDE_HIP_PLATFORM: This can be used to exclude a test case from HCC, NVCC or both platforms.


#### RUN command

The supported syntax for the RUN command is:
```
RUN: %t <arguments_to_test_executable> EXCLUDE_HIP_PLATFORM <hcc|nvcc|all>
```
%t: refers to target executable named derived by removing the extension from the current source file. Alternatively a target executable name can be specified.
EXCLUDE_HIP_PLATFORM: This can be used to exclude a test case from HCC, NVCC or both platforms. Note that if the test has been excluded for a specific platform in the BUILD command, it is automatically excluded from the RUN command as well for the same platform.


#### RUN_NAMED command

When using the RUN command, HIT will squash and append the arguments specified to the test executable name to generate the CMAKE test name. Sometimes we might want to specify a more descriptive name. The RUN_NAMED command is used for that. The supported syntax for the RUN_NAMED command is:
```
RUN: %t CMAKE_TEST_NAME <arguments_to_test_executable> EXCLUDE_HIP_PLATFORM <hcc|nvcc|all>
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


### If a test fails - how to debug a test

Find the test and commandline that fail:

(From the build directory, perhaps hip/build)
grep -IR hipMemcpy-modes -IR ../tests/
../tests/src/runtimeApi/memory/hipMemcpy.cpp: * RUN_NAMED: %t hipMemcpy-modes --tests 0x1

# Guidelines for adding new tests

- Prefer to enhance an existing test as opposed to writing a new one. Tests have overhead to start and many small tests spend precious test time on startup and initialization issues.
- Make the test run standalone without requirement for command-line arguments.  THis makes it easier to debug since the name of the test is shown in the test report and if you know the name of the test you can the run the test.
- For long-running tests or tests with multiple phases, consider using the --tests option as an optional mechanism to allow debuggers to start with the failing subset of the test.


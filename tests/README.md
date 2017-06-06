# HIP testing environment.

This document explains how to use the HIP CMAKE testing environment.  

### Quickstart
Usage :
```
$ mkdir build
$ cd build
$ cmake ../src
$ make
$ make test
```

### How to add a new test

The tests/src/runtimeApi/memory/hipMemtest.cpp file contains a simple unit test and is a good starting point for other tests.  
Copy this to a new test name and modify tests/src/CMakefiles.txt to add the test to the build environment.

Recent versions of the test infrastructure use a hierarchy of folders.  Each folder contains src and CMakefiles.txt file. 
See the CMakefiles.txt files for description of the intended purpose for each sub-directory.


#### Edit CMakefiles.txt:
// Example:
```
# Build the test executable:
build_hip_executable (hipMemset hipMemset.cpp) 


# This runs the tests with the specified command-line testing.  
# Multiple make_test may be specified.  
make_test(hipMemset " ")
```

It is recommended to place the build and run steps adjacent in the CMakefiles.txt.


### Running tests:
```
ctest
```

### Run subsets of all tests:
```
# Run one test on the commandline (obtain commandline parms from CMakefiles.tst)
./hipMemset

# Run all the memory tests:
ctest -R Memcpy
```


### If a test fails - how to debug a test

Find the test and commandline that fail:

(From the test build directory, perhaps hip/tests/build)
grep -IR hipMemcpy-modes -IR ../tests/
../tests/src/runtimeApi/memory/hipMemcpy.cpp: * RUN_NAMED: %t hipMemcpy-modes --tests 0x1

# Guidelines for adding new tests

- Prefer to enhance an existing test as opposed to writing a new one. Tests have overhead to start and many small tests spend precious test time on startup and initialization issues.
- Make the test run standalone without requirement for command-line arguments.  THis makes it easier to debug since the name of the test is shown in the test report and if you know the name of the test you can the run the test.
- For long-running tests or tests with multiple phases, consider using the --tests option as an optional mechanism to allow debuggers to start with the failing subset of the test.


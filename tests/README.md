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


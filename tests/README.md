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

Extract the commandline from the testing log:

(From the test build directory, perhaps hip/tests/build)
$ grep -A3 -m2  hipMemcpy-size  Testing/Temporary/LastTest.log
36/47 Testing: hipMemcpy-size
36/47 Test: hipMemcpy-size
Command: "/home/bensander/git/compute/external/hip/hip/tests/b6.hcc-LC.debug/runtimeApi/memory/hipMemcpy" "--tests" "0x6"
Directory: /home/bensander/git/compute/external/hip/hip/tests/b6.hcc-LC.debug/runtimeApi/memory

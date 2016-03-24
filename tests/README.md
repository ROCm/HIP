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

The tests/src/hipMemtest.cpp file contains a simple unit test and is a good starting point for other tests.  
Copy this to a new test name and modify tests/src/CMakefiles.txt to add the test to the build environment.

#### Edit CMakefiles.txt:
// Example:
```
make_hip_executable (hipMemset hipMemset.cpp) 
make_test(hipMemset " ")
```

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


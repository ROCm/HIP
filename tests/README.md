Tests uses CMAKE as teh build infrastructure.

Use :

> mkdir build
> cd build
> cmake ../src
> make
> make test


#-----
# How to add a new test;

# edit src/CMakeFiles to add the test:

# add the executable and list of required CPP files, ie:
# make_test (EXE CPP_FILES)
> make_hip_executable (hipMemset hipMemset.cpp) 

# Add to automated Test framework:
# make_test (TESTNAME ARGS)
> make_test(hipMemset " ")



# Running tests:
make test

# Run a specific test:
./hipMemset




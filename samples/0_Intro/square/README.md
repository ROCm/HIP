# square.md

Simple test which shows how to use hipify to port CUDA code to HIP.  Covered in more detail in blog.

1. Add hip/bin path to the PATH  :
    export PATH=$PATH:[MYHIP]/bin

2. hipify square.cu > square.cpp

3. Manually edit square.cpp to add hipLaunchParms lp to kernel parms:
    vector_square(hipLaunchParm lp, T *C_d, const T *A_d, size_t N)

    (see square.hipref.cpp for the correct output after running hipify and the above manual step)

4. make

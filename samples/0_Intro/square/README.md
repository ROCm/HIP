# Square.md

Simple test which shows how to use hipify to port CUDA code to HIP.  
See related [blog](http://gpuopen.com/hip-to-be-squared-an-introductory-hip-tutorial) that explains the example.

1. Add hip/bin path to the PATH  :
    <code>export PATH=$PATH:[MYHIP]/bin</code>

2. Do <code>$ hipify square.cu > square.cpp </code>

3. Manually edit square.cpp to add hipLaunchParms lp to kernel parms:
    <code>vector_square(hipLaunchParm lp, T *C_d, const T *A_d, size_t N)</code>

    (see square.hipref.cpp for the correct output after running hipify and the above manual step)

4. make

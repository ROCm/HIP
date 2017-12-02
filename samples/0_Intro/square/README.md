# Square.md

Simple test which shows how to use hipify to port CUDA code to HIP.  
See related [blog](http://gpuopen.com/hip-to-be-squared-an-introductory-hip-tutorial) that explains the example. 
Now it is even simpler and requires no manual modification to the hipified source code - just hipify and compile:

1. Add hip/bin path to the PATH  :
    <code>export PATH=$PATH:[MYHIP]/bin</code>

2. <code>$ make </code>
   Make runs these steps.  This can be performed on either CUDA or AMD platform:
   <code>hipify-perl square.cu > square.cpp </code>    # convert cuda code to hip code
   <code>hipcc square.cpp</code>                       # compile into executable

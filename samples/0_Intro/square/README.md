# Square.md

Simple test which shows how to use hipify-perl to port CUDA code to HIP.
See related [blog](http://gpuopen.com/hip-to-be-squared-an-introductory-hip-tutorial) that explains the example.
Now it is even simpler and requires no manual modification to the hipified source code - just hipify and compile:

- Add hip/bin path to the PATH

```
$ export PATH=$PATH:[MYHIP]/bin
```

- Define environment variable

```
$ export HIP_PATH=[MYHIP]
```

- Build executible file

```
$ cd ~/hip/samples/0_Intro/square
$ make
/home/user/hip/bin/hipify-perl square.cu > square.cpp
/home/user/hip/bin/hipcc  square.cpp -o square.out
/home/user/hip/bin/hipcc -use-staticlib  square.cpp -o square.out.static
```
- Execute file
```
$ ./square.out
info: running on device Navi 14 [Radeon Pro W5500]
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED!
```

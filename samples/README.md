Build procedure

We provide Makefile and CMakeLists.txt to build the samples seperately.

1.Makefile supports shared lib of hip-rocclr runtime and nvcc.

To build a sample, just type in sample folder,

make



2.CMakeLists.txt can support shared and static libs of hip-rocclr runtime.

To build a sample, type in sample folder,

mkdir build (if build folder is missing)

cd build

cmake ..

make

If you want debug version, follow,

cmake -DCMAKE_BUILD_TYPE=Debug ..
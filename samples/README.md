Build procedure

We provide Makefile and CMakeLists.txt to build the samples seperately.

1.Makefile supports shared lib of hip-rocclr runtime and nvcc.

To build a sample, just type in sample folder,

make



2.CMakeLists.txt can support shared and static libs of hip-rocclr runtime.

To build a sample, run in the sample folder,

mkdir -p build && cd build

rm -rf * (to clear up)

a. to build with shared libs, run

cmake ..

b. to build with static libs, run

cmake -DCMAKE_PREFIX_PATH="/opt/rocm/llvm/lib/cmake" ..

Then run,

make

Note that if you want debug version, add "-DCMAKE_BUILD_TYPE=Debug" in cmake cmd.
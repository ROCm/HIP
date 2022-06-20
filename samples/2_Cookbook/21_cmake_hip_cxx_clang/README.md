### This sample tests CXX Language support with amdclang++
I. Build
mkdir -p build; cd build
rm -rf *;
CXX=`hipconfig -l`/amdclang++ cmake .. (or)
cmake -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ .. (or)
cmake -DCMAKE_CXX_COMPILER=/opt/rocm-X.Y.Z/llvm/bin/amdclang++ ..
make

II. Test
$ ./square
info: running on device Vega 20 [Radeon Pro Vega 20]
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED!

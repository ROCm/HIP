### This will test linking hip::device interface in cmake
I. Build
mkdir -p build; cd build
rm -rf *; CXX=`hipconfig -l`/clang++ cmake ..
make

II. Test
$ ./test_cpp
info: running on device Vega 20 [Radeon Pro Vega 20]
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED!
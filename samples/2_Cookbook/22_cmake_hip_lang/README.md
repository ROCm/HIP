### This will test HIP language support in upstream CMake
I. Build
mkdir -p build; cd build
rm -rf *; cmake ..
make

II. Test
$ ./square
info: running on device
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED!

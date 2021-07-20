### This will test cmake lang support: CXX and Fortran
I. Prepare
1) You must install cmake version 3.18 or above to support LINK_LANGUAGE.
   Otherwise, Fortran build will fail.
   To download the latest cmake, see https://cmake.org/download/.
2) If there is no Fortran on your system, you must install it via,
   sudo apt install gfortran

II. Build
mkdir -p build; cd build
rm -rf *; CXX=`hipconfig -l`/clang++ FC=$(which gfortran) cmake ..
make

III. Test
# ./test_fortran
 Succeeded testing Fortran!

# ./test_cpp
Device name Device 66a7
PASSED!

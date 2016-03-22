# HIPIFY
The CLANG-based HIPIFY tool - translates CUDA to [HIP](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/blob/master/README.md)

# How to Build tool *in llvm source tree*
- get LLVM, CLANG and clang-tools-extra sources
- put hipify folder into llvm/tools/clang/tools/extra, add it to llvm/tools/clang/tools/extra/CMakeLists.txt
- run cmake and build tool
```
git clone http://llvm.org/git/llvm.git llvm
git clone http://llvm.org/git/clang.git llvm/tools/clang
git clone http://llvm.org/git/clang-tools-extra.git llvm/tools/clang/tools/extra
git clone https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP-hipify.git llvm/tools/clang/tools/extra/hipify
echo "add_subdirectory(hipify)" >> llvm/tools/clang/tools/extra/CMakeLists.txt

mkdir llvm_build && cd llvm_build
cmake -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" ../llvm
make
```

# How to Build *standalone* tool
- Get LLVM and CLANG sources, build them and install to a **LLVM_INSTALL_PATH** folder
- mkdir for hipify, run cmake there and build the tool:
```
git clone http://llvm.org/git/llvm.git llvm
git clone http://llvm.org/git/clang.git llvm/tools/clang

mkdir llvm_build && cd llvm_build
cmake -DCMAKE_INSTALL_PREFIX="LLVM_INSTALL_PATH" -DLLVM_INSTALL_UTILS=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" ../llvm
make && make install

git clone https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP-hipify.git path_to_hipify_src
mkdir path_to_hipify_src/build && cd path_to_hipify_src/build
cmake -DLLVM_DIR="LLVM_INSTALL_PATH" -DHIPIFY_STANDLONE=1 ..
make
```

# How to run tests (for *standalone* tool only)
- install Python and add python-setuptools
- install lit python script
- make sure that FileCheck util is installed to **LLVM_INSTALL_PATH/bin/FileCheck**
- run tests from path_to_hipify_src/build
```
sudo apt-get install python python-setuptools
sudo easy_install lit
make -C path_to_hipify_src/build test
```

# Notes
- To run, the tool needs "cuda minimal build" package:
 1. Download target installer from https://developer.nvidia.com/cuda-downloads. Choose "deb(network)" installer type to reduce downloaded packages size (we don't need the whole set)
 2. Run `sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb`
 3. Run `sudo apt-get update`
 4. Run `sudo apt-get install cuda-minimal-build-7-5` - this will install needed files, (without nvidia drivers, runtime, tools etc.)

# HIPIFY
The CLANG-based HIPIFY tool - translates CUDA to [HIP](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/blob/master/README.md)

# How to Build *standalone* tool
- Get LLVM and CLANG sources, build them and install to a **LLVM_INSTALL_PATH** folder
- mkdir for hipify, run cmake there and build the tool:
```
git clone http://llvm.org/git/llvm.git llvm
git clone http://llvm.org/git/clang.git llvm/tools/clang

mkdir llvm_build && cd llvm_build
cmake -DCMAKE_INSTALL_PREFIX="LLVM_INSTALL_PATH" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" ../llvm
make && make install

git clone https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP-hipify.git path_to_hipify_src
mkdir path_to_hipify_src/build && cd path_to_hipify_src/build
cmake -DLLVM_DIR="LLVM_INSTALL_PATH" -DHIPIFY_STANDLONE=1 ..
make
```

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
make -C tools/clang/tools/extra/hipify
```


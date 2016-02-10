# HIPIFY
The CLANG-based HIPIFY tool - translates CUDA to [HIP](https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/blob/master/README.md)

# How to Build
- Get LLVM and CLANG sources and build CLANG, then install it to a **LLVM_INSTALL_PATH** folder
- make build dir for hipify and run the folloing commands there:
```
> cmake -DLLVM_DIR="LLVM_INSTALL_PATH" path_to_hipify_src
> make
```

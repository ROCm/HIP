# Compile to LLVM IR and create an executable from modified IR

This sample shows how to generate the LLVM IR for a simple HIP source application, then re-compiling it and generating a valid HIP executable.

This sample uses a previous HIP application sample, please see [0_Intro/square](https://github.com/ROCm-Developer-Tools/HIP/blob/master/samples/0_Intro/square).

## Compiling the HIP source into LLVM IR
Using HIP flags `-c -emit-llvm` will help generate the host x86_64 and the device LLVM bitcode when paired with `--cuda-host-only` and `--cuda-device-only` respectively. In this sample we use these commands:
```
/opt/rocm/hip/bin/hipcc -c -emit-llvm --cuda-host-only -target x86_64-linux-gnu -o square_host.bc square.cpp
/opt/rocm/hip/bin/hipcc -c -emit-llvm --cuda-device-only --offload-arch=gfx900 --offload-arch=gfx906 square.cpp
```
The device LLVM IR bitcode will be output into two separate files:
- square-hip-amdgcn-amd-amdhsa-gfx900.bc
- square-hip-amdgcn-amd-amdhsa-gfx906.bc

You may modify `--offload-arch` flag to build other archs and choose to enable or disable xnack and sram-ecc.

To transform the LLVM bitcode into human readable LLVM IR, use these commands:
```
/opt/rocm/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx900.bc -o square-hip-amdgcn-amd-amdhsa-gfx900.ll
/opt/rocm/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx906.bc -o square-hip-amdgcn-amd-amdhsa-gfx906.ll
```

**Warning:** We cannot ensure any compiler besides the ROCm hipcc and clang will be compatible with this process. Also, there is no guarantee that the starting IR produced with `-x cl` will run with HIP runtime. Experimenting with other compilers or starting IR will be the responsibility of the developer.

## Modifying the LLVM IR
***Warning: The LLVM Language Specification may change across LLVM major releases, therefore the user must make sure the modified LLVM IR conforms to the LLVM Language Specification corresponding to the used LLVM version.***

At this point, you may evaluate the LLVM IR and make modifications if you are familiar with the LLVM IR language. Since the LLVM IR can vary between compiler versions, the safest approach would be to use the same compiler to consume the IR as the compiler producing it. It is the responsibility of the developer to ensure the IR is valid when manually modifying it.

## Compiling the LLVM IR into a valid HIP executable
If valid, the modified host and device IR may be compiled into a HIP executable. First, the readable IR must be compiled back in LLVM bitcode. The host IR can be compiled into an object using this command:
```
/opt/rocm/llvm/bin/llvm-as square_host.ll -o square_host.bc
/opt/rocm/hip/bin/hipcc -c square_host.bc -o square_host.o
```

However, the device IR will require a few extra steps. The device bitcodes needs to be compiled into device objects, then offload-bundled into a HIP fat binary using the clang-offload-bundler, then llvm-mc embeds the binary inside of a host object using the MC directives provided in `hip_obj_gen.mcin`. The output is a host object with an embedded device object. Here are the steps for device side compilation into an object:
```
/opt/rocm/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx900.ll -o square-hip-amdgcn-amd-amdhsa-gfx900.bc
/opt/rocm/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx906.ll -o square-hip-amdgcn-amd-amdhsa-gfx906.bc
/opt/rocm/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx900 square-hip-amdgcn-amd-amdhsa-gfx900.bc -o square-hip-amdgcn-amd-amdhsa-gfx900.o
/opt/rocm/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx906 square-hip-amdgcn-amd-amdhsa-gfx906.bc -o square-hip-amdgcn-amd-amdhsa-gfx906.o
/opt/rocm/hip/../llvm/bin/clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-gfx900,hip-amdgcn-amd-amdhsa-gfx906 -inputs=/dev/null,square-hip-amdgcn-amd-amdhsa-gfx900.o,square-hip-amdgcn-amd-amdhsa-gfx906.o -outputs=offload_bundle.hipfb
/opt/rocm/llvm/bin/llvm-mc hip_obj_gen.mcin -o square_device.o --filetype=obj
```

**Note:** Using option `-bundle-align=4096` only works on ROCm 4.0 and newer compilers. Also, the architecture must match the same arch as when compiling to LLVM IR.

Finally, using the system linker, hipcc, or clang, link the host and device objects into an executable:
```
/opt/rocm/hip/bin/hipcc square_host.o square_device.o -o square_ir.out
```
If you haven't modified the GPU archs, this executable should run on both `gfx900` and `gfx906`.

## How to build and run this sample:
Use these make commands to compile into LLVM IR, compile IR into executable, and execute it.
- To compile the HIP application into host and device LLVM IR: `make src_to_ir`.
- To disassembly the LLVM IR bitcode into human readable LLVM IR: `make bc_to_ll`.
- To assembly the human readable LLVM IR bitcode back into LLVM IR bitcode: `make ll_to_bc`.
- To compile the LLVM IR files into an executable: `make ir_to_exec`.
- To execute, run `./square_ir.out`.

**Note:** The default arch is `gfx900` and `gfx906`, this can be modified with make argument `GPU_ARCH1` and `GPU_ARCH2`.

## For More Information, please refer to the HIP FAQ.

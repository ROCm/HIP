# Compile to assembly and create an executable from modified asm

This sample shows how to generate the assembly code for a simple HIP source application, then re-compiling it and generating a valid HIP executable.

This sample uses a previous HIP application sample, please see [0_Intro/square](https://github.com/ROCm-Developer-Tools/HIP/blob/master/samples/0_Intro/square).

## Compiling the HIP source into assembly
Using HIP flags `-c -S` will help generate the host x86_64 and the device AMDGCN assembly code when paired with `--cuda-host-only` and `--cuda-device-only` respectively. In this sample we use these commands:
```
/opt/rocm/hip/bin/hipcc -c -S --cuda-host-only -target x86_64-linux-gnu -o square_host.s square.cpp
/opt/rocm/hip/bin/hipcc -c -S --cuda-device-only --offload-arch=gfx900 --offload-arch=gfx906 square.cpp
```

The device assembly will be output into two separate files:
- square-hip-amdgcn-amd-amdhsa-gfx900.s
- square-hip-amdgcn-amd-amdhsa-gfx906.s

You may modify `--offload-arch` flag to build other archs and choose to enable or disable xnack and sram-ecc.

**Note:** At this point, you may evaluate the assembly code, and make modifications if you are familiar with the AMDGCN assembly language and architecture.

## Compiling the assembly into a valid HIP executable
If valid, the modified host and device assembly may be compiled into a HIP executable. The host assembly can be compiled into an object using this command:
```
/opt/rocm/hip/bin/hipcc -c square_host.s -o square_host.o
```

However, the device assembly code will require a few extra steps. The device assemblies needs to be compiled into device objects, then offload-bundled into a HIP fat binary using the clang-offload-bundler, then llvm-mc embeds the binary inside of a host object using the MC directives provided in `hip_obj_gen.mcin`. The output is a host object with an embedded device object. Here are the steps for device side compilation into an object:
```
/opt/rocm/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx900 square-hip-amdgcn-amd-amdhsa-gfx900.s -o square-hip-amdgcn-amd-amdhsa-gfx900.o
/opt/rocm/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx906 square-hip-amdgcn-amd-amdhsa-gfx906.s -o square-hip-amdgcn-amd-amdhsa-gfx906.o
/opt/rocm/llvm/bin/clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-gfx900,hip-amdgcn-amd-amdhsa-gfx906 -inputs=/dev/null,square-hip-amdgcn-amd-amdhsa-gfx900.o,square-hip-amdgcn-amd-amdhsa-gfx906.o -outputs=offload_bundle.hipfb
/opt/rocm/llvm/bin/llvm-mc -triple x86_64-unknown-linux-gnu hip_obj_gen.mcin -o square_device.o --filetype=obj
```

**Note:** Using option `-bundle-align=4096` only works on ROCm 4.0 and newer compilers. Also, the architecture must match the same arch as when compiling to assembly.

Finally, using the system linker, hipcc, or clang, link the host and device objects into an executable:
```
/opt/rocm/hip/bin/hipcc square_host.o square_device.o -o square_asm.out
```
If you haven't modified the GPU archs, this executable should run on both `gfx900` and `gfx906`.

## How to build and run this sample:
Use these make commands to compile into assembly, compile assembly into executable, and execute it.
- To compile the HIP application into host and device assembly: `make src_to_asm`.
- To compile the assembly files into an executable: `make asm_to_exec`.
- To execute, run `./square_asm.out`.

**Note:** The default arch is `gfx900` and `gfx906`, this can be modified with make argument `GPU_ARCH1` and `GPU_ARCH2`.

## For More Information, please refer to the HIP FAQ.

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc
CLANG=$(HIP_PATH)/../llvm/bin/clang
LLVM_MC=$(HIP_PATH)/../llvm/bin/llvm-mc
CLANG_OFFLOAD_BUNDLER=$(HIP_PATH)/../llvm/bin/clang-offload-bundler

SRCS=square.cpp

# Extracting ASM code, then creating an executable with the modified asm.

SQ_HOST_ASM=square_host.s
SQ_HOST_OBJ=square_host.o
SQ_DEVICE_HIPFB=offload_bundle.hipfb
SQ_DEVICE_OBJ=square_device.o
SQ_ASM_EXE=square_asm.out

MCIN_OBJ_GEN=hip_obj_gen.mcin
GPU_ARCH1=gfx900
GPU_ARCH2=gfx906

.PHONY: test

all: src_to_asm asm_to_exec

src_to_asm:
	$(HIPCC) -c -S --cuda-host-only -target x86_64-linux-gnu -o $(SQ_HOST_ASM) $(SRCS)
	$(HIPCC) -c -S --cuda-device-only --offload-arch=$(GPU_ARCH1) --offload-arch=$(GPU_ARCH2) $(SRCS)

# You may modify the .s assembly files before the next step
# By default, their names will be:
#   square-hip-amdgcn-amd-amdhsa-gfx900.s
#   square-hip-amdgcn-amd-amdhsa-gfx906.s
#
# Note: hipcc does not work to convert .s to .o, use clang instead.

asm_to_exec:
	$(HIPCC) -c $(SQ_HOST_ASM) -o $(SQ_HOST_OBJ)
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH1) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).s -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH2) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).s -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).o
	$(CLANG_OFFLOAD_BUNDLER) -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-$(GPU_ARCH1),hip-amdgcn-amd-amdhsa-$(GPU_ARCH2) -inputs=/dev/null,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).o -outputs=$(SQ_DEVICE_HIPFB)
	$(LLVM_MC) $(MCIN_OBJ_GEN) -o $(SQ_DEVICE_OBJ) --filetype=obj
	$(HIPCC) $(SQ_HOST_OBJ) $(SQ_DEVICE_OBJ) -o $(SQ_ASM_EXE)

clean:
	rm -f *.o *.out *.hipfb *.s *.ll *.bc


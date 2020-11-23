HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc
CLANG=$(HIP_PATH)/../llvm/bin/clang
LLVM_MC=$(HIP_PATH)/../llvm/bin/llvm-mc
CLANG_OFFLOAD_BUNDLER=$(HIP_PATH)/../llvm/bin/clang-offload-bundler
LLVM_AS=$(HIP_PATH)/../llvm/bin/llvm-as
LLVM_DIS=$(HIP_PATH)/../llvm/bin/llvm-dis

SRCS=square.cpp

# Extracting the IR code, then creating an executable with the modified IR.

SQ_HOST_BC=square_host.bc
SQ_HOST_LL=square_host.ll
SQ_HOST_OBJ=square_host.o
SQ_DEVICE_OBJ=square_device.o
SQ_DEVICE_HIPFB=offload_bundle.hipfb
SQ_IR_EXE=square_ir.out

MCIN_OBJ_GEN=hip_obj_gen.mcin
GPU_ARCH1=gfx900
GPU_ARCH2=gfx906

.PHONY: test

all: src_to_ir bc_to_ll ll_to_bc ir_to_exec

src_to_ir:
	$(HIPCC) -c -emit-llvm --cuda-host-only -target x86_64-linux-gnu -o $(SQ_HOST_BC) $(SRCS)
	$(HIPCC) -c -emit-llvm --cuda-device-only --offload-arch=$(GPU_ARCH1) --offload-arch=$(GPU_ARCH2) $(SRCS)

# By default, the LLVM IR Bitcode file names will be:
#   square-hip-amdgcn-amd-amdhsa-gfx900.bc
#   square-hip-amdgcn-amd-amdhsa-gfx906.bc

bc_to_ll:
	$(LLVM_DIS) $(SQ_HOST_BC) -o $(SQ_HOST_LL)
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).ll
	$(LLVM_DIS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).ll

# You may modify the .ll LLVM IR files before the next step
#
# Note: hipcc does not work to convert .bc to .o, use clang instead.

ll_to_bc:
	$(LLVM_AS) $(SQ_HOST_LL) -o $(SQ_HOST_BC)
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).bc
	$(LLVM_AS) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).ll -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).bc

ir_to_exec:
	$(HIPCC) -c $(SQ_HOST_BC) -o $(SQ_HOST_OBJ)
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH1) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).o
	$(CLANG) -target amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH2) square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).bc -o square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).o
	$(CLANG_OFFLOAD_BUNDLER) -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-$(GPU_ARCH1),hip-amdgcn-amd-amdhsa-$(GPU_ARCH2) -inputs=/dev/null,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH1).o,square-hip-amdgcn-amd-amdhsa-$(GPU_ARCH2).o -outputs=$(SQ_DEVICE_HIPFB)
	$(LLVM_MC) $(MCIN_OBJ_GEN) -o $(SQ_DEVICE_OBJ) --filetype=obj
	$(HIPCC) $(SQ_HOST_OBJ) $(SQ_DEVICE_OBJ) -o $(SQ_IR_EXE)

clean:
	rm -f *.o *.out *.hipfb *.s *.ll *.bc


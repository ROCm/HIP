HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
OPT=
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)

all: vcpy_kernel.compile runKernel.hip.out

runKernel.hip.out: runKernel.cpp
	$(HIPCC) $(OPT) runKernel.cpp -o runKernel.hip.out

ifeq (${HIP_PLATFORM}, hcc)

vcpy_kernel.compile: vcpy_kernel.cpp
	$(HIPCC) --genco --target-isa=fiji vcpy_kernel.cpp -o vcpy_kernel.co

clean:
	rm -f *.co *.out

endif

ifeq (${HIP_PLATFORM}, nvcc)

vcpy_kernel.compile: vcpy_kernel.cpp
	$(HIPCC) --genco vcpy_kernel.cpp -o vcpy_kernel.ptx

clean:
	rm -f *.ptx *.out

endif


HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
OPT=
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)

ifeq (${HIP_PLATFORM}, hcc)
	GENCODEOBJECT_FLAGS=--target-isa=fiji

all: runKernel.hip.out

vcpy_isa.compile: vcpy_isa.cpp
	$(HIPCC) --genco $(GENCODEOBJECT_FLAGS) vcpy_isa.cpp -o vcpy_isa.co

clean:
	rm -f *.co *.out

endif

ifeq (${HIP_PLATFORM}, nvcc)

vcpy_isa.compile: vcpy_isa.cu
	$(HIPCC) --genco vcpy_isa.cu -o vcpy_isa.ptx

clean:
	rm -f *.ptx *.out

endif

all: vcpy_isa.compile runKernel.hip.out

runKernel.hip.out: runKernel.cpp
	$(HIPCC) $(OPT) runKernel.cpp -o runKernel.hip.out


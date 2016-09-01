HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)

ifeq (${HIP_PLATFORM}, hcc)
	GENCODEOBJECT_FLAGS=--target-isa-fiji
endif

all: vcpy_isa.compile runKernel.hip.out

vcpy_isa.compile: vcpy_isa.cpp
	$(HIPCC) --gencodeobject $(GENCODEOBJECT_FLAGS) vcpy_isa.cpp -o vcpy_isa.co

runKernel.hip.out: runKernel.cpp
	$(HIPCC) runKernel.cpp -o runKernel.hip.out

clean:
	rm -f *.co *.out

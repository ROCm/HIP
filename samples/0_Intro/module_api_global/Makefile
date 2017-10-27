HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)

all: vcpy_kernel.code runKernel.hip.out

runKernel.hip.out: runKernel.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

vcpy_kernel.code: vcpy_kernel.cpp
	$(HIPCC)  --genco  $(GENCO_FLAGS) $^ -o $@

clean:
	rm -f *.code *.out

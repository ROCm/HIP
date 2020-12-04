HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)

EXE_TARGETS=runKernel launchKernelHcc defaultDriver

TARGETS=${EXE_TARGETS} vcpy_kernel.code

all: ${TARGETS}

clean:
	rm -f $(TARGETS) *.o

run: $(TARGETS)
	@for t in $(EXE_TARGETS); do\
		echo "Running $$t"; \
		./$$t; \
	done

runKernel: runKernel.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

launchKernelHcc: launchKernelHcc.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

defaultDriver: defaultDriver.cpp
	$(HIPCC)  $(HIPCC_FLAGS) $< -o $@

vcpy_kernel.code: vcpy_kernel.cpp
	$(HIPCC)  --genco  $(GENCO_FLAGS) $^ -o $@

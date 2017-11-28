HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)

all: tex2dKernel.code texture2dDrv.out

texture2dDrv.out: texture2dDrv.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@
	
tex2dKernel.code: tex2dKernel.cpp
	$(HIPCC)  --genco  $(GENCO_FLAGS) $^ -o $@

clean:
	rm -f *.code *.out

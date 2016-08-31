HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

all: vcpy_isa.compile runKernel.hip.out

runKernel.cuda.out: runKernel.cpp
	nvcc runKernel.cpp -o $@

vcpy_isa.compile: vcpy_isa.cpp
	$(HIPCC) --genisa --target-isa=fiji vcpy_isa.cpp -o vcpy_isa.co	

runKernel.hip.out: runKernel.cpp
	$(HIPCC) runKernel.cpp -o runKernel.hip.out

clean:
	rm -f *.co *.out

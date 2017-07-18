HIP_PATH?= $(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc

all: square.hip.out

HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)

ifeq (${HIP_PLATFORM}, hcc)
	HIPCC_FLAGS = --amdgpu-target=gfx701 --amdgpu-target=gfx801 --amdgpu-target=gfx803 --amdgpu-target=gfx900
endif

square.cuda.out : square.cu
	nvcc square.cu -o $@

#hipify square.cu > square.cpp
# Then review & finish port in square.cpp
#

square.hip.out: square.hipref.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(CXXFLAGS) square.hipref.cpp -o $@




clean:
	rm -f *.o *.out	

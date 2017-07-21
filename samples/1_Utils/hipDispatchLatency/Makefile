HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)

ifeq (${HIP_PLATFORM}, hcc)
	HIPCC_FLAGS = --amdgpu-target=gfx701 --amdgpu-target=gfx801 --amdgpu-target=gfx803 --amdgpu-target=gfx900
endif

EXE=hipDispatchLatency

CXXFLAGS = -O3

all: ${EXE}

$(EXE): hipDispatchLatency.cpp ResultDatabase.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(CXXFLAGS) hipDispatchLatency.cpp ResultDatabase.cpp -o $@

install: $(EXE)
	cp $(EXE) $(HIP_PATH)/bin

clean:
	rm -f *.o $(EXE)

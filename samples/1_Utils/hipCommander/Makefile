HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)

ifeq (${HIP_PLATFORM}, hcc)
	HIPCC_FLAGS = --amdgpu-target=gfx701 --amdgpu-target=gfx801 --amdgpu-target=gfx803 --amdgpu-target=gfx900
endif

EXE=hipCommander
OPT=-O3
#CXXFLAGS = -O3 -g
CXXFLAGS =  $(OPT) --std=c++11

CODE_OBJECTS=nullkernel.hsaco

all: ${EXE} ${CODE_OBJECTS}

$(EXE): hipCommander.cpp 
	$(HIPCC) $(HIPCC_FLAGS) $(CXXFLAGS) $^ -o $@

nullkernel.hsaco : nullkernel.hip.cpp
	$(HIPCC) $(HIPCC_FLAGS) --genco nullkernel.hip -o nullkernel.hsaco


install: $(EXE)
	cp $(EXE) $(HIP_PATH)/bin


clean:
	rm -f *.o *.co $(EXE)

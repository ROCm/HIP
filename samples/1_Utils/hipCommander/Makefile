HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=hipCommander
OPT=-O3
#CXXFLAGS = -O3 -g
CXXFLAGS =  $(OPT) --std=c++11

HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)

CODE_OBJECTS=nullkernel.hsaco

all: ${EXE} ${CODE_OBJECTS}

$(EXE): hipCommander.cpp 
	$(HIPCC) $(CXXFLAGS) $^ -o $@

nullkernel.hsaco : nullkernel.hip.cpp
	$(HIPCC) --genco nullkernel.hip -o nullkernel.hsaco


install: $(EXE)
	cp $(EXE) $(HIP_PATH)/bin


clean:
	rm -f *.o *.co $(EXE)

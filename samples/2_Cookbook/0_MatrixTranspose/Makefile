HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)

ifeq (${HIP_PLATFORM}, hcc)
	HIPCC_FLAGS = --amdgpu-target=gfx701 --amdgpu-target=gfx801 --amdgpu-target=gfx803 --amdgpu-target=gfx900
endif

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./MatrixTranspose

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g
CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(HIPCC_FLAGS) $(OBJECTS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)


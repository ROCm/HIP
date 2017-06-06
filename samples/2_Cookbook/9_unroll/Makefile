HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

ifeq (gfx701, $(findstring gfx701,$(HCC_AMDGPU_TARGET)))
	$(error gfx701 is not a supported device for this sample)
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = unroll.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./unroll

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g
CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

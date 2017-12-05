HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

ifeq (${HIP_PLATFORM}, nvcc)
	SOURCES=square.cu
else
	SOURCES=square.cpp
endif

all: square.out

# Step
square.cpp: square.cu
	$(HIP_PATH)/bin/hipify-perl square.cu > square.cpp

square.out: $(SOURCES)
	$(HIPCC) $(CXXFLAGS) $(SOURCES) -o $@

clean:
	rm -f *.o *.out square.cpp

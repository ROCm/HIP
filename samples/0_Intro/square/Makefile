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

TARGETS=square

all: $(TARGETS)

run: $(TARGETS)
	@for t in $(TARGETS); do\
		echo "Running $$t"; \
		./$$t; \
	done

clean:
	rm -f $(TARGETS) *.o square.cpp

square.cpp: square.cu
	$(HIP_PATH)/bin/hipify-perl square.cu > square.cpp

square: $(SOURCES)
	$(HIPCC) $(CXXFLAGS) $(SOURCES) -o $@

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

# Show how to use PLATFORM to specify different options for each compiler:
ifeq (${HIP_PLATFORM}, nvcc)
	HIPCC_FLAGS = -gencode=arch=compute_20,code=sm_20
endif

TARGETS=bit_extract

.PHONY: all clean run

all: $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

run: $(TARGETS)
	@for t in $(TARGETS); do\
		echo "Running $$t"; \
		./$$t; \
	done

bit_extract: bit_extract.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

#Dependencies : [MYHIP]/bin must be in user's path.

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

EXE=bit_extract
EXE_STATIC=bit_extract_static

$(EXE): bit_extract.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@

$(EXE_STATIC): bit_extract.cpp
	$(HIPCC) -use-staticlib  $(HIPCC_FLAGS) $< -o $@

all: $(EXE) $(EXE_STATIC)

clean:
	rm -f *.o $(EXE) $(EXE_STATIC)

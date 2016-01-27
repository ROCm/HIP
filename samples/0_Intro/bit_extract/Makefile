#Dependencies : [MYHIP]/bin must be in user's path.

HIP_PATH?=$(shell hipconfig --path)
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

# Show how to use PLATFORM to specify different options for each compiler:
ifeq (${HIP_PLATFORM}, nvcc)
	HIPCC_FLAGS = -gencode=arch=compute_20,code=sm_20 
endif
ifeq (${HIP_PLATFORM}, hcc)
	HIPCC_FLAGS =
endif


EXE=bit_extract

$(EXE): bit_extract.cpp
	$(HIPCC) $(HIPCC_FLAGS) $< -o $@


clean:
	rm -f *.o $(EXE)
	

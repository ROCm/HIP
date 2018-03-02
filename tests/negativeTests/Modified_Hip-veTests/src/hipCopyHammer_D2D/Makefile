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

EXE=hipCopyHammer_D2D

$(EXE): hipCopyHammer_D2D.cpp
	$(HIPCC) $(HIPCC_FLAGS) -g $< -o $@


clean:
	rm -f *.o $(EXE)
	

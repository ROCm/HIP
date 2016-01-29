#must also be specified in doxygen-input/doxy.cfg, 

HIP_PATH ?= .
HIPCC = $(HIP_PATH)/bin/hipcc
HSA_PATH ?= /opt/hsa


HIP_SOURCES = $(HIP_PATH)/src/hip_hcc.cpp 
HIP_OBJECTS = $(HIP_SOURCES:.cpp=.o)

$(HIP_OBJECTS): HIPCC_FLAGS += -I$(HSA_PATH)/include

$(HIP_OBJECTS): 

%.o:: %.cpp
	    $(HIPCC) $(HIPCC_FLAGS) $< -c -O3 -o $@


clean:
	rm -f $(HIP_OBJECTS)



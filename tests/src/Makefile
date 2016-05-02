HIP_PATH=..

TARGET=hcc
include $(HIP_PATH)/examples/common/hip.prologue.make

SOURCES = hipMemset.cpp 
OBJECTS=$(SOURCES:.cpp=.o)

EXECUTABLE=hipMemset

$(EXECUTABLE): $(HIP_DEPS) $(OBJECTS) 
    $(HCC) $(HLDFLAGS) $(OBJECTS) -o $@

.cpp.o:
    $(HCC) $(HCFLAGS)   -c $< -o $@   
    @$(CC) -MM -MT $@ $(CFLAGS)    -c $< > $(@:.o=.d)

clean: hip_clean
    rm -rf $(EXECUTABLE) $(OBJECTS)

include $(HIP_PATH)/examples/common/hip.epilogue.make

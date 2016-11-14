HIP_PATH?= $(wildcard /opt/rocm/hip)

HIPCC=$(HIP_PATH)/bin/hipcc


HIPPROFILER=/opt/rocm/bin/rocm-profiler
PROFILER_OPT=-A -o MT.atp -e HIP_PROFILE_API=1
HIPPROFILER_POST_CMD=$(HIP_PATH)/bin/hipdemangleatp MT.atp

TARGET=hcc

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./MatrixTranspose

.PHONY: test


all: $(EXECUTABLE) profile



OPT =-g 
CXXFLAGS =$(OPT) 
CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@


profile: $(EXECUTABLE)
	$(HIPPROFILER) $(PROFILER_OPT) $(EXECUTABLE)
	$(HIPPROFILER_POST_CMD)


# Pass option to control start and stop iterations for profiling - see MatrixTranspose.cpp for implementation:
# Note we start profiler in --startdisabled mode - no timing collected until app enabled it via hipProfilerStart()
profile_trigger: $(EXECUTABLE)
	$(HIPPROFILER) $(PROFILER_OPT)  --startdisabled   $(EXECUTABLE) 3 6
	$(HIPPROFILER_POST_CMD)


run: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o


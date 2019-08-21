HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./MatrixTranspose

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-O3


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(CXXFLAGS)  $(OBJECTS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o


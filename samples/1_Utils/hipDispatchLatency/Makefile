HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=hipDispatchLatency

CXXFLAGS = -O3

all: ${EXE}

$(EXE): hipDispatchLatency.cpp ResultDatabase.cpp
	$(HIPCC) $(CXXFLAGS) hipDispatchLatency.cpp ResultDatabase.cpp -o $@

install: $(EXE)
	cp $(EXE) $(HIP_PATH)/bin

clean:
	rm -f *.o $(EXE)

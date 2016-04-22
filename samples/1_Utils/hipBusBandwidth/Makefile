HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=hipBusBandwidth
CXXFLAGS = -O3 -g

all: install

$(EXE): hipBusBandwidth.cpp ResultDatabase.cpp
	$(HIPCC) $(CXXFLAGS) $^ -o $@

install: $(EXE)
	cp $(EXE) $(HIP_PATH)/bin


clean:
	rm -f *.o $(EXE)

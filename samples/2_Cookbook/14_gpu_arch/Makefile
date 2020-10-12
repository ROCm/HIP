HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=./gpuarch

.PHONY: test

all: test

$(EXE): gpuarch.cpp
	$(HIPCC) $^ -o $@

test: $(EXE)
	$(EXE)

clean:
	rm -f *.o $(EXE)

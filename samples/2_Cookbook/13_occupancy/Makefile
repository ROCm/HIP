HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=./occupancy

.PHONY: test

all: test

$(EXE): occupancy.cpp
	$(HIPCC) $^ -o $@

test: $(EXE)
	$(EXE)

clean:
	rm -f *.o $(EXE)

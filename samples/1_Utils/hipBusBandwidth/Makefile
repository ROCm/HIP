HIP_PATH?=$(shell hipconfig -p)
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=hipBusBandwidth

all: install

$(EXE): hipBusBandwidth.cpp ResultDatabase.cpp
	$(HIPCC) $^ -o $@

install: $(EXE)
	cp $(EXE) $(HIP_PATH)/bin


clean:
	rm -f *.o $(EXE)

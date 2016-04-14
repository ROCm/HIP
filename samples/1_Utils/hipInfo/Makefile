HIP_PATH?=../../..
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=hipInfo

all: install

$(EXE): hipInfo.cpp
	$(HIPCC) hipInfo.cpp -o $@

install: $(EXE)
	cp $(EXE) $(HIP_PATH)/bin


clean:
	rm -f *.o $(EXE)

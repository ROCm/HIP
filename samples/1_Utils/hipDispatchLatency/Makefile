HIP_PATH=../../..
HIPCC=$(HIP_PATH)/bin/hipcc

EXE=hipDispatchLatency

all: install

$(EXE): hipDispatchLatency.cpp
	$(HIPCC) hipDispatchLatency.cpp ResultDatabase.cpp -o $@

install: $(EXE)
	cp $(EXE) $(HIP_PATH)/bin

clean:
	rm -f *.o $(EXE)

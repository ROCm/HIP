HIP_PATH?=../../..
HIPCC=$(HIP_PATH)/bin/hipcc

all: square.hip.out

square.cuda.out : square.cu
	nvcc square.cu -o $@

#hipify square.cu > square.cpp
# Then review & finish port in square.cpp

square.hip.out: square.hipref.cpp
	$(HIPCC) square.hipref.cpp -o $@



clean:
	rm -f *.o *.out	

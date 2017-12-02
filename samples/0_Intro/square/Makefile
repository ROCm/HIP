HIP_PATH?= $(wildcard /opt/rocm/hip)
HIPCC=$(HIP_PATH)/bin/hipcc

all: square.out

# Step
square.cpp:
	$(HIP_PATH)/bin/hipify-perl square.cu > square.cpp

square.out: square.cpp
	$(HIPCC) $(CXXFLAGS) square.cpp -o $@

# make step to make native cuda app, if on cuda machine
square.cuda.out : square.cu
	nvcc square.cu -o $@
clean:
	rm -f *.o *.out	

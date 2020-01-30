HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc -std=c++11

CXXFLAGS = -O3

all: test_kernel.code hipDispatchLatency.out singleThreadMultiStream.out

hipDispatchLatency.out: hipDispatchLatency.cpp
	$(HIPCC) $(CXXFLAGS) hipDispatchLatency.cpp -o $@
singleThreadMultiStream.out: singleThreadMultiStream.cpp
	$(HIPCC) $(CXXFLAGS) singleThreadMultiStream.cpp -o $@

test_kernel.code: test_kernel.cpp
	$(HIP_PATH)/bin/hipcc --genco  $(GENCO_FLAGS) $^ -o $@
clean:
	rm -f *.code *.out

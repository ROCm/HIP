HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc -std=c++11

CXXFLAGS = -O3

all: test_kernel.code singleThreadNullStream.out singleThreadMultiStream.out multiThreadMultiStream.out

singleThreadNullStream.out: singleThreadNullStream.cpp
	$(HIPCC) $(CXXFLAGS) singleThreadNullStream.cpp -o $@
singleThreadMultiStream.out: singleThreadMultiStream.cpp
	$(HIPCC) $(CXXFLAGS) singleThreadMultiStream.cpp -o $@
multiThreadMultiStream.out: multiThreadMultiStream.cpp
	$(HIPCC) $(CXXFLAGS) multiThreadMultiStream.cpp -o $@

test_kernel.code: test_kernel.cpp
	$(HIP_PATH)/bin/hipcc --genco  $(GENCO_FLAGS) $^ -o $@
clean:
	rm -f *.code *.out

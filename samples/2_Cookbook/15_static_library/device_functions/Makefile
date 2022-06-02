ifeq ($(OS),Windows_NT)
 $(error Makefile is not supported on windows platform. Please use cmake instead to build sample.)
endif
ROCM_PATH?= $(wildcard /opt/rocm/)
HIP_PATH?= $(wildcard $(ROCM_PATH)/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

.PHONY: test

all: $(RDC_EXE) test

STATIC_LIB_SRC=hipDevice.cpp
STATIC_LIB=./libHipDevice.a
STATIC_MAIN_SRC=hipMain2.cpp
RDC_EXE=./test_device_static.out

$(STATIC_LIB):
	$(HIPCC) $(STATIC_LIB_SRC) -c -fgpu-rdc -fPIC -o hipDevice.o
	ar rcsD $@ hipDevice.o

# Compiles hipMain2 with hipcc and links with libHipDevice.a which contains device function.
$(RDC_EXE): $(STATIC_LIB)
	$(HIPCC) $(STATIC_LIB) $(STATIC_MAIN_SRC) -fgpu-rdc -o $@

test: $(RDC_EXE)
	$(RDC_EXE)

clean:
	rm -f $(RDC_EXE)
	rm -f $(STATIC_LIB)
	rm -f *.o

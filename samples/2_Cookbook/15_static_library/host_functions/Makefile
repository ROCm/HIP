ifeq ($(OS),Windows_NT)
 $(error Makefile is not supported on windows platform. Please use cmake instead to build sample.)
endif
ROCM_PATH?= $(wildcard /opt/rocm/)
HIP_PATH?= $(wildcard $(ROCM_PATH)/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc
GXX=g++

EMIT_STATIC_LIB_SRC=hipOptLibrary.cpp
EMIT_STATIC_LIB=./libHipOptLibrary.a
EMIT_STATIC_MAIN_SRC=hipMain1.cpp
HIPCC_EXE=./test_emit_static_hipcc_linker.out
HOST_EXE=./test_emit_static_host_linker.out

.PHONY: test

all: $(HIPCC_EXE) $(HOST_EXE) test

$(EMIT_STATIC_LIB):
	$(HIPCC) $(EMIT_STATIC_LIB_SRC) --emit-static-lib -fPIC -o $@

# Compiles hipMain1 with hipcc and links with libHipOptLibrary.a which contains host function.
$(HIPCC_EXE): $(EMIT_STATIC_LIB)
	$(HIPCC) $(EMIT_STATIC_MAIN_SRC) -L. -lHipOptLibrary -o $@

# Compiles hipMain1 with g++ and links with libHipOptLibrary.a which contains host function.
$(HOST_EXE): $(EMIT_STATIC_LIB)
	$(GXX) $(EMIT_STATIC_MAIN_SRC) -L. -lHipOptLibrary -L$(HIP_PATH)/lib -lamdhip64 -Wl,-rpath=$(HIP_PATH)/lib -o $@

test: $(HIPCC_EXE) $(HOST_EXE)
	$(HIPCC_EXE)
	$(HOST_EXE)

clean:
	rm -f $(HIPCC_EXE)
	rm -f $(HOST_EXE)
	rm -f $(EMIT_STATIC_LIB)
	rm -f *.o

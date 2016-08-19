HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc

HIPCC_FLAGS += -std=c++11
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
ifeq (${HIP_PLATFORM}, nvcc)
	LIBS = -lcublas
endif
ifeq (${HIP_PLATFORM}, hcc)
	HCBLAS_ROOT?= $(wildcard /opt/rocm/hcblas)
	HIPCC_FLAGS += -stdlib=libc++ -I$(HCBLAS_ROOT)/include
	LIBS = -L$(HCBLAS_ROOT)/lib -lhcblas
endif


all: saxpy.hipblas.out

saxpy.cublas.out : saxpy.cublas.cpp
	nvcc -std=c++11 -I$(CUDA_HOME)/include saxpy.cublas.cpp -o $@ -L$(CUDA_HOME)/lib64 -lcublas

# $HIPBLAS_ROOT/bin/hipifyblas ./saxpy.cublas.cpp > ./saxpy.hipblas.cpp
# Then review & finish port in saxpy.hipblas.cpp

saxpy.hipblasref.o: saxpy.hipblasref.cpp
	$(HIPCC) $(HIPCC_FLAGS) -c $< -o $@

saxpy.hipblas.out: saxpy.hipblasref.o
	$(HIPCC) $< -o $@ $(LIBS)

clean:
	rm -f *.o *.out	

HIPCC=/opt/rocm/hip/bin/hipcc
INCLUDE_DIR=../../include -g
BIN_DIR=../../bin/memory

all: hipFree hipHostFree hipHostUnregister hipHostGetDevicePointer hipHostMalloc

hipFree:
	mkdir -p $(BIN_DIR)
	$(HIPCC) -I$(INCLUDE_DIR) -g hipFree.cpp -o $(BIN_DIR)/hipFree

hipHostFree:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipHostFree.cpp -o $(BIN_DIR)/hipHostFree

hipHostUnregister:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipHostUnregister.cpp -o $(BIN_DIR)/hipHostUnregister

hipHostGetDevicePointer:
	$(HIPCC)  -w -I$(INCLUDE_DIR) -g hipHostGetDevicePointer.cpp -o $(BIN_DIR)/hipHostGetDevicePointer

hipHostMalloc:
	$(HIPCC)  -w -I$(INCLUDE_DIR) -g hipHostMalloc.cpp -o $(BIN_DIR)/hipHostMalloc


clean:
	rm -rf $(BIN_DIR)

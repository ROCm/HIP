HIPCC=/opt/rocm/hip/bin/hipcc
INCLUDE_DIR=../../include -g
BIN_DIR=../../bin/stream

all: hipStreamCreate

hipFree:
	mkdir -p $(BIN_DIR)
	$(HIPCC) -I$(INCLUDE_DIR) -g hipStreamCreate.cpp -o $(BIN_DIR)/hipStreamCreate

clean:
	rm -rf $(BIN_DIR)

HIPCC=/opt/rocm/hip/bin/hipcc
INCLUDE_DIR=../../include -g
BIN_DIR=../../bin/device

all: clean hipChooseDevice hipDeviceGetAttribute hipDeviceGetByPCIBusId hipDeviceGetCacheConfig hipDeviceGetLimit hipDeviceGetPCIBusId hipGetDevice hipGetDeviceCount hipGetDeviceProperties hipIpcCloseMemHandle hipIpcGetMemHandle hipSetDevice hipSetDeviceFlags hipIpcOpenMemHandle hipSetValidDevices

#hipIpcOpenEventHandle 

#hipIpcOpenMemHandle

hipChooseDevice:
	mkdir -p $(BIN_DIR)
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipChooseDevice.cpp -o $(BIN_DIR)/hipChooseDevice

hipDeviceGetAttribute:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipDeviceGetAttribute.cpp -o $(BIN_DIR)/hipDeviceGetAttribute

hipDeviceGetByPCIBusId:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipDeviceGetByPCIBusId.cpp -o $(BIN_DIR)/hipDeviceGetByPCIBusId

hipDeviceGetCacheConfig:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipDeviceGetCacheConfig.cpp -o $(BIN_DIR)/hipDeviceGetCacheConfig

hipDeviceGetLimit:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipDeviceGetLimit.cpp -o $(BIN_DIR)/hipDeviceGetLimit

hipDeviceGetPCIBusId:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipDeviceGetPCIBusId.cpp -o $(BIN_DIR)/hipDeviceGetPCIBusId
hipGetDevice:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipGetDevice.cpp -o $(BIN_DIR)/hipGetDevice

hipGetDeviceCount:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipGetDeviceCount.cpp -o $(BIN_DIR)/hipGetDeviceCount

hipGetDeviceProperties:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipGetDeviceProperties.cpp -o $(BIN_DIR)/hipGetDeviceProperties

hipIpcCloseMemHandle:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipIpcCloseMemHandle.cpp -o $(BIN_DIR)/hipIpcCloseMemHandle

hipIpcGetMemHandle:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipIpcGetMemHandle.cpp -o $(BIN_DIR)/hipIpcGetMemHandle

hipSetDevice:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipSetDevice.cpp -o $(BIN_DIR)/hipSetDevice

hipSetDeviceFlags:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipSetDeviceFlags.cpp -o $(BIN_DIR)/hipSetDeviceFlags

hipIpcOpenEventHandle: 
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipIpcOpenEventHandle.cpp -o $(BIN_DIR)/hipIpcOpenEventHandle

hipIpcOpenMemHandle:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipIpcOpenMemHandle.cpp -o $(BIN_DIR)/hipIpcOpenMemHandle

#this func is not found
hipSetValidDevices:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipSetValidDevices.cpp -o $(BIN_DIR)/hipSetValidDevices

#identifier not found
hipGetDeviceFlags:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipGetDeviceFlags.cpp -o $(BIN_DIR)/hipGetDeviceFlags

# This function is not found
hipDeviceGetP2PAttribute:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipDeviceGetP2PAttribute.cpp -o $(BIN_DIR)/hipDeviceGetP2PAttribute

#identifier not found
hipDeviceGetStreamPriorityRange:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipDeviceGetStreamPriorityRange.cpp -o $(BIN_DIR)/hipDeviceGetStreamPriorityRange

#identifier not found
hipDeviceSetLimit:
	$(HIPCC) -w -I$(INCLUDE_DIR) -g hipDeviceSetLimit.cpp -o $(BIN_DIR)/hipDeviceSetLimit

clean:
	rm -rf $(BIN_DIR)

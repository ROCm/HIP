SRC_DEVICE_DIR=./src/device
SRC_MEMORY_DIR=./src/memory
SRC_STREAM_DIR=./src/stream
#SRC_MATHFUNCS_DIR=./src/mathFuncs

all: device memory stream mathFuncs

device:
	make -C $(SRC_DEVICE_DIR)

memory:
	make -C $(SRC_MEMORY_DIR)

stream:
	make -C $(SRC_STREAM_DIR)

#mathFuncs:
#	make -C $(SRC_MATHFUNCS_DIR)

clean:
	rm -rf ./bin

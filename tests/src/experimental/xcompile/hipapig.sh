#!/bin/bash

gcc -D__HIP_PLATFORM_HCC__= -I${HIP_PATH}/include -I${HCC_HOME}/include gApi.c ${HIP_PATH}/lib/device_util.cpp.o ${HIP_PATH}/lib/hip_device.cpp.o ${HIP_PATH}/lib/hip_error.cpp.o ${HIP_PATH}/lib/hip_event.cpp.o ${HIP_PATH}/lib/hip_hcc.cpp.o ${HIP_PATH}/lib/hip_memory.cpp.o ${HIP_PATH}/lib/hip_peer.cpp.o ${HIP_PATH}/lib/hip_stream.cpp.o ${HIP_PATH}/lib/unpinned_copy_engine.cpp.o -L${HCC_HOME}/lib -lhc_am -L${HSA_PATH}/lib -lhsa-runtime64 -lc++ -lmcwamp -ldl -lm -o gApi

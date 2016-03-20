#!/bin/bash

rm $HIP_PATH/src/hip_hcc.o
mkdir build
$HIP_PATH/bin/hipcc hipAPIStreamDisable.cpp test_common.cpp -o ./build/hipAPIStreamDisable
rm $HIP_PATH/src/hip_hcc.o
$HIP_PATH/bin/hipcc hipAPIStreamEnable.cpp test_common.cpp -o ./build/hipAPIStreamEnable
rm $HIP_PATH/src/hip_hcc.o

#!/bin/bash
set -x 

export WORKSPACE=$PWD/hipanl
cd ${WORKSPACE}
               
            
cd ${WORKSPACE}/clr
rm -rf build
mkdir -p build
cd build

cmake -DCLR_BUILD_HIP=ON -DHIP_PLATFORM=nvidia -DHIPCC_BIN_DIR=$HIPCC_DIR/bin -DHIP_COMMON_DIR=$HIP_DIR -DCMAKE_INSTALL_PREFIX=$PWD/install ..


make -j$(nproc)
make install  -j$(nproc)


cd ${WORKSPACE}/hip-tests
export HIP_PATH="${CLR_DIR}"/build/install

rm -rf build
mkdir -p build
cd build
echo "testing $HIP_PATH"


export HIP_PLATFORM=nvidia
cmake -DHIP_PLATFORM=nvidia -DHIP_PATH=$CLR_DIR/build/install ../catch


make -j$(nproc) build_tests

 

cd ${WORKSPACE}/hip-tests
cd build
ctest --overwrite BuildDirectory=. --output-junit hiptest_output_catch_nvidia.xml -E 'Unit_hipMemcpyHtoD_Positive_Synchronization_Behavior|Unit_hipMemcpy_Positive_Synchronization_Behavior|Unit_hipFreeNegativeHost'




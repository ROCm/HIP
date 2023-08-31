
#!/bin/bash

export WORKSPACE=$PWD/hipanl
rm -rf $WORKSPACE
mkdir $WORKSPACE
cd ${WORKSPACE}
               
            
cd ${WORKSPACE}/clr
rm -rf build
mkdir -p build
cd build

cmake -DCLR_BUILD_HIP=ON -DHIP_PATH=$PWD/install -DHIPCC_BIN_DIR=$HIPCC_DIR/bin -DHIP_COMMON_DIR=$HIP_DIR -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install ..


make -j$(nproc)
make install  -j$(nproc)


cd ${WORKSPACE}/hip-tests
export HIP_PATH="${CLR_DIR}"/build/install

rm -rf build
mkdir -p build
cd build
echo "testing $HIP_PATH"

cmake -DHIP_PLATFORM=amd -DHIP_PATH=$CLR_DIR/build/install ../catch 

make -j$(nproc) build_tests

cd ${WORKSPACE}/hip-tests
cd build
ctest --overwrite BuildDirectory=. --output-junit hiptest_output_catch_amd.xml

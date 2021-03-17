#!/bin/bash
# Copyright (C) 2017-2021 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Parse command-line options
# Option strings
SHORT=hr:
LONG=help,rocclr-src:
# read the options
OPTS=$(getopt --options $SHORT --long $LONG --name "$0" -- "$@")
if [ $? != 0 ] ; then echo "Failed to parse options...exiting." >&2 ; exit 1 ; fi

usage() {
    echo "Usage: $0 -r|--roccclr-src <PATH to the rocclr src>" ;
    exit 1;
}

[ $# -eq 0 ] && usage

eval set -- "$OPTS"

# extract options and their arguments into variables.
while true ; do
  case "$1" in
    -r | --rocclr-src )
      ROCCLR_DIR="$2"
      shift 2
      ;;
    -h | --help )
      usage
      shift
      ;;
    -- )
      shift
      break
      ;;
    *)
      echo "Internal error!"
      exit 1
      ;;
  esac
done

BUILD_ROOT="$( mktemp -d )"
SRC_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKING_DIR=$PWD
DASH_JAY="-j $(getconf _NPROCESSORS_ONLN)"

err() {
    echo "${1-Died}." >&2
}

die() {
    err "$1"
    exit 1
}

pushd () {
    command pushd "$@" > /dev/null
}

popd () {
    command popd "$@" > /dev/null
}

function setupENV()
{
    sudo apt-get update
    sudo apt-get install dpkg-dev rpm doxygen libelf-dev rename
}

function buildHIP()
{
    pushd $BUILD_ROOT
    OPENCL_RUNTIME="$BUILD_ROOT/opencl"
    mkdir $OPENCL_RUNTIME
    git clone https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime/ $OPENCL_RUNTIME
    ROCCLR_BUILD_DIR="$BUILD_ROOT/rocclr_build"
    mkdir  $ROCCLR_BUILD_DIR
    pushd $ROCCLR_BUILD_DIR
    cmake $ROCCLR_DIR -DOPENCL_DIR=$OPENCL_RUNTIME -DCMAKE_BUILD_TYPE=Release
    make $DASH_JAY
    popd
    HIP_BUILD_DIR="$BUILD_ROOT/hip_build"
    mkdir $HIP_BUILD_DIR
    pushd $HIP_BUILD_DIR
    cmake $SRC_ROOT -DCMAKE_PREFIX_PATH="$ROCCLR_BUILD_DIR;/opt/rocm" -DCMAKE_BUILD_TYPE=Release
    make $DASH_JAY
    make package
    cp hip-*.deb $WORKING_DIR
    sudo dpkg -i -B hip-base*.deb hip-rocclr*.deb hip-sample*.deb hip-doc*.deb
    popd
    popd
    rm -rf $BUILD_ROOT
}

echo "Preparing build environment"
setupENV || die "setupENV failed"
echo "Building and installing HIP packages"
buildHIP || die "buildHIP failed"
echo "Finished building HIP packages"

#!/bin/bash

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
    cmake $SRC_ROOT -DCMAKE_BUILD_TYPE=Release -DCOMPILE_HIP_ATP_MARKER=1
    make $DASH_JAY
    make package
    rename -v 's/([a-z0-9_.\-]).deb/$1-amd64.deb/' *.deb;rename -v 's/([a-z0-9_.\-]).rpm/$1.x86_64.rpm/' *.rpm
    cp hip_*.deb $WORKING_DIR
    sudo dpkg -i hip_base*.deb hip_hcc*.deb hip_sample*.deb hip_doc*.deb
    popd
    rm -rf $BUILD_ROOT
}

echo "Preparing build environment"
setupENV || die "setupENV failed"
echo "Building and installing HIP packages"
buildHIP || die "buildHIP failed"
echo "Finished building HIP packages"

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
    cmake $SRC_ROOT -DCMAKE_BUILD_TYPE=Release
    make $DASH_JAY
    make package
    cp hip-*.deb $WORKING_DIR
    sudo dpkg -i -B hip-base*.deb hip-hcc*.deb hip-sample*.deb hip-doc*.deb
    popd
    rm -rf $BUILD_ROOT
}

echo "Preparing build environment"
setupENV || die "setupENV failed"
echo "Building and installing HIP packages"
buildHIP || die "buildHIP failed"
echo "Finished building HIP packages"

#!/bin/bash

#usage : hipexamine.sh DIRNAME [hipify options] [--] [clang options]

# Generate CUDA->HIP conversion statistics for all the code files in the specified directory.

SCRIPT_DIR=`dirname $0`
SEARCH_DIR=$1

hipify_args=''
while (( "$#" )); do
  shift
  if [ "$1" != "--" ]; then
    hipify_args="$hipify_args $1"
  else
    shift
    break
  fi
done
clang_args="$@"

$SCRIPT_DIR/hipify-clang -examine $hipify_args `$SCRIPT_DIR/findcode.sh $SEARCH_DIR` -- -x cuda $clang_args

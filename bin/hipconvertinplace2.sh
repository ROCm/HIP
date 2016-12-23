#!/bin/bash

#usage : hipconvertinplace.sh DIRNAME [hipify options] [--] [clang options]

#hipify "inplace" all code files in specified directory.
# This can be quite handy when dealing with an existing CUDA code base since the script
# preserves the existing directory structure.

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

$SCRIPT_DIR/hipify-clang -inplace -print-stats $hipify_args `$SCRIPT_DIR/findcode.sh $SEARCH_DIR` -- -x cuda $clang_args

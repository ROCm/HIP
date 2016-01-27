#!/bin/bash

#usage : hipconvertinplace.sh [DIRNAME] [HIPIFY_OPTIONS]

#hipify "inplace" all code files in specified directory.    
# This can be quite handy when dealing with an existing CUDA code base since the script
# preseeves the existing directory structure.

#  For each code file, this script will:
#   - If ".prehip file does not exist, copy the original code to a new file withextension ".prehip".  Then Hipify the code file.
#   - If ".prehip" file exists, this is used as input to hipify.
# (this is useful for testing improvements to the hipify toolset).


SCRIPT_DIR=`dirname $0`
SEARCH_DIR=$1
shift
$SCRIPT_DIR/hipify -inplace -print-stats "$@" `$SCRIPT_DIR/findcode.sh $SEARCH_DIR`

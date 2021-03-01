#!/bin/bash
# Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#usage : hipconvertinplace-perl.sh DIRNAME [hipify-perl options]

#hipify "inplace" all code files in specified directory.
# This can be quite handy when dealing with an existing CUDA code base since the script
# preserves the existing directory structure.

#  For each code file, this script will:
#   - If ".prehip file does not exist, copy the original code to a new file with extension ".prehip". Then hipify the code file.
#   - If ".prehip" file exists, this is used as input to hipify.
# (this is useful for testing improvements to the hipify-perl toolset).


SCRIPT_DIR=`dirname $0`
SEARCH_DIR=$1
shift
$SCRIPT_DIR/hipify-perl -inplace -print-stats "$@" `$SCRIPT_DIR/findcode.sh $SEARCH_DIR`

#!/usr/bin/env bash

set -o errexit

# Run a single LIT test file in a magical way that preserves colour output, to work around
# a known flaw in lit.

# Capture lit substitutions
HIPIFY=$1
IN_FILE=$2
TMP_FILE=$3
CUDA_ROOT=$4
ROC=$5
shift 5

# Remaining args are the ones to forward to clang proper.

$HIPIFY -o=$TMP_FILE $IN_FILE $CUDA_ROOT $ROC -- $@ && cat $TMP_FILE | sed -Ee 's|//.+|// |g' | FileCheck $IN_FILE

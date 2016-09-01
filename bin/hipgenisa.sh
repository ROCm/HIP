#!/bin/bash

if [ $1 = " " ]
then
exit
fi

ROCM_PATH=$1
GEN_ISA=$2
FILE_NAMES=$3
OUT=$4
OUTPUT_FILE=$5
TARGET=""
if [ ${GEN_ISA:0:12} = "--target-isa" ]
then
  TARGET=${GEN_ISA:13:12}
fi

SOURCE="${BASH_SOURCE[0]}"
HIP_PATH="$( command cd -P "$( dirname "$SOURCE" )/.." && pwd )"

export KMDUMPISA=1
export KMDUMPLLVM=1
mkdir /tmp/hipgenisa
$HIP_PATH/bin/hipcc $FILE_NAMES -o /tmp/hipgenisa/a.out
mv dump.* /tmp/hipgenisa/
$ROCM_PATH/hcc-lc/bin/llvm-mc -arch=amdgcn -mcpu=$TARGET -filetype=obj /tmp/hipgenisa/dump.isa -o /tmp/hipgenisa/dump.o
$ROCM_PATH/llvm/bin/clang -target amdgcn--amdhsa /tmp/hipgenisa/dump.o -o $OUTPUT_FILE
rm -r /tmp/hipgenisa
export KMDUMPISA=0
export KMDUMPLLVM=0

#!/bin/bash

if [ $1 = " " ]
then
exit
fi

ROCM_PATH=$1
GEN_ISA=$3
FILE_NAMES=$4
OUT=$5
OUTPUT_FILE=$6
TARGET=""
if [ ${GEN_ISA:0:12} = "--target-isa" ]
then
  TARGET=${GEN_ISA:13:12}
fi

export KMDUMPISA=1
export KMDUMPLLVM=1
hipcc $FILE_NAMES -o /tmp/hipgenisa/a.out
mv dump.* /tmp/hipgenisa/
${ROCM_PATH}/hcc-lc/bin/llvm-mc -arch=amdgcn -mcpu=$TARGET -filetype=obj /tmp/hipgenisa/dump.isa -o /tmp/hipgenisa/dump.o
${ROCM_PATH}/llvm/bin/clang -target amdgcn--amdhsa /tmp/hipgenisa/dump.o -o $OUTPUT_FILE
export KMDUMPISA=0
export KMDUMPLLVM=0

#!/bin/bash

if [ $1 = " " ]
then
exit
fi

: ${ROCM_PATH:=/opt/rocm}
GEN_ISA=$1
FILE_NAMES=$2
OUT=$3
OUTPUT_FILE=$4
TARGET=""
if [ ${GEN_ISA:0:12} = "--target-isa" ]
then
  TARGET=${GEN_ISA:13:12}
fi

SOURCE="${BASH_SOURCE[0]}"
HIP_PATH="$( command cd -P "$( dirname "$SOURCE" )/.." && pwd )"

export KMDUMPISA=1
export KMDUMPLLVM=1
hipgenisa_dir=`mktemp -d --tmpdir=/tmp hip.XXXXXXXX`;
sed 's/extern \+"C" \+//g' $FILE_NAMES > $FILE_NAMES.kernel.tmp.cpp
echo "
int main(){}
" >> $FILE_NAMES.kernel.tmp.cpp
$HIP_PATH/bin/hipcc $FILE_NAMES.kernel.tmp.cpp -o $hipgenisa_dir/a.out
mv dump.* $hipgenisa_dir
$ROCM_PATH/hcc-lc/bin/llvm-mc -arch=amdgcn -mcpu=$TARGET -filetype=obj $hipgenisa_dir/dump.isa -o $hipgenisa_dir/dump.o
$ROCM_PATH/llvm/bin/clang -target amdgcn--amdhsa $hipgenisa_dir/dump.o -o $hipgenisa_dir/dump.co
map_sym=""
kernels=$(objdump -t $hipgenisa_dir/dump.co | grep grid_launch_parm | sed 's/ \+/ /g; s/\t/ /g' | cut -d" " -f6)
for mangled_sym in $kernels
do
  real_sym=$(c++filt $(c++filt _$mangled_sym | cut -d: -f3 | sed 's/_functor//g') | cut -d\( -f1)
  map_sym="--redefine-sym $mangled_sym=$real_sym $map_sym"
done
objcopy -F elf64-little $map_sym $hipgenisa_dir/dump.co $OUTPUT_FILE
rm $FILE_NAMES.kernel.tmp.cpp
rm -r $hipgenisa_dir

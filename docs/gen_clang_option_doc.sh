#!/bin/bash
# Copyright (C) 2019-2021 Advanced Micro Devices, Inc. All Rights Reserved.
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

## generates documentation about clang options.

clang=/opt/rocm*/llvm/bin/clang

exec > clang_options.md

echo "# Support of Clang options"
echo " Clang version: $($clang --version | head -1|sed 's:\(.*\) (.* \(.*\)).*:\1 \2:')"
echo
echo "|Option|Support|Description|"
echo "|-------|------|-------|"

declare -A db
while read a b; do
  if [[ "$a" != "" && "$b" != "" ]]; then
    db[$a]="$b"
    #echo "db[$a]=${db[$a]}"
  fi
done <clang_options.txt
#for K in "${!db[@]}"; do echo $K; done

tmpf=tmp_clang_option.txt

[[ -f $tmpf ]] && rm $tmpf

$clang --help | sed '1,5d'|  while read a b; do
  if [[ "$a" != "-"* ]]; then
    desc="$a $b"
  elif [[ "$b" = *'>'* ]]; then
    opt=$(echo $a $b| sed -e 's:\(^-[^ ]*[= ]*<[^<>]*>\) *\(.*\):\1:')
    desc=$(echo $a $b| sed -e 's:\(^-[^ ]*[= ]*<[^<>]*>\) *\(.*\):\2:')
    if [[ "$opt" == "$desc" ]]; then
      opt="$a"
      desc="$b"
    fi
  else
    opt="$a"
    desc="$b"
  fi
  supp=
  key=$(printf "%s" "$opt" |sed 's:\([^ =<]*\).*:\1:')
  if [[ "$key" != "" ]]; then
    supp="${db[$key]}"
    #echo "opt=$opt supp=${db[$opt]}"
  fi
  if [[ "$supp" == "" ]]; then
    if [[ "$desc" = *AArch* ||\
          "$desc" = *MIPS* || \
          "$desc" = *ARM* || \
          "$desc" = *Arm* || \
          "$desc" = *SYCL* || \
          "$desc" = *PPC* || \
          "$desc" = *RISC-V* || \
          "$desc" = *WebAssembly* || \
          "$desc" = *Objective-C* || \
          "$opt" = *xray* \
       ]]; then
      supp="n"
    elif [[ "$opt" = *sanity* ]]; then
      supp="h"
    else
      supp="s"
    fi
  fi
  s=$supp
  case $supp in
    s) supp="Supported";;
    n) supp="Unsupported";;
    h) supp="Supported on Host only";;
  esac

  desc=$(echo "$desc"| sed -e 's:^ *::' -e 's:|:\\|:g')
  #echo a=$a
  #echo b=$b
  #echo opt=$opt
  #echo desc=$desc
  if [[ "$desc" != "" ]]; then
    printf "%s %s\n" "$key" "$s" >>$tmpf
    echo '|`'$opt'`|'$supp'|`'$desc'`|'
  fi
done

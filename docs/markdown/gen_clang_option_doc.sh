#!/bin/bash

## generates documentation about clang options.

clang=/opt/rocm/llvm/bin/clang

exec > clang_options.md

echo "# Support of Clang options"
echo " Clang version: $($clang --version | head -1)"
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
  key=$(echo $opt |sed 's:\([^ =<]*\).*:\1:')
  if [[ "$key" != "" ]]; then
    supp="${db[$key]}"
    #echo "opt=$opt supp=${db[$opt]}"
  fi
  if [[ "$supp" == "" ]]; then
    if [[ "$desc" = *AArch* ||\
          "$desc" = *MIPS* || \
	  "$desc" = *ARM* || \
	  "$desc" = *Objective-C* ]]; then
      supp="n"
    elif [[ "$desc" = *OpenCL* ||\
	    "$desc" = *HIP* || \
	    "$desc" = *AMDGPU* ]]; then
      supp="s"
    fi
  fi
  case $supp in
    s) supp="Supported";;
    n) supp="Unsupported";;
    h) supp="Supported on Host only";;
  esac

  desc=$(echo "$desc"| sed -e 's:|:\\|:g')
  #echo a=$a
  #echo b=$b
  #echo opt=$opt
  #echo desc=$desc
  if [[ "$desc" != "" ]]; then
    echo $key >>$tmpf
    echo '|`'$opt'`|'$supp'|`'$desc'`|'
  fi
done

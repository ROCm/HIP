#!/usr/bin/env bash

set -euo pipefail

RANGE=""

while [[ $# -gt 0 ]]; do
  echo $1
  echo $2
  case "$1" in
  --range)
    RANGE="$2"
    shift 2
    ;;
  *)
    echo "Unknown arg $1" >&2
    exit 64
    ;;
  esac
done

regex='\.(c|cc|cpp|cxx|h|hh|hpp|hxx)$'

if [[ -n $RANGE ]]; then
  files=$(git diff --name-only "$RANGE" | grep -E "$regex" || true)
else
  files=$(git diff --cached --name-only --diff-filter=ACMR | grep -E "$regex" || true)
fi
echo "Checking $files"
[[ -z $files ]] && exit 0

clang_bin="${CLANG_FORMAT:-clang-format}"
if ! command -v "$clang_bin" >/dev/null 2>&1; then
  if [[ -x "/c/Program Files/LLVM/bin/clang-format.exe" ]]; then
    clang_bin="/c/Program Files/LLVM/bin/clang-format.exe"
  fi
fi

clang_format_diff="${CLANG_FORMAT_DIFF:-clang-format-diff}"
if ! command -v "$clang_format_diff" >/dev/null 2>&1; then
  if [[ -x "/c/Program Files/LLVM/share/clang/clang-format-diff.py" ]]; then
    clang_format_diff="/c/Program Files/LLVM/share/clang/clang-format-diff.py"
  fi
fi

for file in $files; do
  echo "Checking lines of $file"

  if [[ -n $RANGE ]]; then
    diff_output=$(git diff -U0 "$RANGE" -- "$file")
  else
    diff_output=$(git diff -U0 --cached -- "$file")
  fi

  echo "$diff_output" | "$clang_format_diff" -style=file -fallback-style=none -p1
done

#!/usr/bin/env bash

set -o errexit

# Run a single LIT test file in a magical way that preserves colour output, to work around
# a known flaw in lit.

# Capture lit substitutions
HIPIFY=$1
IN_FILE=$2
TMP_FILE=$3
shift 3

# Remaining args are the ones to forward to clang proper.

# Time for the classic insane little trick for making colour output work.
# A self-deleting shell-script that does the thing we want to do...
TMP_SCRIPT=$(mktemp)
cat << EOF > $TMP_SCRIPT
set -o errexit
set -o xtrace
rm $TMP_SCRIPT
$HIPIFY -o=$TMP_FILE $IN_FILE -- $@ && cat $TMP_FILE | sed -Ee 's|//.+|// |g' | FileCheck $IN_FILE
EOF
chmod a+x $TMP_SCRIPT

# Run the script via socat, spawning a virtual terminal and propagating exit code, and hence failure.
socat -du EXEC:$TMP_SCRIPT,pty,stderr STDOUT

#!/bin/bash

set -exuo pipefail

script_dir=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

mkdir -p build/docs
export HIP_DOXYGEN_OUTDIR="$script_dir/build/docs/doxygen"
doxygen docs/Doxyfile
sphinx-build docs build/docs/sphinx

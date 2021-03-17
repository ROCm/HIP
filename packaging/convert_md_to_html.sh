#!/bin/bash
# Copyright (C) 2016-2021 Advanced Micro Devices, Inc. All Rights Reserved.
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

function die {
    echo "${1-Died}." >&2
    exit 1
}

function cleanup {
    rm -rf "$workdir"
}

# parse arguments
hip_srcdir=$1
html_destdir=$2
[ "$hip_srcdir" != "" ] || [ "$html_destdir" != "" ] || die "Invalid arguments!"

# create temporary directory for grip settings
workdir=`mktemp -d`
trap cleanup EXIT

# setup grip
export GRIPURL=$hip_srcdir
export GRIPHOME=$workdir
echo "CACHE_DIRECTORY = '$html_destdir/asset'" > $workdir/settings.py
mkdir -p $html_destdir $html_destdir/docs/markdown

# convert all md files to html
pushd $hip_srcdir
for f in *.md docs/markdown/*.md; do grip --export --no-inline $f $html_destdir/${f%.*}.html; done
popd

# convert absolute links to relative links
pushd $html_destdir
for f in *.html; do sed -i "s?$GRIPURL/??g" $f; done
for f in docs/markdown/*.html; do sed -i "s?$GRIPURL/?../../?g" $f; done
popd

# update document titles
pushd $html_destdir
for f in *.html; do sed -i "s?.md - Grip??g" $f; done
for f in docs/markdown/*.html; do sed -i "s?.md - Grip??g" $f; done
popd

# replace .md with .html in links
pushd $html_destdir
for f in *.html; do sed -i "s?.md\"?.html\"?g" $f; done
for f in *.html; do sed -i "s?.md#?.html#?g" $f; done
for f in docs/markdown/*.html; do sed -i "s?.md\"?.html\"?g" $f; done
for f in docs/markdown/*.html; do sed -i "s?.md#?.html#?g" $f; done
popd

# replace github.io links
pushd $html_destdir
sed -i "s?http://rocm-developer-tools.github.io/HIP?docs/RuntimeAPI/html/index.html?g" README.html
sed -i "s?http://rocm-developer-tools.github.io/HIP?docs/RuntimeAPI/html/?g" RELEASE.html
popd

exit 0

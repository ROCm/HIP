#!/bin/bash
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
mkdir -p $html_destdir $html_destdir/hipify-clang $html_destdir/docs/markdown

# convert all md files to html
pushd $hip_srcdir
for f in *.md hipify-clang/*.md docs/markdown/*.md; do grip --export --no-inline $f $html_destdir/${f%.*}.html; done
popd

# convert absolute links to relative links
pushd $html_destdir
for f in *.html; do sed -i "s?$GRIPURL/??g" $f; done
for f in hipify-clang/*.html; do sed -i "s?$GRIPURL/?../?g" $f; done
for f in docs/markdown/*.html; do sed -i "s?$GRIPURL/?../../?g" $f; done
popd

# update document titles
pushd $html_destdir
for f in *.html; do sed -i "s?.md - Grip??g" $f; done
for f in hipify-clang/*.html; do sed -i "s?.md - Grip??g" $f; done
for f in docs/markdown/*.html; do sed -i "s?.md - Grip??g" $f; done
popd

# replace .md with .html in links
pushd $html_destdir
for f in *.html; do sed -i "s?.md\"?.html\"?g" $f; done
for f in *.html; do sed -i "s?.md#?.html#?g" $f; done
for f in hipify-clang/*.html; do sed -i "s?.md\"?.html\"?g" $f; done
for f in hipify-clang/*.html; do sed -i "s?.md#?.html#?g" $f; done
for f in docs/markdown/*.html; do sed -i "s?.md\"?.html\"?g" $f; done
for f in docs/markdown/*.html; do sed -i "s?.md#?.html#?g" $f; done
popd

# replace github.io links
pushd $html_destdir
sed -i "s?http://rocm-developer-tools.github.io/HIP?docs/RuntimeAPI/html/index.html?g" README.html
sed -i "s?http://rocm-developer-tools.github.io/HIP?docs/RuntimeAPI/html/?g" RELEASE.html
popd

exit 0

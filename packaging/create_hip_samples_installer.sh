#!/bin/bash
function die {
    echo "${1-Died}." >&2
    exit 1
}

payload=$1
script=$2
[ "$payload" != "" ] || [ "$script" != "" ] || die "Invalid arguments!"
tmp=__extract__$RANDOM

printf "#!/bin/bash
samples_dir=\$1
[ \"\$samples_dir\" != \"\" ] || read -e -p \"Enter the path to extract the HIP samples: \" samples_dir
mkdir -p \$samples_dir
PAYLOAD=\`awk '/^__PAYLOAD_BELOW__/ {print NR + 1; exit 0; }' \$0\`
tail -n+\$PAYLOAD \$0 | tar -xz -C \$samples_dir
echo \"HIP samples installed in \$samples_dir\"
exit 0
__PAYLOAD_BELOW__\n" > "$tmp"

cat "$tmp" "$payload" > "$script" && rm "$tmp"
chmod +x "$script"

#!/bin/bash
# fetch_sources.sh -- put the upstream sources of the measured build next to this Dockerfile, at the exact pins,
# then check every file against SOURCES.gitblob (git blob sha1 of the bytes that went into the measured image).
set -euo pipefail
cd "$(dirname "$0")"
TC=cd83ade14c43a42babad39895d75bdd0e3453d49   # CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE (Apache-2.0)
EX=d3739fd393337b1ff4d6c2a342b12f0c87a9592f   # turboderp-org/exllamav3 v1.5.3 (MIT)
tmp=$(mktemp -d)
git clone -q https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE "$tmp/tc" && git -C "$tmp/tc" checkout -q $TC
git clone -q --filter=blob:none https://github.com/turboderp-org/exllamav3 "$tmp/ex" && git -C "$tmp/ex" checkout -q $EX
rm -rf tc_moe exllamav3_ext
cp -r "$tmp/tc/vllm_exl3/tc_moe" tc_moe
cp "$tmp/tc/exl3_tc_hook.py" "$tmp/tc/gate0.py" .
cp "$tmp/tc/LICENSE" LICENSE.curtis; cp "$tmp/tc/NOTICE.md" NOTICE.curtis.md
mkdir exllamav3_ext
while read -r sha rel; do
  case "$rel" in exllamav3_ext/LICENSE.exllamav3) cp "$tmp/ex/LICENSE" "$rel" ;;
    exllamav3_ext/*) mkdir -p "$(dirname "$rel")"; cp "$tmp/ex/exllamav3/$rel" "$rel" ;; esac
done < SOURCES.gitblob
bad=0
while read -r sha rel; do
  [ "$(git hash-object "$rel")" = "$sha" ] || { echo "MISMATCH $rel"; bad=1; }
done < SOURCES.gitblob
rm -rf "$tmp"
[ $bad = 0 ] && echo "sources OK ($(wc -l < SOURCES.gitblob) files)"
exit $bad

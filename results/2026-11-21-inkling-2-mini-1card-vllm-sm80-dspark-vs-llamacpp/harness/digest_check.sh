#!/bin/bash
# digest_check.sh <repo> <sha256:...> — anonymous Docker Hub manifest lookup, prints HTTP code
t=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$1:pull" | python3 -c "import json,sys;print(json.load(sys.stdin)['token'])")
curl -s -o /dev/null -w "$1@$2 %{http_code}\n" -H "Authorization: Bearer $t" \
  -H 'Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json' \
  "https://registry-1.docker.io/v2/$1/manifests/$2"

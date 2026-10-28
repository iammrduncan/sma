#!/bin/sh
# Pull the task images in images.txt one at a time, by digest, and tag each as <repo>:latest, the
# name mini-swe-agent and the harness look up (README.md). Docker Hub limits anonymous pulls, so
# it waits between pulls and retries. It stops while Docker's disk has less than 30 GB free.
cd "$(dirname "$0")" || exit 1
root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null) || root=/
n=0; ok=0; failed=0; total=$(wc -l < images.txt)
while read -r ref; do
  n=$((n+1)); name="${ref%@*}"
  free_gb=$(df -BG --output=avail "$root" | tail -1 | tr -dc 0-9)
  if [ "$free_gb" -lt 30 ]; then echo "$(date -u +%FT%TZ) STOP: only ${free_gb} GB free under $root"; break; fi
  if docker image inspect "$ref" >/dev/null 2>&1; then
    docker tag "$ref" "$name" && echo "$(date -u +%FT%TZ) [$n/$total] have $ref"; ok=$((ok+1)); continue
  fi
  for try in 1 2 3; do
    if docker pull -q "$ref" >/dev/null 2>pull.err && docker tag "$ref" "$name"; then
      echo "$(date -u +%FT%TZ) [$n/$total] pulled $ref"; ok=$((ok+1)); break
    fi
    echo "$(date -u +%FT%TZ) [$n/$total] try $try failed: $(tail -c 300 pull.err)"; sleep $((60*try))
    [ "$try" = 3 ] && failed=$((failed+1))
  done
  sleep 30
done < images.txt
echo "$(date -u +%FT%TZ) DONE: $ok of $total present, $failed failed"; df -h "$root" | tail -1
[ "$ok" = "$total" ]

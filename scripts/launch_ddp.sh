#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  -m fastslow.train \
  "$@"

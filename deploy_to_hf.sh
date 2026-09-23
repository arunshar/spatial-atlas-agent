#!/bin/bash
# Deploy Spatial Atlas to HuggingFace Spaces
#
# Usage: HF_SPACE_REPO=https://huggingface.co/spaces/<user>/<space> ./deploy_to_hf.sh
#
# Prerequisites:
#   1. Install HF CLI: pip install huggingface_hub
#   2. Login: huggingface-cli login  (use a write token from https://huggingface.co/settings/tokens)
#   3. Before running this script, configure OPENAI_API_KEY and ATLAS_BEARER_TOKEN
#      as Space secrets. The bearer token must be a unique random value of at least
#      32 characters. Never place either real value in this script or a tracked file.
#   4. Commit your changes first. This script deploys the committed HEAD of this git
#      repository, so uncommitted edits and untracked local files never reach the Space.

set -euo pipefail

SPACE_REPO="${HF_SPACE_REPO:?Set HF_SPACE_REPO to https://huggingface.co/spaces/<user>/<space>}"
# A URL with embedded credentials would be printed below and saved in the clone's .git/config.
case "$SPACE_REPO" in
  *@*)
    echo "HF_SPACE_REPO must not contain credentials. Use huggingface-cli login." >&2
    exit 1
    ;;
  https://huggingface.co/spaces/*/*) ;;
  *)
    echo "HF_SPACE_REPO must be https://huggingface.co/spaces/<user>/<space>" >&2
    exit 1
    ;;
esac
SPACE_ID="${SPACE_REPO#https://huggingface.co/spaces/}"

REPO_DIR="$(cd "$(dirname "$0")" && pwd -P)"
if [ "$(git -C "$REPO_DIR" rev-parse --show-toplevel 2>/dev/null)" != "$REPO_DIR" ]; then
  echo "REFUSING TO DEPLOY: this script must sit at the top of a git repository." >&2
  exit 1
fi
if [ -n "$(git -C "$REPO_DIR" status --porcelain)" ]; then
  echo "Note: uncommitted changes are not deployed. Only the committed HEAD is copied."
fi

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

echo "=== Deploying Spatial Atlas to HuggingFace Spaces ==="
echo "Cloning Space repo..."
git clone "$SPACE_REPO" "$WORK_DIR/space"

echo "Copying the committed project files..."
# git archive exports only the files committed at HEAD, so local-only files such as .env,
# key material, logs, and session notes cannot reach the Space. The Dockerfile rebuilds from
# pyproject.toml and uv.lock, so no virtual environment is copied. The excluded paths are
# not needed by the Space. eval_bench.py holds a function under NVIDIA's non-commercial
# license, and assets/ holds figures with all rights reserved, so neither ships under the
# Space's MIT card.
git -C "$REPO_DIR" archive --format=tar HEAD -- . \
  ':(exclude)tests' \
  ':(exclude)scenarios' \
  ':(exclude)paper' \
  ':(exclude)poster' \
  ':(exclude).github' \
  ':(exclude)eval_bench.py' \
  ':(exclude)assets' \
  | tar -x -C "$WORK_DIR/space"

# Fail closed if anything sensitive is in the Space checkout. A silent leak to a public repo
# is unrecoverable, so this refuses to push rather than warn. The check also covers files
# that an earlier deploy may have left in the Space.
for forbidden in sessions .venv-test .hypothesis junit.xml .env .claude; do
  if [ -e "$WORK_DIR/space/$forbidden" ]; then
    echo "REFUSING TO DEPLOY: '$forbidden' is in the Space checkout." >&2
    exit 1
  fi
done
leak=$(find "$WORK_DIR/space" -path "$WORK_DIR/space/.git" -prune -o \( \
  -name '*.pem' -o -name '*.key' -o -name '*.p12' -o -name 'id_rsa*' -o -name '.netrc' \
  -o -name 'secrets.*' -o -name 'credentials.*' \
  -o -name '*.tfstate*' -o -name '*.tfvars' -o -name '*.tfplan' \
  -o -name '*.sbatch' -o -name '*HANDOFF*' -o -name 'qspatial_pilot_common.sh' \
  -o -name 'build_p5_assets.py' -o -name 'build_p3_table.py' -o -name '*.log' \
  -o -name 'sessions' -o -name '.claude' -o -name '.env' -o -name '.env.*' \
  -o -name 'eval_bench.py' -o -name 'assets' \
  \) -print -quit)
if [ -n "$leak" ]; then
  echo "REFUSING TO DEPLOY: a local-only or excluded file is in the Space checkout:" >&2
  echo "  ${leak#"$WORK_DIR/space/"}" >&2
  echo "Remove it from the Space. If the Space history must not hold it, recreate the Space." >&2
  exit 1
fi

# Create the Space-specific README (overwrites the project README)
cat > "$WORK_DIR/space/README.md" << 'SPACE_README'
---
title: Spatial Atlas
license: mit
sdk: docker
app_port: 9019
colorFrom: purple
colorTo: indigo
pinned: false
---

# Spatial Atlas

Spatial Atlas is a spatial-aware research agent built on compute-grounded reasoning (CGR).

It routes FieldWork spatial questions and opt-in MLE tasks through one A2A server. This repository reports no benchmark result.

The optional metric path calls NVIDIA's SpatialClaw, which NVIDIA licenses separately for non-commercial research. See NOTICE.

The MIT License covers this Space's own code. NOTICE names the third-party portions that keep other terms.

Before non-loopback startup, configure `OPENAI_API_KEY` and
`ATLAS_BEARER_TOKEN` as Space secrets. The bearer token must be a unique random
value of at least 32 characters. Clients must send
`Authorization: Bearer <ATLAS_BEARER_TOKEN>` on non-read-only requests. This
shared token is suitable only for bounded private staging, not public
multi-user identity.

Source: [github.com/arunshar/spatial-atlas-agent](https://github.com/arunshar/spatial-atlas-agent)
SPACE_README

echo "Pushing to HuggingFace Space..."
cd "$WORK_DIR/space"
git add -A
git commit -m "Deploy spatial-atlas agent"
git push

echo ""
echo "=== Done! ==="
echo "Space URL: $SPACE_REPO"
echo "Agent Card: https://$(echo "$SPACE_ID" | tr '/' '-').hf.space/.well-known/agent-card.json"
echo ""
echo "Required Space secrets, which must be configured before running this script:"
echo "  Settings > Repository secrets > New secret > OPENAI_API_KEY"
echo "  Settings > Repository secrets > New secret > ATLAS_BEARER_TOKEN"
echo "Use a unique random ATLAS_BEARER_TOKEN of at least 32 characters."
echo "The shared bearer token is for bounded private staging, not public multi-user identity."

#!/usr/bin/env bash
# Downloads the frozen DINOv2 ViT-S/14 that notebook 03 uses: the official code at a fixed commit and
# the pretrained weights, into checkpoints/ (not stored in git). Checks the weights' SHA-256.
# Usage (from the repository folder):   bash learned/setup_dinov2.sh
set -euo pipefail

COMMIT="7764ea0f912e53c92e82eb78a2a1631e92725fc8"   # facebookresearch/dinov2, Apache-2.0
CODE_URL="https://github.com/facebookresearch/dinov2/archive/${COMMIT}.zip"
WEIGHTS_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
WEIGHTS_SHA256="b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="$HERE/checkpoints/dinov2"
WEIGHTS="$HERE/checkpoints/torch_hub/hub/checkpoints/dinov2_vits14_pretrain.pth"

echo "== 1/2  DINOv2 code (commit ${COMMIT:0:7}) =="
if [ -f "$CODE_DIR/dinov2-$COMMIT/hubconf.py" ]; then
  echo "already present, skipping"
else
  mkdir -p "$CODE_DIR"
  curl -L --fail -o "$CODE_DIR/dinov2-$COMMIT.zip" "$CODE_URL"
  (cd "$CODE_DIR" && unzip -q -o "dinov2-$COMMIT.zip")
fi

echo "== 2/2  DINOv2 ViT-S/14 weights (about 85 MB) =="
mkdir -p "$(dirname "$WEIGHTS")"
if [ -s "$WEIGHTS" ]; then
  echo "already present, skipping"
else
  curl -L --fail -o "$WEIGHTS" "$WEIGHTS_URL"
fi
if command -v sha256sum >/dev/null 2>&1; then
  GOT="$(sha256sum "$WEIGHTS" | cut -d' ' -f1)"
else
  GOT="$(shasum -a 256 "$WEIGHTS" | cut -d' ' -f1)"
fi
if [ "$GOT" != "$WEIGHTS_SHA256" ]; then
  echo "The weights' SHA-256 is $GOT, expected $WEIGHTS_SHA256. Delete $WEIGHTS and run again." >&2
  exit 1
fi
echo "Done. The weights match the recorded SHA-256."

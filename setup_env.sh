#!/usr/bin/env bash
# One-command setup: creates the conda environment, installs SAM 2.1,
# downloads the SAM 2.1-large model weights and registers the Jupyter kernel.
# Usage (from the repository folder):   bash setup_env.sh
set -euo pipefail

ENV_NAME="extruder-sam2"
# Fixed SAM 2 version, so everyone runs exactly the same code
SAM2_COMMIT="2b90b9f5ceec907a1c18123530e92e794ad901a4"
CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda or Anaconda first, then re-run this script." >&2
  exit 1
fi

echo "== 1/4  Creating (or updating) the conda environment '$ENV_NAME' =="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda env update -n "$ENV_NAME" -f "$HERE/environment.yml" --prune
else
  conda env create -f "$HERE/environment.yml"
fi

echo "== 2/4  Installing SAM 2.1 (commit ${SAM2_COMMIT:0:7}) =="
# SAM2_BUILD_CUDA=0 skips an optional compiled extension; SAM 2 works the same without it.
SAM2_BUILD_CUDA=0 conda run -n "$ENV_NAME" --no-capture-output \
  pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@${SAM2_COMMIT}"

echo "== 3/4  Downloading the SAM 2.1-large weights (about 900 MB) =="
mkdir -p "$HERE/checkpoints"
if [ ! -s "$HERE/checkpoints/sam2.1_hiera_large.pt" ]; then
  curl -L --fail -o "$HERE/checkpoints/sam2.1_hiera_large.pt" "$CKPT_URL"
else
  echo "already present, skipping"
fi

echo "== 4/4  Registering the Jupyter kernel and installing this project's helper code =="
conda run -n "$ENV_NAME" --no-capture-output pip install -e "$HERE"
conda run -n "$ENV_NAME" --no-capture-output python -m ipykernel install --user \
  --name "$ENV_NAME" --display-name "Python (extruder-sam2)"

echo
echo "Checking the installation ..."
conda run -n "$ENV_NAME" --no-capture-output python - <<'EOF'
import torch, sam2, cv2, extruder
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU found")
print("sam2 imported from", sam2.__file__)
EOF
echo
echo "Done. Open notebooks/01_cut_frame_pipeline.ipynb in VS Code and pick the kernel 'Python (extruder-sam2)'."

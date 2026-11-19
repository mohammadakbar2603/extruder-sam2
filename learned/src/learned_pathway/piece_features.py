"""Piece pictures and their DINOv2 descriptions, for DL-1 and DL-2 (notebook 03).

Each mask is redrawn as a 224 x 224 picture at exactly 4 px per mm with its clip's ruler, centred on
the piece (`dl1.crop_affine`, `dl1.make_crops`): an **outline** (white piece on black) or a
**colour** crop (the video's pixels inside the piece, black elsewhere). Every picture is also
turned by 90, 180 and 270 degrees and mirrored: 8 **views**, so DL-2 can learn that a piece's
direction in the picture does not change its size. Each view goes through the frozen, pretrained
DINOv2 ViT-S/14 and becomes 768 numbers (CLS token + mean patch token, `dl1_embed.embed`).

Nothing here reads the lab values. The result is cached on disk, keyed by a hash of every input
(mask files, rulers, crop version, encoder), so a change in any of them rebuilds it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from learned_pathway import dl1, nb02

N_VIEWS = 8
DIM = 768
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_DIR = nb02.ROOT / "checkpoints/dinov2" / f"dinov2-{DINOV2_COMMIT}"  # learned/setup_dinov2.sh
TORCH_HOME = nb02.ROOT / "checkpoints/torch_hub"
WEIGHTS = TORCH_HOME / "hub/checkpoints/dinov2_vits14_pretrain.pth"
WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
VERSIONS = ("outline", "colour")


def dihedral(img: np.ndarray) -> np.ndarray:
    """The 8 views of a square picture: turned 0/90/180/270 degrees, then the same mirrored."""
    turns = [np.rot90(img, k) for k in range(4)]
    flipped = np.fliplr(img)
    return np.stack(turns + [np.rot90(flipped, k) for k in range(4)])


def colour_frames(video: Path, frames: Sequence[int], origin: tuple[int, int],
                  size: int = 360) -> dict[int, np.ndarray]:
    """RGB crops (size x size) at `origin` (x, y) of the wanted frames, read once in order with extruder's
    own reader (`extruder.video.iter_crops`), i.e. the same pixels SAM 2 saw (brightness 1)."""
    from extruder import video as vid

    want, out = set(int(f) for f in frames), {}
    last = max(want) if want else -1
    for i, fr in enumerate(vid.iter_crops(str(video), int(origin[0]), int(origin[1]), size)):
        if i in want:
            out[i] = fr.copy()
        if i >= last:
            break
    missing = sorted(want - set(out))
    if missing:
        raise ValueError(f"{Path(video).name}: frames {missing[:5]} not in the video")
    return out


def _crop(mask: np.ndarray, px_to_mm: np.ndarray, frame_rgb: np.ndarray | None) -> tuple[np.ndarray, bool]:
    piece = dl1.largest_component(mask)  # as extruder's measure keeps it: loose specks dropped
    outline, colour, touches = dl1.make_crops(piece, dl1.crop_affine(piece, px_to_mm), frame_rgb)
    return (outline if frame_rgb is None else colour), touches


def piece_crop(mask: np.ndarray, px_to_mm: np.ndarray, frame_rgb: np.ndarray | None) -> np.ndarray:
    """The 224 x 224 outline (uint8) or, with `frame_rgb`, colour crop (uint8, 3 channels)."""
    return _crop(mask, px_to_mm, frame_rgb)[0]


def _file_key(path: Path) -> str:
    st = Path(path).stat()
    return f"{path}|{st.st_size}|{st.st_mtime_ns}"


def inputs_hash(pieces: pd.DataFrame, rulers: Mapping[str, np.ndarray], version: str, encoder_id: str,
                videos: Mapping[str, Path] | None = None) -> str:
    """SHA-256 over every input of the features: mask files (path, size, time), frames, rulers,
    crop version, encoder and, for colour, the video files."""
    h = hashlib.sha256()
    for r in pieces[["clip", "frame", "mask_path"]].itertuples(index=False):
        h.update(f"{r.clip}|{int(r.frame)}|{_file_key(Path(r.mask_path))}\n".encode())
    for clip in sorted(rulers):
        h.update(f"{clip}|{np.round(np.asarray(rulers[clip], float), 9).tolist()}\n".encode())
    for clip in sorted(videos or {}):
        h.update(f"video|{_file_key(Path(videos[clip]))}\n".encode())
    h.update(f"{version}|{encoder_id}|{dl1.CROP_PX}|{dl1.CROP_PX_PER_MM}|{N_VIEWS}".encode())
    return h.hexdigest()


def _pick_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dinov2_encoder(device: str = "auto") -> Callable:
    """The frozen DINOv2 ViT-S/14 as a function (B, 3, 224, 224) tensor -> (B, 768) tensor, after
    checking the weights' SHA-256 against the recorded one."""
    from learned_pathway import dl1_embed

    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise RuntimeError(f"DINOv2 weights changed: SHA-256 {digest}, recorded {WEIGHTS_SHA256}")
    dev = _pick_device(device)
    model = dl1_embed.load_dinov2(DINOV2_DIR, TORCH_HOME).to(dev)

    def encode(x):
        return dl1_embed.embed(model, x.to(dev))

    return encode


def build_features(pieces: pd.DataFrame, results_dir: Path, version: str, cache_dir: Path, *,
                   videos_dir: Path | None = None, device: str = "auto", batch: int = 64,
                   encoder: Callable | None = None, log: Callable = print) -> tuple[pd.DataFrame, np.ndarray]:
    """(index, features): one index row per piece, in the order of `pieces`, with `touches_border`;
    features float32 (n, 8, 768). Reused from `cache_dir` when every input is unchanged."""
    import torch

    from learned_pathway import dl1_embed

    if version not in VERSIONS:
        raise ValueError(f"unknown crop version {version!r}; expected one of {VERSIONS}")
    if version == "colour" and videos_dir is None:
        raise ValueError("colour crops need videos_dir")
    pieces = pieces.reset_index(drop=True)
    clips = list(dict.fromkeys(pieces["clip"]))
    rulers = {c: nb02.clip_ruler(results_dir, c) for c in clips}
    videos = {c: Path(videos_dir) / f"{c}.MP4" for c in clips} if version == "colour" else None
    key = inputs_hash(pieces, rulers, version, "custom" if encoder is not None else f"dinov2-{DINOV2_COMMIT}",
                      videos)
    cache_dir = Path(cache_dir)
    meta_path, npz_path, idx_path = (cache_dir / f"{version}.json", cache_dir / f"{version}.npz",
                                     cache_dir / f"{version}_index.csv")
    if meta_path.exists() and json.loads(meta_path.read_text()).get("hash") == key and npz_path.exists():
        log(f"{version}: features re-used from {npz_path}")
        return pd.read_csv(idx_path, dtype={"clip": str, "setting": str}), np.load(npz_path)["features"]

    encode = encoder if encoder is not None else dinov2_encoder(device)
    feats = np.zeros((len(pieces), N_VIEWS, DIM), np.float32)
    touches = np.zeros(len(pieces), bool)
    for clip in clips:
        rows = pieces.index[pieces["clip"].eq(clip)].to_numpy()
        g = pieces.loc[rows]
        frames = None
        if version == "colour":
            frames = colour_frames(videos[clip], g["frame"].astype(int).tolist(),
                                   (int(g["origin_x"].iloc[0]), int(g["origin_y"].iloc[0])))
        for start in range(0, len(rows), batch):
            part = rows[start:start + batch]
            crops = []
            for i in part:
                r = pieces.loc[i]
                mask = np.asarray(Image.open(r["mask_path"])) > 0
                img, touches[i] = _crop(mask, rulers[clip], None if frames is None else frames[int(r["frame"])])
                crops.append(dihedral(img))
            views = np.concatenate(crops)  # (b * 8, 224, 224[, 3])
            with torch.no_grad():
                out = encode(torch.from_numpy(dl1_embed.to_model_input(views)))
            feats[part] = out.detach().float().cpu().numpy().reshape(len(part), N_VIEWS, DIM)
        log(f"{version}: {clip} encoded ({len(rows)} pieces)")
    index = pieces.drop(columns=[c for c in ("touches_border",) if c in pieces]).assign(touches_border=touches)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, features=feats)
    index.to_csv(idx_path, index=False)
    meta_path.write_text(json.dumps({"hash": key, "n": len(pieces), "version": version,
                                     "dinov2_commit": DINOV2_COMMIT, "weights_sha256": WEIGHTS_SHA256}))
    return index, feats


def bags(pset: pd.DataFrame, features: np.ndarray) -> dict[str, np.ndarray]:
    """Each clip's pieces' features (m, 8, 768), clips in order of appearance; `pset` from
    `nb02.piece_set` (its `row` column points into `features`)."""
    return {c: features[g["row"].to_numpy()] for c, g in pset.groupby("clip", sort=False)}


def crops_of(rows: pd.DataFrame, results_dir: Path, version: str, videos_dir: Path | None = None) -> list[np.ndarray]:
    """The crops of a few pieces, for figures (view 0, as encoded)."""
    out = []
    for _, r in rows.iterrows():
        mask = np.asarray(Image.open(r["mask_path"])) > 0
        frame = None
        if version == "colour":
            frame = colour_frames(Path(videos_dir) / f"{r['clip']}.MP4", [int(r["frame"])],
                                  (int(r["origin_x"]), int(r["origin_y"])))[int(r["frame"])]
        out.append(piece_crop(mask, nb02.clip_ruler(results_dir, r["clip"]), frame))
    return out

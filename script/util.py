"""Shared helpers of the MIASOD pipeline: tile geometry, ranking, top-K, Soft-NMS, timing, configs and MOT-format I/O."""
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "script"

TILE = 256      # scoring tile size (px)
STRIDE = 192    # scoring tile stride (px); neighbouring tiles overlap by at least TILE - STRIDE


# ---------------------------------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------------------------------

def tile_origins(width, height, tile=TILE, stride=STRIDE):
    """Origins (x, y) of the scoring tiles in row-major order.

    Tiles start at 0 and step by `stride`; the last tile of each row and column is shifted to end at the frame border,
    which changes the final overlap but never the tile size.
    """
    xs = sorted(set(range(0, max(1, width - tile + 1), stride)) | {width - tile})
    ys = sorted(set(range(0, max(1, height - tile + 1), stride)) | {height - tile})
    xs = [max(0, x) for x in xs]
    ys = [max(0, y) for y in ys]
    return [(x, y) for y in ys for x in xs]


def xyxy_to_xywh(b):
    b = np.asarray(b, np.float32).reshape(-1, 4)
    return np.c_[b[:, :2], b[:, 2:] - b[:, :2]]


def centres(boxes_xywh):
    b = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
    return b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3] / 2.0


def tile_label(boxes_xywh, tx, ty, tile=TILE, min_px=2.0):
    """1 if a box centre lies in the tile and at least `min_px` of that box falls inside the tile in each axis."""
    for x, y, w, h in boxes_xywh:
        cx, cy = x + w / 2.0, y + h / 2.0
        if tx <= cx < tx + tile and ty <= cy < ty + tile:
            if min(x + w, tx + tile) - max(x, tx) >= min_px and min(y + h, ty + tile) - max(y, ty) >= min_px:
                return 1
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# Ranking and selection
# ---------------------------------------------------------------------------------------------------------------------

def rank(scores):
    """Ascending within-frame rank / N in (0, 1]; tied scores share their mean rank."""
    s = np.asarray(scores, np.float64).reshape(-1)
    return rankdata(s) / len(s)


def n_keep(keep, n):
    """n_K = max(1, round(K n)), the number of regions retained at keep fraction K."""
    if not 0.0 < keep <= 1.0:
        raise ValueError(f"keep fraction must be in (0, 1], got {keep}")
    return max(1, int(round(keep * n)))


def topk_mask(scores, keep):
    """Boolean mask of the n_K highest scores; ties at the cut are trimmed in index order."""
    s = np.asarray(scores, np.float64).reshape(-1)
    n = len(s)
    k = n_keep(keep, n)
    thr = np.partition(s, n - k)[n - k]
    sel = s >= thr
    if sel.sum() > k:
        idx = np.flatnonzero(sel)[np.argsort(-s[sel], kind="stable")][:k]
        sel = np.zeros(n, bool)
        sel[idx] = True
    return sel


class TieBreaker:
    """Seeded jitter that only reorders tied ranks.

    Each call returns U(0, scale / n) for n regions, i.e. less than one rank step, so it never reorders two distinct
    ranks. Drawn once per frame in decode order, so a run is reproducible from (seed, frame order).
    """

    def __init__(self, seed, n, scale=0.25):
        self.rng = np.random.default_rng(seed)
        self.n, self.high = n, scale / n

    def __call__(self):
        return self.rng.uniform(0.0, self.high, self.n)


def tie_seed(key):
    """Deterministic seed from a text key (e.g. the clip name)."""
    return zlib.crc32(str(key).encode())


# ---------------------------------------------------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------------------------------------------------

def soft_nms(boxes_xywh, scores, sigma=0.5, floor=0.001):
    """Class-agnostic Gaussian Soft-NMS (Bodla et al., ICCV 2017).

    Returns (indices of the kept boxes in selection order, their decayed scores). A box whose score decays below
    `floor` is dropped.
    """
    s = np.asarray(scores, np.float64).copy()
    b = np.asarray(boxes_xywh, np.float64).reshape(-1, 4)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = x1 + b[:, 2], y1 + b[:, 3]
    area = b[:, 2] * b[:, 3]
    idx = np.arange(len(s))
    keep, kept = [], []
    while idx.size:
        j = idx[np.argmax(s[idx])]
        keep.append(j)
        kept.append(s[j])
        idx = idx[idx != j]
        if not idx.size:
            break
        iw = np.maximum(0.0, np.minimum(x2[j], x2[idx]) - np.maximum(x1[j], x1[idx]))
        ih = np.maximum(0.0, np.minimum(y2[j], y2[idx]) - np.maximum(y1[j], y1[idx]))
        inter = iw * ih
        iou = inter / np.maximum(area[j] + area[idx] - inter, 1e-9)
        s[idx] *= np.exp(-(iou ** 2) / sigma)
        idx = idx[s[idx] >= floor]
    return np.asarray(keep, np.int64), np.asarray(kept, np.float64)


def merge_soft_nms(boxes, scores, sigma=0.5, floor=0.001, max_det=300):
    """Soft-NMS over the union of all boxes, then the `max_det` highest decayed scores."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    if not len(boxes):
        return boxes, scores
    keep, decayed = soft_nms(boxes, scores, sigma, floor)
    order = np.argsort(-decayed, kind="stable")[:max_det]
    return boxes[keep][order], decayed[order].astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------------------------------

class StageClock:
    """Wall-clock per pipeline stage. With `sync`, waits for the GPU at each boundary so GPU time lands in its stage."""

    def __init__(self, sync=None):
        self.sync = sync or (lambda: None)
        self.ms = {}
        self._name, self._t = None, None

    def start(self, name):
        self.stop()
        self.sync()
        self._name, self._t = name, time.perf_counter()

    def stop(self):
        if self._name is not None:
            self.sync()
            self.ms[self._name] = self.ms.get(self._name, 0.0) + (time.perf_counter() - self._t) * 1000.0
            self._name = None

    def pop(self):
        self.stop()
        out, self.ms = self.ms, {}
        return out


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return torch.cuda.synchronize
    except ImportError:
        pass
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------------------------------------------------

def load_config(path):
    """Read a YAML or JSON config into a dict; remembers its own location under '_path'."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    text = path.read_text()
    if path.suffix.lower() == ".json":
        cfg = json.loads(text)
    else:
        import yaml
        cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: a detector config must be a mapping")
    cfg["_path"] = str(path.resolve())
    return cfg


def parse_overrides(items):
    """['conf=0.01', 'max_det=200', 'half=true'] -> {'conf': 0.01, 'max_det': 200, 'half': True}."""
    import yaml
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not key=value")
        k, v = item.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def resolve_path(p, base=None):
    """Absolute path of `p`: as given, else relative to the working directory, `base`, or the repository root."""
    if p is None:
        return None
    p = Path(os.path.expanduser(str(p)))
    if p.is_absolute():
        return p
    for root in (Path.cwd(), Path(base) if base else None, REPO):
        if root is not None and (root / p).exists():
            return (root / p).resolve()
    return (Path.cwd() / p).resolve()


# ---------------------------------------------------------------------------------------------------------------------
# MOTChallenge-format I/O
# ---------------------------------------------------------------------------------------------------------------------

def read_mot_boxes(path, drop_invalid=True):
    """gt.txt / det.txt -> {frame: (boxes xywh (n, 4) float32, scores (n,) float32)}.

    For ground truth, rows whose seventh column (the MOT 'consider' flag) is 0 are dropped when `drop_invalid`.
    """
    rows = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            if drop_invalid and float(p[6]) == 0:
                continue
            fr = int(float(p[0]))
            rows.setdefault(fr, []).append([float(v) for v in p[2:7]])
    return {fr: (np.asarray(r, np.float32)[:, :4], np.asarray(r, np.float32)[:, 4]) for fr, r in rows.items()}


def write_mot_dets(f, frame, boxes_xywh, scores):
    """Append one frame of detections as 'frame,-1,x,y,w,h,score,-1,-1,-1'."""
    for (x, y, w, h), s in zip(np.asarray(boxes_xywh).reshape(-1, 4), np.asarray(scores).reshape(-1)):
        f.write(f"{frame},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.5f},-1,-1,-1\n")

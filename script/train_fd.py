#!/usr/bin/env python3
"""Train the FD network f_theta (1,361 parameters) of the frame-difference cue.

Inputs are the FD tile inputs of score.FDCue on the first `--train-frac` of every training clip, in decode order and
with the clip's FD state from its first frame. A tile is positive when it contains an annotated object centre and at
least 2 px of that box in each axis. Recipe of the paper: class-weighted binary cross-entropy (positive weight =
negatives / positives), Adam (lr 2e-3), cosine schedule over 30 epochs, batch 4,096, seed 0; 10% of the rows are held
out for validation and the epoch with the best validation ROC-AUC is kept.

    python script/train_fd.py --clip train1.mp4 train1/gt/gt.txt --clip train2.mp4 train2/gt/gt.txt \\
                              --out weights/fd_gate.pt

Inputs are cached as uint8 in a memory-mapped file (8 KB per tile per frame) next to --out, removed at the end.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode import count_frames, open_video  # noqa: E402
from score import FDCue, FDNet  # noqa: E402
from util import STRIDE, TILE, read_mot_boxes, tile_label, tile_origins  # noqa: E402


def roc_auc(scores, labels):
    labels = np.asarray(labels).astype(bool)
    npos, nneg = labels.sum(), (~labels).sum()
    if not npos or not nneg:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def collect(a, cache):
    """FD inputs and tile labels of the training frames of every clip -> (X memmap (rows, 2, g, g) uint8, y)."""
    plan = []
    for video, gt in a.clip:
        n = count_frames(video, a.decoder_threads)
        plan.append((video, read_mot_boxes(gt), max(1, int(n * a.train_frac)), n))
    g = a.tile // a.fd_cell
    rows = 0
    sizes = []
    for video, _, nt, _ in plan:
        with open_video(video, bitcost=False, tile=a.tile, stride=a.stride) as r:
            f = next(r)
        sizes.append((f.width, f.height))
        rows += nt * len(tile_origins(f.width, f.height, a.tile, a.stride))
    X = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=(rows, 2, g, g))
    y = np.zeros(rows, np.uint8)
    i = 0
    for (video, gt, nt, n), (W, H) in zip(plan, sizes):
        origins = tile_origins(W, H, a.tile, a.stride)
        fd = FDCue(None, W, H, origins, a.device, a.tile, a.fd_cell, a.fd_history)       # fresh state per clip
        t0 = time.time()
        with open_video(video, bitcost=False, tile=a.tile, stride=a.stride, threads=a.decoder_threads) as frames:
            for frame in frames:
                if frame.index > nt:
                    break
                X[i:i + len(origins)] = fd.update(frame.y).cpu().numpy()
                boxes = gt.get(frame.index, (np.zeros((0, 4)),))[0]
                y[i:i + len(origins)] = [tile_label(boxes, x, yy, a.tile, a.min_px) for x, yy in origins]
                i += len(origins)
        print(f"  {Path(video).name}: {nt} of {n} frames, {nt * len(origins):,} tiles, "
              f"{int(y[i - nt * len(origins):i].sum()):,} positive, {time.time() - t0:.0f} s", flush=True)
    assert i == rows, (i, rows)
    return X, y


def batch(X, idx, dev):
    return torch.from_numpy(np.ascontiguousarray(X[idx])).to(dev).float() / 255.0


@torch.no_grad()
def score(net, X, idx, dev, bs=8192):
    net.eval()
    return np.concatenate([torch.sigmoid(net(batch(X, idx[i:i + bs], dev))).cpu().numpy()
                           for i in range(0, len(idx), bs)]) if len(idx) else np.zeros(0, np.float32)


def train(a, X, y):
    dev = torch.device(a.device)
    rng = np.random.default_rng(a.seed)
    perm = np.arange(len(y))
    rng.shuffle(perm)
    nv = max(1, int(len(perm) * a.val_frac))
    val, tr = np.sort(perm[:nv]), np.sort(perm[nv:])
    torch.manual_seed(a.seed)
    net = FDNet(2).to(dev)
    yt = torch.from_numpy(y.astype(np.float32))
    pos = max(int(y[tr].sum()), 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(tr) - pos) / pos], device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_state = -1.0, None
    print(f"train {len(tr):,} rows ({pos:,} positive), validation {len(val):,} rows, "
          f"{sum(p.numel() for p in net.parameters()):,} parameters", flush=True)
    for ep in range(a.epochs):
        net.train()
        order = tr[np.random.default_rng(ep).permutation(len(tr))]
        t0 = time.time()
        for i in range(0, len(order), a.batch):
            j = np.sort(order[i:i + a.batch])
            opt.zero_grad()
            loss_fn(net(batch(X, j, dev)), yt[j].to(dev)).backward()
            opt.step()
        sched.step()
        auc = roc_auc(score(net, X, val, dev), y[val])
        if auc >= best:
            best, best_state = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}
        print(f"  epoch {ep + 1}/{a.epochs}: validation AUC {auc:.4f} (best {best:.4f}), {time.time() - t0:.0f} s", flush=True)
    net.load_state_dict(best_state)
    return net, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--clip", nargs=2, action="append", required=True, metavar=("VIDEO", "GT"),
                    help="training clip: H.264 video and its MOTChallenge gt.txt (repeat per clip)")
    ap.add_argument("--out", default="weights/fd_gate.pt")
    ap.add_argument("--train-frac", type=float, default=0.2, help="leading fraction of each clip used for training")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--fd-cell", type=int, default=4)
    ap.add_argument("--fd-history", type=int, default=7)
    ap.add_argument("--min-px", type=float, default=2.0, help="positive tile: box extent inside the tile per axis")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--decoder-threads", type=int, default=4)
    ap.add_argument("--keep-cache", action="store_true", help="keep the cached inputs (<out>.inputs.npy)")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.with_suffix(".inputs.npy")
    try:
        X, y = collect(a, cache)
        net, best = train(a, X, y)
    finally:
        if not a.keep_cache and cache.exists():
            cache.unlink()
    torch.save(net.state_dict(), out)
    out.with_suffix(".json").write_text(json.dumps(dict(
        validation_auc=best, rows=int(len(y)), positives=int(y.sum()), clips=[v for v, _ in a.clip],
        **{k: v for k, v in vars(a).items() if k not in ("clip", "out")}), indent=2) + "\n")
    print(f"-> {out} (validation AUC {best:.4f})")


if __name__ == "__main__":
    main()

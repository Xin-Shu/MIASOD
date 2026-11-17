"""Tile scores of MIASOD and their fusion.

Every scoring tile tau_j (256 px, stride 192) of frame t receives up to three cue scores:
  coarse appearance  g^C   max confidence of the coarse-pass detections whose centre lies in the tile
  frame difference   g^FD  a 1,361-parameter CNN on the pooled current and accumulated frame difference
  bit cost           g^BC  max macroblock bit cost in the tile (computed by the decoder, see decode.py)
and rank_mean() fuses them: g^M = mean over cues of rank(g) / N, ranked within the frame.

FD is the only stateful cue. FDCue keeps, per clip, the previous grey frame and the pooled differences of the last
`history` frames. Feed it every decoded frame in decode order, and call reset() at every clip boundary.
"""
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

from scipy.stats import rankdata

from util import TILE


# ---------------------------------------------------------------------------------------------------------------------
# Coarse appearance
# ---------------------------------------------------------------------------------------------------------------------

def tile_max_by_centre(boxes_xywh, scores, origins, tile=TILE):
    """g^C(j) = max({0} u {p_k : centre of box k in tile j})."""
    origins = np.asarray(origins, np.float64).reshape(-1, 2)
    out = np.zeros(len(origins), np.float32)
    if not len(scores):
        return out
    b = np.asarray(boxes_xywh, np.float64)
    cx, cy = b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3] / 2.0
    ox, oy = origins[:, 0], origins[:, 1]
    inside = (cx[:, None] >= ox) & (cx[:, None] < ox + tile) & (cy[:, None] >= oy) & (cy[:, None] < oy + tile)
    return np.where(inside, np.asarray(scores, np.float32)[:, None], 0).max(0).astype(np.float32)


class CoarseCue:
    """Coarse pass: one detector call on the whole frame (resized by the detector, 1,024 px longer side by default).

    After each call, `boxes` (xywh, frame coordinates) and `scores` hold the coarse-pass detections, which the merge
    stage reuses.
    """

    def __init__(self, detector, origins, tile=TILE):
        self.detector, self.origins, self.tile = detector, origins, tile
        self.boxes, self.scores = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

    def __call__(self, bgr):
        self.boxes, self.scores = self.detector([bgr])[0]
        return tile_max_by_centre(self.boxes, self.scores, self.origins, self.tile)


# ---------------------------------------------------------------------------------------------------------------------
# Frame difference
# ---------------------------------------------------------------------------------------------------------------------

class FDNet(nn.Module):
    """f_theta: two stride-2 convolutions (2 -> 8 -> 16 channels), global average pooling, one linear unit.

    1,361 parameters. Input (n, 2, 64, 64) in [0, 1], output one logit per tile.
    """

    def __init__(self, channels=2):
        super().__init__()

        def stage(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU())

        self.body = nn.Sequential(stage(channels, 8), stage(8, 16))
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        return self.head(self.body(x).mean((2, 3))).squeeze(1)


def limited_to_full_range(y):
    """Limited-range luma (16..235) to full-range grey (0..255), bit-exact with FFmpeg's '-pix_fmt gray' output."""
    return (((y.int() - 16) * 19077 + 8192) >> 14).clamp(0, 255).to(torch.uint8)


def gather_index(origins, cells_h, cells_w, g, cell):
    """Flat indices (n, g, g) of the g x g cells of every tile in a cells_h x cells_w grid (clamped at the border)."""
    d = np.arange(g)
    idx = np.empty((len(origins), g, g), np.int64)
    for i, (x, y) in enumerate(origins):
        rr = np.clip(y // cell + d, 0, cells_h - 1)
        cc = np.clip(x // cell + d, 0, cells_w - 1)
        idx[i] = rr[:, None] * cells_w + cc[None, :]
    return idx


def minmax_u8(x):
    """Min-max normalise over all cells of all tiles of the frame and quantise to 8 bits (returned as /255 floats)."""
    lo, hi = x.min(), x.max()
    x = torch.where((hi - lo) > 0, (x - lo) / (hi - lo).clamp_min(1e-12), torch.zeros_like(x))
    return (x * 255).clamp(0, 255).to(torch.uint8)


class FDCue:
    """g^FD with its temporal state.

    Per decoded frame t (luma Y_t):
        Delta_t = |grey(Y_t) - grey(Y_{t-1})|, with Delta_1 = 0 at the first frame of a clip
        P_t     = mean of Delta_t over 4 x 4-px cells
        A_t     = P_t + P_{t-1} + ... + P_{t-6}          (P = 0 before the clip starts)
        X_t(j)  = [P_t, A_t] on the 64 x 64 cells of tile j, each min-max normalised over the frame, 8-bit
        g^FD    = sigmoid(f_theta(X_t(j)))

    State (all on `device`): the previous grey frame, the last `history` cell maps P and their running sum A. Memory is
    one full-resolution uint8 frame plus (history + 1) cell maps of (H/4) x (W/4) float32, e.g. about 82 MB at 8K.
    P holds exact multiples of 1/16 below 256, so the running sum A is exact in float32 and never drifts.
    """

    def __init__(self, weights, width, height, origins, device="cuda:0", tile=TILE, cell=4, history=7):
        if tile % cell:
            raise ValueError("tile must be a multiple of the FD cell size")
        self.device = torch.device(device)
        self.width, self.height, self.cell, self.history = width, height, cell, history
        self.ch, self.cw = -(-height // cell), -(-width // cell)
        self.idx = torch.from_numpy(gather_index(origins, self.ch, self.cw, tile // cell, cell)).to(self.device)
        self.net = FDNet(2).to(self.device).eval()
        if weights is not None:
            state = torch.load(str(weights), map_location=self.device)
            self.net.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
        self.reset()

    def reset(self):
        """Forget the clip: call before the first frame of every new clip (and after any warm-up on real frames)."""
        self.prev = None
        self.past = collections.deque()
        self.acc = None

    @torch.no_grad()
    def update(self, y):
        """Advance the state by one decoded luma plane; returns the uint8 network input (n, 2, g, g) of this frame."""
        if y.shape != (self.height, self.width):
            raise ValueError(f"luma {y.shape} does not match the clip size {(self.height, self.width)}")
        grey = limited_to_full_range(torch.from_numpy(np.ascontiguousarray(y)).to(self.device))
        if self.prev is None:
            diff = torch.zeros(grey.shape, dtype=torch.float32, device=self.device)
        else:
            diff = (grey.int() - self.prev.int()).abs().float()
        self.prev = grey
        pad_h, pad_w = self.ch * self.cell - self.height, self.cw * self.cell - self.width
        if pad_h or pad_w:
            diff = TF.pad(diff, (0, pad_w, 0, pad_h))
        p = TF.avg_pool2d(diff[None, None], self.cell)[0, 0]
        self.past.append(p)
        self.acc = p.clone() if self.acc is None else self.acc + p
        if len(self.past) > self.history:
            self.acc -= self.past.popleft()
        return torch.stack([minmax_u8(p.reshape(-1)[self.idx]), minmax_u8(self.acc.reshape(-1)[self.idx])], 1)

    @torch.no_grad()
    def __call__(self, y):
        x = self.update(y).float() / 255.0
        return torch.sigmoid(self.net(x)).float().cpu().numpy()

    @torch.no_grad()
    def warmup(self, n=3):
        """Run the network on dummy inputs without touching the temporal state."""
        x = torch.zeros((self.idx.shape[0], 2, self.idx.shape[1], self.idx.shape[2]), device=self.device)
        for _ in range(n):
            self.net(x)


# ---------------------------------------------------------------------------------------------------------------------
# Bit cost
# ---------------------------------------------------------------------------------------------------------------------

def bc_from_grid(grid, origins, tile=TILE, mb=16):
    """g^BC(j) = max macroblock bit cost in tile j, from a (mb_h, mb_w) grid; equals the decoder's own tile scores."""
    grid = np.asarray(grid)
    mh, mw = grid.shape
    n = tile // mb
    out = np.empty(len(origins), np.float32)
    for i, (x, y) in enumerate(origins):
        r0, c0 = y // mb, x // mb
        rows = np.minimum(np.arange(r0, r0 + n), mh - 1)
        cols = np.minimum(np.arange(c0, c0 + n), mw - 1)
        out[i] = grid[np.ix_(rows, cols)].max()
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------------------------------------------------

def rank_mean(*cues):
    """g^M(j) = mean over the given cues of rank(g(j)) / N; ranks are within-frame and ascending, ties share ranks."""
    if not cues:
        raise ValueError("rank_mean needs at least one cue")
    n = len(cues[0])
    if any(len(c) != n for c in cues):
        raise ValueError("all cues must score the same tiles")
    # Sum the integer-valued ranks before dividing: tiles with equal rank sums then get exactly equal fused scores.
    return sum(rankdata(np.asarray(c, np.float64)) for c in cues) / (len(cues) * n)

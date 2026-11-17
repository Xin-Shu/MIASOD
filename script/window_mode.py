"""Window mode: detect the top-K windows at native resolution.

1. Project the fused tile scores onto windows: s(w) = max of g^M over the tiles whose centre lies in window w.
2. Keep the n_K = max(1, round(K H)) windows with the highest s(w) (H windows per frame).
3. Run the window detector on the kept windows at native resolution.
4. Ownership merge: keep a window box only if, among the kept windows that contain its centre, the producing window
   holds that centre most centrally. Two overlapping kept windows thus split their overlap at the middle line, and
   a box is never assigned to a skipped neighbour.
5. Add the global (coarse-pass) boxes scoring at least `merge_conf` and apply Gaussian Soft-NMS to the union.
"""
import numpy as np

from util import TILE, TieBreaker, merge_soft_nms, rank, tie_seed, topk_mask


def axis(length, size, overlap):
    """Evenly spaced window starts covering [0, length): first at 0, last at length - size, overlap >= `overlap`."""
    if length <= size:
        return [0]
    n = -(-(length - overlap) // (size - overlap))
    return [round(i * (length - size) / (n - 1)) for i in range(n)]


def window_grid(width, height, size=512, overlap=128):
    """Window origins (x, y), row-major."""
    if not 0 <= overlap < size:
        raise ValueError("window overlap must be in [0, size)")
    return np.array([(x, y) for y in axis(height, size, overlap) for x in axis(width, size, overlap)], np.int64)


def membership(origins, windows, tile=TILE, size=512):
    """(H, N) bool: tile j belongs to window w when the tile centre lies inside the window."""
    c = np.asarray(origins, np.float64) + tile / 2.0
    w = np.asarray(windows, np.float64)
    m = ((c[None, :, 0] >= w[:, None, 0]) & (c[None, :, 0] < w[:, None, 0] + size) &
         (c[None, :, 1] >= w[:, None, 1]) & (c[None, :, 1] < w[:, None, 1] + size))
    if not m.any(1).all():
        raise ValueError("a window contains no tile centre: choose a window size of at least the tile stride")
    return m


def _lex_greater(a, b):
    """Row-wise lexicographic a > b."""
    gt = np.zeros(len(a), bool)
    eq = np.ones(len(a), bool)
    for k in range(a.shape[1]):
        gt |= eq & (a[:, k] > b[:, k])
        eq &= a[:, k] == b[:, k]
    return gt


def ownership(boxes_xywh, win, xs, ys, size, ran):
    """Mask of the boxes that their producing window owns.

    Candidate owners of a box are the executed windows (`ran`) containing its centre, plus the producing window. The
    owner maximises (distance from the centre to the nearest window edge, sum of the x and y margins, x, y), in that
    order; xs and ys are the sorted window starts of the row-major grid.
    """
    nx = len(xs)
    ran = np.asarray(ran, bool)
    own = np.asarray(win, np.int64)
    b = np.asarray(boxes_xywh, np.float64)
    cx, cy = b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3] / 2.0
    jx, jy = np.searchsorted(xs, cx, side="right"), np.searchsorted(ys, cy, side="right")
    ox, oy = xs[own % nx], ys[own // nx]
    mx, my = np.minimum(cx - ox, ox + size - cx), np.minimum(cy - oy, oy + size - cy)
    best = np.stack([np.minimum(mx, my), mx + my, ox, oy], 1)
    best_w = own.copy()
    gap = lambda v: np.diff(v).min() if len(v) > 1 else size
    mxs, mys = int(np.ceil(size / gap(xs))) + 1, int(np.ceil(size / gap(ys))) + 1
    for dy in range(1, mys + 1):
        for dx in range(1, mxs + 1):
            ix, iy = jx - dx, jy - dy
            ok = (ix >= 0) & (iy >= 0)
            ix_, iy_ = np.where(ok, ix, 0), np.where(ok, iy, 0)
            w = iy_ * nx + ix_
            ok &= ran[w]
            ox, oy = xs[ix_], ys[iy_]
            mx, my = np.minimum(cx - ox, ox + size - cx), np.minimum(cy - oy, oy + size - cy)
            ok &= (cx >= ox) & (cx < ox + size) & (cy >= oy) & (cy < oy + size)
            cand = np.stack([np.minimum(mx, my), mx + my, ox, oy], 1)
            better = ok & _lex_greater(cand, best)
            best[better] = cand[better]
            best_w[better] = w[better]
    return best_w == own


def cut(bgr, x, y, size):
    """size x size region at (x, y); zero-padded at the bottom/right when the frame is smaller."""
    r = bgr[y:y + size, x:x + size]
    if r.shape[:2] != (size, size):
        out = np.zeros((size, size) + r.shape[2:], r.dtype)
        out[:r.shape[0], :r.shape[1]] = r
        r = out
    return np.ascontiguousarray(r)


class WindowMode:
    def __init__(self, detector, width, height, origins, keep=0.2, size=512, overlap=128, tile=TILE,
                 merge_conf=0.25, sigma=0.5, floor=0.001, max_det=300, tie_key=None):
        self.detector, self.keep, self.size = detector, keep, size
        self.merge_conf, self.sigma, self.floor, self.max_det = merge_conf, sigma, floor, max_det
        self.windows = window_grid(width, height, size, overlap)
        self.xs = np.unique(self.windows[:, 0]).astype(np.float64)
        self.ys = np.unique(self.windows[:, 1]).astype(np.float64)
        self.members = membership(origins, self.windows, tile, size)
        # Seeded jitter below one rank step breaks exact ties at the K cut; the key of the paper runs is '<clip>/win512'.
        self.jitter = TieBreaker(tie_seed(f"{tie_key}/win{size}"), len(self.windows)) if tie_key is not None else None

    @property
    def n_regions(self):
        return len(self.windows)

    def window_scores(self, fused):
        """s(w): max fused tile score over the tiles centred in window w."""
        return np.where(self.members, np.asarray(fused, np.float64)[None, :], -np.inf).max(1)

    def select(self, fused):
        """Boolean mask of the kept windows. The jitter is drawn every frame, so call this once per decoded frame."""
        s = rank(self.window_scores(fused))
        if self.jitter is not None:
            s = s + self.jitter()
        return topk_mask(s, self.keep)

    def regions(self, bgr, run):
        idx = np.flatnonzero(run)
        return idx, [cut(bgr, int(self.windows[w, 0]), int(self.windows[w, 1]), self.size) for w in idx]

    def detect(self, idx, images):
        """Window detections in frame coordinates: boxes xywh, scores, producing window index."""
        B, S, W = [], [], []
        for w, (b, s) in zip(idx, self.detector(images)):
            if len(b):
                B.append(b + np.array([self.windows[w, 0], self.windows[w, 1], 0, 0], np.float32))
                S.append(s)
                W.append(np.full(len(b), w, np.int64))
        if not B:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        return np.concatenate(B).astype(np.float32), np.concatenate(S).astype(np.float32), np.concatenate(W)

    def merge(self, boxes, scores, win, run, global_boxes, global_scores):
        if len(boxes):
            own = ownership(boxes, win, self.xs, self.ys, self.size, run)
            boxes, scores = boxes[own], scores[own]
        g = np.asarray(global_scores) >= self.merge_conf
        boxes = np.concatenate([boxes, np.asarray(global_boxes, np.float32).reshape(-1, 4)[g]])
        scores = np.concatenate([scores, np.asarray(global_scores, np.float32)[g]])
        return merge_soft_nms(boxes, scores, self.sigma, self.floor, self.max_det)

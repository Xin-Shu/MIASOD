"""Crop mode: group the top-K tiles into crops, detect the crops resized to a longer side of 1,024 px.

1. Keep the n_K = max(1, round(K N)) tiles with the highest fused score g^M (N tiles per frame).
2. Group the kept tiles that are adjacent in the 8-connected sense on the tile grid.
3. Split the bounding rectangle of each group into equal crops of at most `max_side` px per side, each extended by
   overlap / 2 px towards its neighbouring crops of the same group.
4. Run the crop detector on every crop (the detector resizes it to its input size, upsampling small crops).
5. Apply Gaussian Soft-NMS to the union of the crop boxes and the global (coarse-pass) boxes.
"""
import numpy as np
from scipy import ndimage

from util import TILE, TieBreaker, merge_soft_nms, n_keep, rank, tie_seed


def select_tiles(fused, keep, jitter=None):
    """Indices of the n_K highest-ranked tiles, best first (ties: lower index first unless jittered)."""
    s = rank(fused)
    if jitter is not None:
        s = s + jitter()
    return np.argsort(-s, kind="stable")[:n_keep(keep, len(s))]


def crops_from_tiles(selected, origins, width, height, tile=TILE, max_side=1024, overlap=64):
    """Kept tiles -> 8-connected groups on the tile grid -> crops (left, top, right, bottom) of at most max_side px
    (plus the overlap margins)."""
    xs = sorted({x for x, _ in origins})
    ys = sorted({y for _, y in origins})
    ix = {x: i for i, x in enumerate(xs)}
    iy = {y: i for i, y in enumerate(ys)}
    grid = np.zeros((len(ys), len(xs)), bool)
    for t in selected:
        grid[iy[origins[t][1]], ix[origins[t][0]]] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3), int))
    half = overlap // 2
    out = []
    for k in range(1, n + 1):
        r, c = np.nonzero(labels == k)
        left, top = xs[c.min()], ys[r.min()]
        right, bottom = min(width, xs[c.max()] + tile), min(height, ys[r.max()] + tile)
        nx, ny = -(-(right - left) // max_side), -(-(bottom - top) // max_side)
        for i in range(nx):
            for j in range(ny):
                x0, x1 = left + (right - left) * i // nx, left + (right - left) * (i + 1) // nx
                y0, y1 = top + (bottom - top) * j // ny, top + (bottom - top) * (j + 1) // ny
                out.append((max(left, x0 - (half if i else 0)), max(top, y0 - (half if j else 0)),
                            min(right, x1 + (half if i < nx - 1 else 0)), min(bottom, y1 + (half if j < ny - 1 else 0))))
    return out


class CropMode:
    def __init__(self, detector, width, height, origins, keep=0.2, tile=TILE, max_side=1024, overlap=64,
                 merge_conf=0.0, sigma=0.5, floor=0.001, max_det=300, tie_key=None):
        self.detector, self.keep, self.tile = detector, keep, tile
        self.width, self.height, self.origins = width, height, [tuple(map(int, o)) for o in origins]
        self.max_side, self.overlap = max_side, overlap
        self.merge_conf, self.sigma, self.floor, self.max_det = merge_conf, sigma, floor, max_det
        # Seeded jitter below one rank step breaks exact ties at the K cut; the key of the paper runs is '<clip>'.
        self.jitter = TieBreaker(tie_seed(tie_key), len(origins)) if tie_key is not None else None

    @property
    def n_regions(self):
        return len(self.origins)

    def select(self, fused):
        """Boolean mask of the kept tiles. The jitter is drawn every frame, so call this once per decoded frame."""
        run = np.zeros(len(self.origins), bool)
        run[select_tiles(fused, self.keep, self.jitter)] = True
        return run

    def regions(self, bgr, run):
        crops = crops_from_tiles(np.flatnonzero(run), self.origins, self.width, self.height, self.tile, self.max_side,
                                 self.overlap)
        return crops, [bgr[t:b, l:r] for (l, t, r, b) in crops]

    def detect(self, crops, images):
        B, S = [], []
        for (l, t, _, _), (b, s) in zip(crops, self.detector(images)):
            if len(b):
                B.append(b + np.array([l, t, 0, 0], np.float32))
                S.append(s)
        if not B:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        return np.concatenate(B).astype(np.float32), np.concatenate(S).astype(np.float32)

    def merge(self, boxes, scores, global_boxes, global_scores):
        g = np.asarray(global_scores) >= self.merge_conf
        boxes = np.concatenate([np.asarray(global_boxes, np.float32).reshape(-1, 4)[g], boxes])
        scores = np.concatenate([np.asarray(global_scores, np.float32)[g], scores])
        return merge_soft_nms(boxes, scores, self.sigma, self.floor, self.max_det)

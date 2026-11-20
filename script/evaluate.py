#!/usr/bin/env python3
"""AP of MIASOD detections against MOTChallenge-format ground truth, pooled over clips.

Reports mAP (IoU .50:.95), AP50, AP25, and on COCO small objects (area <= 32^2 px^2) mAP_small, AP50_small and
AP25_small, in percent. Detections of all clips are pooled into one evaluation (AP is not averaged over clips).

    python script/evaluate.py --pair runs/clipA/detections.txt data/clipA/gt/gt.txt \\
                              --pair runs/clipB/detections.txt data/clipB/gt/gt.txt --max-det 300

The evaluated frames of a clip are those that hold ground truth or detections (frames with neither change no AP);
--frames A:B restricts every clip to frames A..B. Every ground-truth row counts, whatever its seventh (consider) column.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import read_mot_boxes  # noqa: E402

IOUS = np.round(np.concatenate([[0.25], np.arange(0.5, 0.951, 0.05)]), 2)
SMALL = 32 ** 2


def build(pairs, frames=None):
    images, gts, dts = [], [], []
    for c, pair in enumerate(pairs):
        det_path, gt_path = pair[:2]
        last = pair[2] if len(pair) > 2 else None          # optional: only frames 1..last were processed
        gt = read_mot_boxes(gt_path, drop_invalid=False)
        dt = read_mot_boxes(det_path, drop_invalid=False)
        keys = sorted(set(gt) | set(dt))
        if frames:
            keys = [f for f in keys if frames[0] <= f <= frames[1]]
        if last:
            keys = [f for f in keys if f <= last]
        for f in keys:
            iid = len(images) + 1
            images.append(dict(id=iid, clip=c, frame=f))
            for b in gt.get(f, (np.zeros((0, 4)),))[0]:
                gts.append(dict(id=len(gts) + 1, image_id=iid, category_id=1, bbox=[float(v) for v in b],
                                area=float(b[2] * b[3]), iscrowd=0))
            if f in dt:
                for b, s in zip(*dt[f]):
                    dts.append(dict(image_id=iid, category_id=1, bbox=[float(v) for v in b], score=float(s)))
    return images, gts, dts


def evaluate(pairs, max_det=300, frames=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, gts, dts = build(pairs, frames)
    if not gts:
        raise SystemExit("no ground truth in the evaluated frames")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = dict(images=images, annotations=gts, categories=[dict(id=1, name="object")])
        gt.createIndex()
        dt = gt.loadRes(dts) if dts else COCO()
        E = COCOeval(gt, dt, "bbox")
        E.params.iouThrs = IOUS
        E.params.areaRng = [[0, 1e10], [0, SMALL]]
        E.params.areaRngLbl = ["all", "small"]
        E.params.maxDets = [max_det]
        E.evaluate()
        E.accumulate()
    prec = E.eval["precision"]                                   # [iou, recall, class, area, maxdet]

    def ap(iou=None, area=0):
        p = prec[:, :, :, area, 0]
        p = p[1:] if iou is None else p[[int(np.flatnonzero(np.isclose(IOUS, iou))[0])]]
        p = p[p > -1]
        return round(100.0 * float(p.mean()), 2) if p.size else float("nan")

    return dict(mAP=ap(), AP50=ap(0.5), AP25=ap(0.25), mAP_small=ap(area=1), AP50_small=ap(0.5, 1),
                AP25_small=ap(0.25, 1), images=len(images), objects=len(gts), detections=len(dts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("DETECTIONS", "GT"),
                    help="detections.txt and gt.txt of one clip (repeat per clip)")
    ap.add_argument("--max-det", type=int, default=300, help="detections per frame (2000 for PANDA)")
    ap.add_argument("--frames", default=None, help="A:B, evaluate only frames A..B of every clip")
    ap.add_argument("--json", default=None, help="also write the metrics to this file")
    a = ap.parse_args()
    frames = tuple(int(v) for v in a.frames.split(":")) if a.frames else None
    res = evaluate(a.pair, a.max_det, frames)
    print("  ".join(f"{k} {v}" for k, v in res.items()))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

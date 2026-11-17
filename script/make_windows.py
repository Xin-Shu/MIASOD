#!/usr/bin/env python3
"""Cut native-resolution training windows for the window detector (YOLO and COCO formats).

Per selected frame, every window of the evaluation grid (512 px, overlap >= 128 px by default) receives the boxes whose
centre lies inside it, clipped to the window; clipped boxes thinner than 2 px are dropped. All windows holding a box are
kept, plus `--empty-per-frame` windows without one (seeded per clip and frame).

    python script/make_windows.py --clip a.mp4 a/gt/gt.txt --clip b.mp4 b/gt/gt.txt --range 0:0.2 --split train \\
        --out data/windows512

Writes <out>/<split>/images/*.png, <out>/<split>/labels/*.txt (YOLO), <out>/<split>_coco.json (COCO, file names
relative to <out>/<split>/) and <out>/dataset.yaml for Ultralytics. Frames come from the decoded video, the same pixels
the detector sees at test time. --range A:B selects the frames f with A < f / n <= B of a clip of n frames, and --step
keeps every step-th of them (paper: --range 0:0.2 for train, --range 0.2:1 --step 10 for val).
"""
import argparse
import json
import random
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode import count_frames, open_video  # noqa: E402
from util import read_mot_boxes  # noqa: E402
from window_mode import cut, window_grid  # noqa: E402


def window_boxes(boxes, x0, y0, size, min_px):
    out = []
    for x, y, w, h in boxes:
        cx, cy = x + w / 2.0, y + h / 2.0
        if x0 <= cx < x0 + size and y0 <= cy < y0 + size:
            a0, b0 = max(x, x0), max(y, y0)
            a1, b1 = min(x + w, x0 + size), min(y + h, y0 + size)
            if a1 - a0 >= min_px and b1 - b0 >= min_px:
                out.append((a0 - x0, b0 - y0, a1 - a0, b1 - b0))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--clip", nargs=2, action="append", required=True, metavar=("VIDEO", "GT"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--range", default="0:0.2", help="frame range A:B as fractions of each clip")
    ap.add_argument("--step", type=int, default=1, help="keep every step-th frame of the range")
    ap.add_argument("--size", type=int, default=512, help="window size (px)")
    ap.add_argument("--overlap", type=int, default=128, help="minimum window overlap (px)")
    ap.add_argument("--empty-per-frame", type=int, default=5, help="windows without objects kept per frame")
    ap.add_argument("--min-px", type=float, default=2.0)
    a = ap.parse_args()
    lo, hi = (float(v) for v in a.range.split(":"))
    root = Path(a.out)
    img_dir, lab_dir = root / a.split / "images", root / a.split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for video, gt_path in a.clip:
        clip, gt, n = Path(video).stem, read_mot_boxes(gt_path), count_frames(video)
        first, last = int(lo * n) + 1, int(hi * n)
        kept = 0
        with open_video(video, bitcost=False) as frames:
            for frame in frames:
                if frame.index > last:
                    break
                if frame.index < first or (frame.index - first) % a.step:
                    continue
                boxes = gt.get(frame.index, (np.zeros((0, 4)),))[0]
                wins = [(int(x), int(y), window_boxes(boxes, x, y, a.size, a.min_px))
                        for x, y in window_grid(frame.width, frame.height, a.size, a.overlap)]
                pos = [w for w in wins if w[2]]
                empty = [w for w in wins if not w[2]]
                rng = random.Random(zlib.crc32(f"{clip}/{frame.index}".encode()))
                for x, y, bb in pos + rng.sample(empty, min(a.empty_per_frame, len(empty))):
                    name = f"{clip}_f{frame.index:04d}_x{x}_y{y}"
                    cv2.imwrite(str(img_dir / f"{name}.png"), cut(frame.bgr, x, y, a.size))
                    with open(lab_dir / f"{name}.txt", "w") as f:
                        for bx, by, bw, bh in bb:
                            f.write(f"0 {(bx + bw / 2) / a.size:.6f} {(by + bh / 2) / a.size:.6f} "
                                    f"{bw / a.size:.6f} {bh / a.size:.6f}\n")
                    iid = len(images) + 1
                    images.append(dict(id=iid, file_name=f"images/{name}.png", width=a.size, height=a.size))
                    for bx, by, bw, bh in bb:
                        anns.append(dict(id=len(anns) + 1, image_id=iid, category_id=1, iscrowd=0,
                                         bbox=[float(bx), float(by), float(bw), float(bh)], area=float(bw * bh)))
                    kept += 1
        print(f"  {clip}: frames {first}..{last} of {n}, {kept} windows", flush=True)
    (root / f"{a.split}_coco.json").write_text(json.dumps(dict(
        images=images, annotations=anns, categories=[dict(id=1, name="object")])))
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        yaml_path.write_text(f"path: {root.resolve()}\ntrain: train/images\nval: val/images\nnames:\n  0: object\n")
    print(f"-> {root / a.split}: {len(images)} windows, {len(anns)} boxes")


if __name__ == "__main__":
    main()

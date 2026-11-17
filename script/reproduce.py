#!/usr/bin/env python3
"""Five-fold evaluation of the paper: run MIASOD on the test clips of every fold with that fold's models, then evaluate
the detections of all folds together.

Data layout:     <data>/<clip>/<clip>.mp4 and <data>/<clip>/gt.txt
Weights layout:  <weights>/f<k>/coarse.pt, window.pt, fd.pt   (DINO-R50: also dino_window.pth, dino_full.pth)
Folds:           script/splits/folds.json

    python script/reproduce.py --dataset omnismall --data data/omnismall --weights weights/omnismall
    python script/reproduce.py --dataset panda --data data/panda --weights weights/panda --keep 0.05 0.1 0.2 0.5
    python script/reproduce.py --dataset omnismall --data data/omnismall --train-clips 1   # training clips of fold 1

Arguments that this script does not know are passed on to miasod.py (e.g. --mode crop, --cues coarse bc).
Writes <out>/k<K>/<clip>/det.txt (+ det_summary.json) and <out>/k<K>/results.json (pooled AP and
FPS = frames / summed wall time).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from util import SCRIPT  # noqa: E402

SPLITS = SCRIPT / "splits" / "folds.json"


def split(dataset):
    cfg = json.loads(SPLITS.read_text())
    if dataset not in cfg:
        raise SystemExit(f"unknown dataset '{dataset}'; choose from {[k for k in cfg if not k.startswith('_')]}")
    return cfg[dataset]


def train_clips(cfg, k):
    test = set(cfg["folds"][k - 1])
    drop = set(cfg["exclude_from_training"].get(str(k), []))
    return [c for f in cfg["folds"] for c in f if c not in test] + [c for c in cfg["train_only"] if c not in drop]


def clip_files(data, clip):
    video, gt = Path(data) / clip / f"{clip}.mp4", Path(data) / clip / "gt.txt"
    for p in (video, gt):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    return str(video), str(gt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", required=True, help="omnismall, panda or seadronessee")
    ap.add_argument("--data", required=True, help="folder with one sub-folder per clip")
    ap.add_argument("--weights", help="folder with one sub-folder f1 ... f5 per fold")
    ap.add_argument("--out", default=None, help="default: runs/<dataset>")
    ap.add_argument("--keep", type=float, nargs="+", default=[0.2], help="one or more budgets K")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="run only these folds (default: all)")
    ap.add_argument("--detector", choices=("yolov11", "dino"), default="yolov11", help="window detector family")
    ap.add_argument("--train-clips", type=int, metavar="K", default=None,
                    help="only print the '--clip VIDEO GT' arguments of the training clips of fold K")
    a, extra = ap.parse_known_args()
    cfg = split(a.dataset)

    if a.train_clips is not None:
        print(" ".join(f"--clip {v} {g}" for v, g in (clip_files(a.data, c) for c in train_clips(cfg, a.train_clips))))
        return
    if not a.weights:
        ap.error("--weights is required")

    import miasod
    from evaluate import evaluate

    folds = a.folds or list(range(1, len(cfg["folds"]) + 1))
    out_root = Path(a.out or f"runs/{a.dataset}")
    dense = ["--max-det", str(cfg["max_det"])] + (["--detector-opt", "window.max_det=200"] if cfg["max_det"] > 300 else [])
    for keep in a.keep:
        out = out_root / f"k{keep:.2f}"
        pairs, frames, wall = [], 0, 0.0
        for k in folds:
            w = Path(a.weights) / f"f{k}"
            files = [clip_files(a.data, c) for c in cfg["folds"][k - 1]]
            models = ["--coarse-weights", str(w / "coarse.pt"), "--fd-weights", str(w / "fd.pt")]
            if a.detector == "yolov11":
                models += ["--window-weights", str(w / "window.pt")]
            else:
                models += ["--window-detector", str(SCRIPT / "detectors/dino/window.yaml"),
                           "--window-weights", str(w / "dino_window.pth"),
                           "--global-detector", str(SCRIPT / "detectors/dino/full.yaml"),
                           "--global-weights", str(w / "dino_full.pth")]
            print(f"== {a.dataset}, K = {keep:.2f}, fold {k}: {', '.join(Path(v).stem for v, _ in files)}", flush=True)
            miasod.main(["-i"] + [v for v, _ in files] + ["-o", str(out / "{stem}" / "det.txt"), "--summary",
                                                          "--keep", str(keep)] + models + dense + extra)
            for v, g in files:
                s = json.loads((out / Path(v).stem / "det_summary.json").read_text())
                frames, wall = frames + s["frames"], wall + s["wall_seconds"]
                pairs.append((str(out / Path(v).stem / "det.txt"), g, s["frames"]))
        res = evaluate(pairs, cfg["max_det"])
        res.update(keep=keep, folds=folds, clips=len(pairs), frames=frames, fps=round(frames / wall, 3))
        (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
        print(f"== {a.dataset}, K = {keep:.2f}: " + "  ".join(f"{m} {res[m]}" for m in
              ("mAP", "mAP_small", "AP50", "AP25", "fps")) + f"  -> {out / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

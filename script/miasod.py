#!/usr/bin/env python3
"""MIASOD: Motion-Informed Acceleration of Small Object Detection in ultra-high-resolution video.

Runs the full pipeline on one or more H.264 videos, one clip per video:
    decode -> tile cues (coarse, FD, BC) -> rank mean -> top-K windows or crops -> native-resolution detection
    -> ownership / Soft-NMS merge with the coarse-pass boxes -> frame-level detections.
Every option defaults to the proposed setting of the paper (window mode, K = 20%, all three cues, YOLOv11-L), and
the models default to weights/yolov11l_coarse1024.pt, weights/yolov11l_window512.pt and weights/fd_gate.pt.

    python script/miasod.py -i video.mp4

writes video_dir/det.txt (MOTChallenge format: frame,-1,x,y,w,h,score,-1,-1,-1). --summary adds det_summary.json
(settings, FPS, mean milliseconds per stage) and --save-regions adds det_regions.npz (fused scores, kept regions).
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")   # avoids page-compaction stalls on large frame buffers

import numpy as np

from util import (SCRIPT, STRIDE, TILE, StageClock, cuda_sync, n_keep, parse_overrides, resolve_path, tile_origins,
                  write_mot_dets)

CUES = ("coarse", "fd", "bc")
ROLES = ("coarse", "window", "crop", "global")
DEFAULT_CONFIGS = {"coarse": SCRIPT / "detectors/yolov11/coarse.yaml", "window": SCRIPT / "detectors/yolov11/window.yaml",
                   "crop": SCRIPT / "detectors/yolov11/crop.yaml", "global": None}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = ap.add_argument_group("input and output")
    g.add_argument("-i", "--input", nargs="+", required=True, metavar="VIDEO",
                   help="H.264 videos (MP4/MKV, I- and P-frames only); each is one clip")
    g.add_argument("-o", "--output", default="{dir}/det.txt",
                   help="detection file per video; {dir} and {stem} are the video's folder and file name")
    g.add_argument("--summary", action="store_true", help="also write <output>_summary.json (settings, FPS, stage times)")
    g.add_argument("--save-regions", action="store_true",
                   help="also write <output>_regions.npz (fused tile scores and kept regions per frame)")

    g = ap.add_argument_group("selection")
    g.add_argument("--mode", choices=("window", "crop"), default="window", help="detection route")
    g.add_argument("--keep", type=float, default=0.2, help="keep fraction K in (0, 1]; 1 = exhaustive (no gating)")
    g.add_argument("--cues", nargs="+", choices=CUES, default=list(CUES), help="cues fused by the rank mean")
    g.add_argument("--tile", type=int, default=TILE, help="scoring tile size (px, multiple of 16)")
    g.add_argument("--stride", type=int, default=STRIDE, help="scoring tile stride (px, multiple of 16)")
    g.add_argument("--window-size", type=int, default=512, help="window mode: window size (px)")
    g.add_argument("--window-overlap", type=int, default=128, help="window mode: minimum window overlap (px)")
    g.add_argument("--crop-max-side", type=int, default=1024, help="crop mode: largest crop side before margins (px)")
    g.add_argument("--crop-overlap", type=int, default=64, help="crop mode: overlap between crops of one group (px)")
    g.add_argument("--tie-key", default=None, help="seed key of the tie-break jitter (default: the clip name)")
    g.add_argument("--no-tie-jitter", action="store_true", help="break ties at the K cut by region index instead")

    g = ap.add_argument_group("cues")
    g.add_argument("--fd-weights", default="weights/fd_gate.pt", help="FD network weights")
    g.add_argument("--fd-cell", type=int, default=4, help="FD cell size (px)")
    g.add_argument("--fd-history", type=int, default=7, help="FD accumulation length: current + previous frames")
    g.add_argument("--fd-device", default=None, help="FD device (default: cuda:0 if available, else cpu)")

    g = ap.add_argument_group("detectors (packed configs; see script/detector.py)")
    for role in ROLES:
        g.add_argument(f"--{role}-detector", default=DEFAULT_CONFIGS[role] and str(DEFAULT_CONFIGS[role]),
                       help=f"{role} detector config" + (" (default: reuse the coarse-pass boxes)" if role == "global" else ""))
        g.add_argument(f"--{role}-weights", default=None,
                       help=f"{role} detector weights (default: the 'weights' entry of its config"
                            + (", i.e. the coarse-pass weights)" if role == "crop" else ")"))
    g.add_argument("--detector-opt", nargs="*", default=[], metavar="ROLE.KEY=VALUE",
                   help="override any detector config key, e.g. window.max_det=200 coarse.half=true")

    g = ap.add_argument_group("merge")
    g.add_argument("--merge-conf", type=float, default=None,
                   help="min score of a global box to join the merge (default: 0.25 window mode, 0 crop mode)")
    g.add_argument("--softnms-sigma", type=float, default=0.5, help="Gaussian Soft-NMS sigma")
    g.add_argument("--softnms-floor", type=float, default=0.001, help="Soft-NMS score floor")
    g.add_argument("--max-det", type=int, default=300, help="boxes kept per frame (2000 for dense scenes, e.g. PANDA)")

    g = ap.add_argument_group("runtime")
    g.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0: all)")
    g.add_argument("--decoder-threads", type=int, default=4)
    g.add_argument("--cpu-threads", type=int, default=4, help="Torch and OpenCV threads")
    g.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up of all models")
    a = ap.parse_args(argv)

    if not 0.0 < a.keep <= 1.0:
        ap.error("--keep must be in (0, 1]")
    if a.tile % 16 or a.stride % 16 or a.stride > a.tile:
        ap.error("--tile and --stride must be multiples of 16, with stride <= tile")
    a.cues = list(dict.fromkeys(a.cues))
    outputs = [output_path(a.output, v) for v in a.input]
    if len(set(outputs)) != len(outputs):
        ap.error("several videos map to the same --output; use e.g. -o '{dir}/{stem}_det.txt'")
    if a.merge_conf is None:
        a.merge_conf = 0.25 if a.mode == "window" else 0.0
    return a


def output_path(template, video):
    v = Path(video).resolve()
    return Path(template.format(dir=v.parent, stem=v.stem))


def detector_overrides(a, role):
    out = {}
    for item in a.detector_opt:
        r, _, kv = item.partition(".")
        if r not in ROLES or not kv:
            raise SystemExit(f"--detector-opt '{item}': expected ROLE.KEY=VALUE with ROLE in {ROLES}")
        if r == role:
            out.update(parse_overrides([kv]))
    w = getattr(a, f"{role}_weights")
    if role == "crop" and w is None and a.crop_detector == str(DEFAULT_CONFIGS["crop"]):
        w = a.coarse_weights                                # crop mode runs the coarse-pass weights on its crops
    if w is not None:
        out["weights"] = w
    return out


class Pipeline:
    """Models shared by every clip; per-clip state (FD history, tie-break jitter) is rebuilt in run()."""

    def __init__(self, a):
        from detector import Detector
        self.a = a
        gated = a.keep < 1.0
        self.use = {c: (c in a.cues and gated) for c in CUES}
        # The coarse pass also feeds the merge, so it runs for exhaustive windows too when it is a chosen cue.
        self.use["coarse"] = "coarse" in a.cues
        role = "window" if a.mode == "window" else "crop"
        try:
            self.coarse = Detector(a.coarse_detector, **detector_overrides(a, "coarse")) if self.use["coarse"] else None
            self.region = Detector(getattr(a, f"{role}_detector"), **detector_overrides(a, role))
            self.glob = Detector(a.global_detector, **detector_overrides(a, "global")) if a.global_detector else None
        except FileNotFoundError as e:
            raise SystemExit(f"{e}\nPlace the weights at the default paths under weights/ or pass --coarse-weights, "
                             f"--{role}-weights (see README, Training).")
        import torch
        self.device = a.fd_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.use["fd"]:
            a.fd_weights = str(resolve_path(a.fd_weights))
            if not Path(a.fd_weights).is_file():
                raise SystemExit(f"FD weights not found: {a.fd_weights} (train them with script/train_fd.py)")

    def run(self, video):
        from crop_mode import CropMode
        from decode import open_video
        from score import CoarseCue, FDCue, rank_mean
        from window_mode import WindowMode

        a, use = self.a, self.use
        clip = Path(video).stem
        out = output_path(a.output, video)
        out.parent.mkdir(parents=True, exist_ok=True)
        sync = cuda_sync()
        clock = StageClock(sync)

        def reader():
            return open_video(video, bitcost=use["bc"], tile=a.tile, stride=a.stride, threads=a.decoder_threads)

        try:
            with reader() as probe:                         # frame size and a real frame for the warm-up
                first = next(probe)
        except (ValueError, StopIteration) as e:
            raise SystemExit(f"{str(e) or f'{video}: no frames'}\nRe-encode it, e.g. ffmpeg -i {video} -c:v libx264 -crf 22 "
                             f"-g 30 -bf 0 -pix_fmt yuv420p out.mp4")
        W, H = first.width, first.height
        origins = tile_origins(W, H, a.tile, a.stride)
        key = None if a.no_tie_jitter else (a.tie_key or clip)
        common = dict(keep=a.keep, tile=a.tile, merge_conf=a.merge_conf, sigma=a.softnms_sigma, floor=a.softnms_floor,
                      max_det=a.max_det, tie_key=key)
        if a.mode == "window":
            mode = WindowMode(self.region, W, H, origins, size=a.window_size, overlap=a.window_overlap, **common)
        else:
            mode = CropMode(self.region, W, H, origins, max_side=a.crop_max_side, overlap=a.crop_overlap, **common)
        coarse = CoarseCue(self.coarse, origins, a.tile) if use["coarse"] else None
        fd = FDCue(a.fd_weights, W, H, origins, self.device, a.tile, a.fd_cell, a.fd_history) if use["fd"] else None
        if not a.no_warmup:
            self.warmup(first, mode, fd)
        if fd is not None:
            fd.reset()                                      # the warm-up never feeds FD, but a clip always starts clean
        k = n_keep(a.keep, mode.n_regions)
        print(f"[{clip}] {W}x{H}, {len(origins)} tiles, {mode.n_regions} {a.mode} candidates, keep {k}, "
              f"cues {'+'.join(c for c in CUES if use[c]) or 'none (exhaustive)'}", flush=True)

        results, regions, stage_ms = [], [], []
        wall0 = time.perf_counter()
        with reader() as frames:
            while not a.max_frames or len(results) < a.max_frames:
                clock.start("decode")                       # decode, BC tile scores and the BGR conversion
                frame = next(frames, None)
                if frame is None:
                    clock.pop()
                    break
                bgr = frame.bgr
                cue_scores = []
                gb, gs = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
                if coarse is not None:
                    clock.start("coarse")
                    g_c = coarse(bgr)
                    gb, gs = coarse.boxes, coarse.scores
                    if a.keep < 1.0:
                        cue_scores.append(g_c)
                if fd is not None:
                    clock.start("fd")
                    cue_scores.append(fd(frame.y))
                if use["bc"]:
                    cue_scores.append(frame.bc)
                clock.start("select")
                if a.keep < 1.0:
                    fused = rank_mean(*cue_scores)
                    run = mode.select(fused)
                else:
                    fused = np.zeros(len(origins))
                    run = np.ones(mode.n_regions, bool)
                clock.start("build")
                idx, images = mode.regions(bgr, run)
                clock.start("detect")
                if a.mode == "window":
                    boxes, scores, win = mode.detect(idx, images)
                else:
                    boxes, scores = mode.detect(idx, images)
                if self.glob is not None:
                    clock.start("global")
                    gb, gs = self.glob([bgr])[0]
                clock.start("merge")
                if a.mode == "window":
                    boxes, scores = mode.merge(boxes, scores, win, run, gb, gs)
                else:
                    boxes, scores = mode.merge(boxes, scores, gb, gs)
                clock.stop()
                results.append((frame.index, boxes, scores))
                stage_ms.append(clock.pop())
                if a.save_regions:
                    regions.append((frame.index, np.asarray(fused, np.float32), run.copy(),
                                    np.asarray(idx, np.int32).reshape(-1, 4) if a.mode == "crop" else None))
        if sync:
            sync()
        wall = time.perf_counter() - wall0
        n = len(results)
        if not n:
            raise SystemExit(f"[{clip}] no frames decoded")

        with open(out, "w") as f:
            for index, boxes, scores in results:
                write_mot_dets(f, index, boxes, scores)
        if a.save_regions:
            # fused: (frames, tiles) fused tile scores; kept: (frames, candidates) kept windows (window mode) or tiles
            # (crop mode); crop mode adds crops (m, 4) = left, top, right, bottom with crop_frame (m,).
            extra = {"windows": mode.windows} if a.mode == "window" else {
                "crops": np.concatenate([r[3] for r in regions]),
                "crop_frame": np.concatenate([np.full(len(r[3]), r[0], np.int32) for r in regions])}
            np.savez_compressed(out.with_name(out.stem + "_regions.npz"), frame=np.array([r[0] for r in regions], np.int32),
                                fused=np.stack([r[1] for r in regions]), kept=np.stack([r[2] for r in regions]),
                                origins=np.array(origins, np.int32), **extra)
        stages = sorted({k for s in stage_ms for k in s})
        summary = dict(
            clip=clip, video=str(Path(video).resolve()), width=W, height=H, frames=n, wall_seconds=round(wall, 3),
            fps=round(n / wall, 3), mean_stage_ms={k: round(float(np.mean([s.get(k, 0.0) for s in stage_ms])), 3) for k in stages},
            mode=a.mode, keep=a.keep, kept_per_frame=k, candidates=mode.n_regions, tiles=len(origins),
            cues=[c for c in CUES if use[c]], tie_key=key,
            settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items() if k != "input"},
            detectors={r: d.describe() for r, d in (("coarse", self.coarse), ("region", self.region), ("global", self.glob)) if d},
            platform=self.platform())
        if a.summary:
            out.with_name(out.stem + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[{clip}] {n} frames in {wall:.1f} s = {n / wall:.2f} FPS -> {out}", flush=True)
        return summary

    def warmup(self, frame, mode, fd):
        """Untimed warm-up of every model at the input shapes and batch sizes of this clip; no cue state changes."""
        bgr = frame.bgr
        for det in (self.coarse, self.glob):
            if det is not None:
                det.warmup([((bgr.shape[0], bgr.shape[1]), [1])])
        k = n_keep(self.a.keep, mode.n_regions)
        if self.a.mode == "window":
            b = self.region.batch
            sizes = [b] * (k // b) + ([k % b] if k % b else [])
            self.region.warmup([((self.a.window_size, self.a.window_size), sizes)])
        else:
            self.region.warmup([((self.a.crop_max_side, self.a.crop_max_side), range(1, self.region.batch + 1))])
        if fd is not None:
            fd.warmup()

    @staticmethod
    def platform():
        info = dict(python=platform.python_version(), numpy=np.__version__)
        try:
            import torch
            info.update(torch=torch.__version__, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
        except ImportError:
            pass
        return info


def main(argv=None):
    a = parse_args(argv)
    import cv2
    import torch
    torch.set_num_threads(a.cpu_threads)
    cv2.setNumThreads(a.cpu_threads)
    torch.backends.cudnn.benchmark = True
    pipe = Pipeline(a)
    return [pipe.run(video) for video in a.input]


if __name__ == "__main__":
    main()

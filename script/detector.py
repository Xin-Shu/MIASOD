"""Detector unit: one interface over any detector family, built from a packed configuration.

A packed configuration is a YAML or JSON mapping. `executor` names the module that runs the model:
    executor: yolov11            -> script/detectors/yolov11/executor.py
    executor: dino               -> script/detectors/dino/executor.py
    executor: my_pkg.my_module   -> any importable module
    executor: /path/to/file.py   -> a module file outside this repository
That module defines a class `Executor(cfg)` with
    DEFAULTS : dict   default values of its keys (merged under the given config)
    REQUIRED : tuple  keys the config must provide
    PATHS    : tuple  keys holding file paths (resolved against the working directory, the config file and the repo)
    predict(images) -> [(boxes_xyxy (n, 4), scores (n,)), ...]   one entry per image, image (pixel) coordinates
The unit adds batching (`batch`), checks every output and returns boxes as xywh float32.

    det = Detector("script/detectors/yolov11/window.yaml", weights="weights/window.pt")
    (boxes, scores), = det([bgr])
"""
import importlib
import importlib.util
from pathlib import Path

import numpy as np

from util import SCRIPT, load_config, resolve_path, xyxy_to_xywh

_CUDA_READY = False


def _cuda_init():
    """Initialise CUDA before any detector library: ultralytics rewrites CUDA_VISIBLE_DEVICES when it is first."""
    global _CUDA_READY
    if not _CUDA_READY:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.init()
        except ImportError:
            pass
        _CUDA_READY = True


def load_executor(name):
    """Executor class from a folder name under script/detectors, a dotted module path or a .py file."""
    if name.endswith(".py"):
        path = resolve_path(name)
        spec = importlib.util.spec_from_file_location(f"executor_{path.stem}", path)
        if spec is None:
            raise ImportError(f"cannot load executor file {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif (SCRIPT / "detectors" / name / "executor.py").is_file():
        module = importlib.import_module(f"detectors.{name}.executor")
    else:
        module = importlib.import_module(name)
    if not hasattr(module, "Executor"):
        raise ImportError(f"executor module '{name}' defines no Executor class")
    return module.Executor


def pack(config, **overrides):
    """Resolve a packed configuration: file or dict, plus overrides (None values are ignored)."""
    cfg = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if "executor" not in cfg:
        raise KeyError("detector config needs an 'executor' entry")
    executor = load_executor(str(cfg["executor"]))
    full = dict(getattr(executor, "DEFAULTS", {}))
    full.update(cfg)
    missing = [k for k in getattr(executor, "REQUIRED", ()) if full.get(k) is None]
    if missing:
        raise KeyError(f"detector config ({cfg.get('_path', 'dict')}) misses {missing}")
    base = Path(full["_path"]).parent if "_path" in full else None
    for k in getattr(executor, "PATHS", ()):
        if full.get(k) is not None:
            p = resolve_path(full[k], base)
            if not p.exists():
                raise FileNotFoundError(f"detector '{k}' not found: {full[k]}")
            full[k] = str(p)
    return executor, full


class Detector:
    def __init__(self, config, **overrides):
        _cuda_init()
        executor, self.cfg = pack(config, **overrides)
        self.batch = int(self.cfg.get("batch", 1))
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        self.executor = executor(self.cfg)
        self.name = self.cfg.get("name") or Path(self.cfg.get("_path", str(self.cfg["executor"]))).stem

    def __call__(self, images):
        """Detections of every image: [(boxes xywh float32 (n, 4), scores float32 (n,)), ...]."""
        out = []
        for i in range(0, len(images), self.batch):
            chunk = images[i:i + self.batch]
            res = self.executor.predict(chunk)
            if len(res) != len(chunk):
                raise RuntimeError(f"{self.name}: {len(res)} results for {len(chunk)} images")
            for boxes, scores in res:
                boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
                scores = np.asarray(scores, np.float32).reshape(-1)
                if len(boxes) != len(scores) or not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
                    raise RuntimeError(f"{self.name}: malformed detections")
                out.append((xyxy_to_xywh(boxes), scores))
        return out

    def warmup(self, shapes, repeats=2):
        """Run every given input shape (h, w) at every batch size it will see, so timing excludes set-up."""
        for (h, w), sizes in shapes:
            for bs in sorted(set(min(s, self.batch) for s in sizes)):
                dummy = [np.zeros((h, w, 3), np.uint8)] * bs
                for _ in range(repeats):
                    self.executor.predict(dummy)

    def describe(self):
        return {k: v for k, v in self.cfg.items() if not k.startswith("_")}

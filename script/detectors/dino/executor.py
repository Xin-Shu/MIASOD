"""DINO executor (MMDetection 3.x). Images are BGR uint8 arrays; boxes come back in image coordinates.

`resize` and `pad` override the Resize scale and Pad size of the config's test pipeline (e.g. 512 x 512 windows at
native resolution); per image, the `max_det` highest-scoring boxes with score >= `score_thr` are returned.
"""
import numpy as np


class Executor:
    DEFAULTS = dict(device="cuda:0", score_thr=0.01, max_det=300, resize=None, pad=None)
    REQUIRED = ("config", "weights")
    PATHS = ("config", "weights")

    def __init__(self, cfg):
        import torch
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg
        from mmengine.dataset import Compose
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.model = init_detector(cfg["config"], cfg["weights"], device=cfg["device"])
        steps = [s for s in get_test_pipeline_cfg(self.model.cfg.copy()) if "LoadAnnotations" not in s["type"]]
        steps[0].type = "mmdet.LoadImageFromNDArray"
        for s in steps:
            if cfg["resize"] is not None and s["type"] in ("Resize", "mmdet.Resize"):
                s["scale"] = tuple(cfg["resize"])
            if cfg["pad"] is not None and s["type"] in ("Pad", "mmdet.Pad"):
                s["size"] = tuple(cfg["pad"])
        self.pipeline = Compose(steps)
        self.thr, self.max_det = cfg["score_thr"], cfg["max_det"]

    def predict(self, images):
        data = [self.pipeline(dict(img=im, img_id=0)) for im in images]
        batch = dict(inputs=[d["inputs"] for d in data], data_samples=[d["data_samples"] for d in data])
        with self.torch.no_grad():
            results = self.model.test_step(batch)
        out = []
        for r in results:
            s = r.pred_instances.scores.cpu().numpy()
            b = r.pred_instances.bboxes.cpu().numpy()
            k = np.argsort(-s, kind="stable")[:self.max_det]
            k = k[s[k] >= self.thr]
            out.append((b[k], s[k]))
        return out

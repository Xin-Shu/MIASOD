"""YOLOv11 executor (Ultralytics). Images are BGR uint8 arrays; boxes come back in image coordinates."""


class Executor:
    DEFAULTS = dict(imgsz=640, conf=0.25, iou=0.7, max_det=300, half=False, device=0, classes=None)
    REQUIRED = ("weights",)
    PATHS = ("weights",)

    def __init__(self, cfg):
        from ultralytics import YOLO
        self.model = YOLO(cfg["weights"])
        self.kw = dict(imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"], max_det=cfg["max_det"], half=cfg["half"],
                       device=cfg["device"], classes=cfg["classes"], verbose=False)

    def predict(self, images):
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) for r in self.model.predict(list(images), **self.kw)]

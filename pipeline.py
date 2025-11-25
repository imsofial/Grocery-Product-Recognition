from typing import List, Tuple, Dict, Any
from collections import Counter
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from recipe_api import get_recipes_by_ingredient

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False


label2id = {
    'apple/fresh': 0, 'apple/rotten': 1,
    'banana/fresh': 2, 'banana/rotten': 3,
    'orange/fresh': 4, 'orange/rotten': 5,
    'potato/fresh': 6, 'potato/rotten': 7,
    'tomato/fresh': 8, 'tomato/rotten': 9
}
id2label = {v: k for k, v in label2id.items()}


class Detector:
    def __init__(self, model_path="yolov8n.pt"):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image, conf=0.25, iou=0.45, imgsz=640):
        results = self.model.predict(source=np.array(image), conf=conf, iou=iou, imgsz=imgsz)
        out = []
        for r in results:
            boxes = getattr(r, "boxes", [])
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().reshape(-1)
                xmin, ymin, xmax, ymax = [int(x) for x in xyxy[:4]]
                conf_score = float(box.conf.cpu().numpy())
                cls_id = int(box.cls.cpu().numpy())
                out.append((xmin, ymin, xmax, ymax, conf_score, cls_id))
        return out



def crop_boxes_from_image(image, boxes, pad=6):
    w, h = image.size
    crops = []
    for (xmin, ymin, xmax, ymax) in boxes:
        x0 = max(0, xmin - pad)
        y0 = max(0, ymin - pad)
        x1 = min(w, xmax + pad)
        y1 = min(h, ymax + pad)
        crops.append(image.crop((x0, y0, x1, y1)))
    return crops


def load_finetuned_resnet(num_classes, device="cpu"):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state = torch.load("finetuned_models/resnet50_fruits.pth",map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]

    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)

    model.to(device).eval()
    return model


def get_resnet_transform(size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# -------- GLOBAL MODELS ----------
RESNET_MODEL = load_finetuned_resnet(10)
TRANSFORM = get_resnet_transform()
DEVICE = "cpu"
# ---------------------------------------------------------------

@torch.no_grad()
def predict_batch_with_model(crops, topk=1):
    if len(crops) == 0:
        return np.zeros((0, topk)), np.zeros((0, topk)), None

    imgs = [TRANSFORM(c.convert("RGB")) for c in crops]
    batch = torch.stack(imgs).to(DEVICE)

    logits = RESNET_MODEL(batch)
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    topk_idx = np.argsort(-probs, axis=1)[:, :topk]
    topk_scores = probs[np.arange(len(probs))[:, None], topk_idx]

    return topk_idx, topk_scores, probs


def aggregate_predictions(topk_idx, topk_scores):
    per = []
    counts = Counter()

    for i in range(len(topk_idx)):
        idx = int(topk_idx[i][0])
        score = float(topk_scores[i][0])
        label = id2label[idx]

        per.append({"label": label, "score": score})
        counts[label] += 1

    return counts, per


def _read_recipes_json_if_exists():
    p = Path("recipes.json")
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except:
        return []

def filter_overlapping_boxes(boxes, iou_threshold=0.5):
    if len(boxes) <= 1:
        return boxes

    # boxes: [(xmin, ymin, xmax, ymax, conf, cls_id), ...]
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    filtered = []

    for box in boxes:
        keep = True
        for f in filtered:
            # compute IoU
            xA = max(box[0], f[0])
            yA = max(box[1], f[1])
            xB = min(box[2], f[2])
            yB = min(box[3], f[3])

            inter = max(0, xB - xA) * max(0, yB - yA)
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            f_area = (f[2] - f[0]) * (f[3] - f[1])
            union = box_area + f_area - inter

            iou = inter / union if union > 0 else 0
            if iou > iou_threshold:
                keep = False
                break

        if keep:
            filtered.append(box)

    return filtered


def full_pipeline(image_path, category="Dessert", topk=1):
    img = Image.open(image_path).convert("RGB")

    DETECTOR = Detector()
    dets = DETECTOR.detect(img)
    dets = filter_overlapping_boxes(dets, iou_threshold=0.6)
    if len(dets) == 0:
        return {"detections": [], "recipes": []}

    boxes = [(x0, y0, x1, y1) for x0, y0, x1, y1, _, _ in dets]
    crops = crop_boxes_from_image(img, boxes)

    topk_idx, topk_scores, _ = predict_batch_with_model(crops, topk)
    counts, per = aggregate_predictions(topk_idx, topk_scores)

    ingredients = sorted({lbl.split("/")[0] for lbl in counts})

    recipes = {}
    for ingr in ingredients:
        msg = get_recipes_by_ingredient(ingr, category_filter=category)
        existing = _read_recipes_json_if_exists()
        recipes[ingr] = existing or msg

    return {"detections": per, "counts": counts, "recipes": recipes}

def detect_and_classify_image(image: Image.Image, topk: int = 1) -> Dict[str, Any]:
    """
    Run YOLO detection + ResNet classification on a PIL image.
    Returns:
      {
        "detections": [
            {"label": "apple/fresh", "score": 0.97, "box": (x0, y0, x1, y1), "det_conf": 0.88},
            ...
        ],
        "counts": Counter({"apple/fresh": 2, ...}),
        "ingredients": ["apple", "banana", ...]  # unique fruit names
      }
    """
    dets = DETECTOR.detect(image)
    dets = filter_overlapping_boxes(dets, iou_threshold=0.6)

    if len(dets) == 0:
        return {"detections": [], "counts": Counter(), "ingredients": []}

    # Get pure boxes (without conf/class) for cropping
    boxes = [(x0, y0, x1, y1) for (x0, y0, x1, y1, _, _) in dets]
    crops = crop_boxes_from_image(image, boxes)

    topk_idx, topk_scores, _ = predict_batch_with_model(crops, topk)
    counts, per = aggregate_predictions(topk_idx, topk_scores)

    # Attach box + detection confidence to each prediction
    for det_dict, (x0, y0, x1, y1, det_conf, _cls_id) in zip(per, dets):
        det_dict["box"] = (x0, y0, x1, y1)
        det_dict["det_conf"] = float(det_conf)

    # Fruit names only (apple, banana, ...)
    ingredients = sorted({lbl.split("/")[0] for lbl in counts})

    return {
        "detections": per,
        "counts": counts,
        "ingredients": ingredients,
    }

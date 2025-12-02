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


# ---------- Label mappings ----------
label2id = {
    "apple/fresh": 0,
    "apple/rotten": 1,
    "banana/fresh": 2,
    "banana/rotten": 3,
    "orange/fresh": 4,
    "orange/rotten": 5,
    "potato/fresh": 6,
    "potato/rotten": 7,
    "tomato/fresh": 8,
    "tomato/rotten": 9,
}
id2label = {v: k for k, v in label2id.items()}


# ---------- Utility: cropping ----------
def crop_boxes_from_image(image: Image.Image, boxes, pad: int = 6):
    w, h = image.size
    crops = []
    for (xmin, ymin, xmax, ymax) in boxes:
        x0 = max(0, xmin - pad)
        y0 = max(0, ymin - pad)
        x1 = min(w, xmax + pad)
        y1 = min(h, ymax + pad)
        crops.append(image.crop((x0, y0, x1, y1)))
    return crops


# ---------- Classification model (ResNet50) ----------
def load_finetuned_resnet(num_classes: int, device: str = "cpu"):
    """
    Load your finetuned resnet50_fruits.pth checkpoint.
    Expected path: finetuned_models/resnet50_fruits.pth
    """
    # Use weights=None to avoid deprecation warnings
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    ckpt_path = Path("finetuned_models/resnet50_fruits.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path.resolve()}. "
            "Make sure finetuned_models/resnet50_fruits.pth is present."
        )

    state = torch.load(str(ckpt_path), map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]

    # Remove 'module.' prefix if model was trained with DataParallel
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)

    model.to(device).eval()
    return model


def get_resnet_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


# ---------- Globals ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESNET_MODEL = load_finetuned_resnet(num_classes=10, device=DEVICE)
TRANSFORM = get_resnet_transform()


# ---------- Classification helpers ----------
@torch.no_grad()
def predict_batch_with_model(crops: List[Image.Image], topk: int = 1):
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
    """
    Convert indices + scores into human-readable labels
    and a Counter of label occurrences.
    """
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
    except Exception:
        return []


# ---------- Dummy "detector" (no YOLO, no cv2) ----------
def dummy_detect_full_image(image: Image.Image):
    """
    Fallback detector for environments where YOLO/cv2 cannot run.
    Treat the whole image as a single object.
    Returns: [(xmin, ymin, xmax, ymax, conf, cls_id)]
    """
    w, h = image.size
    xmin, ymin, xmax, ymax = 0, 0, w, h
    conf = 1.0
    cls_id = 0  # dummy class id (not used downstream)
    return [(xmin, ymin, xmax, ymax, conf, cls_id)]


def filter_overlapping_boxes(boxes, iou_threshold: float = 0.5):
    """
    Simple NMS: keep highest-confidence boxes, drop others if IoU > threshold.
    boxes: [(xmin, ymin, xmax, ymax, conf, cls_id), ...]
    """
    if len(boxes) <= 1:
        return boxes

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

            iou = inter / union if union > 0 else 0.0
            if iou > iou_threshold:
                keep = False
                break

        if keep:
            filtered.append(box)

    return filtered


# ---------- Legacy full pipeline (kept for completeness) ----------
def full_pipeline(image_path, category: str = "Dessert", topk: int = 1):
    """
    Full pipeline taking a path, including recipe-fetching.
    Uses dummy detection (whole image as one object).
    """
    img = Image.open(image_path).convert("RGB")

    dets = dummy_detect_full_image(img)
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


# ---------- Main function used by app.py ----------
def detect_and_classify_image(image: Image.Image, topk: int = 1) -> Dict[str, Any]:
    """
    Run "detection" + ResNet classification on a PIL image.

    On Streamlit Cloud, detection is a dummy: whole image is one box.
    Returns:
      {
        "detections": [
            {
              "label": "apple/fresh",
              "score": 0.97,
              "box": (x0, y0, x1, y1),
              "det_conf": 1.0
            },
            ...
        ],
        "counts": Counter({"apple/fresh": 1, ...}),
        "ingredients": ["apple", ...]
      }
    """
    # Dummy detection (no YOLO)
    dets = dummy_detect_full_image(image)
    dets = filter_overlapping_boxes(dets, iou_threshold=0.6)

    if len(dets) == 0:
        return {"detections": [], "counts": Counter(), "ingredients": []}

    # Boxes only for cropping
    boxes = [(x0, y0, x1, y1) for (x0, y0, x1, y1, _, _) in dets]
    crops = crop_boxes_from_image(image, boxes)

    topk_idx, topk_scores, _ = predict_batch_with_model(crops, topk)
    counts, per = aggregate_predictions(topk_idx, topk_scores)

    # Attach box + dummy detection confidence
    for det_dict, (x0, y0, x1, y1, det_conf, _cls_id) in zip(per, dets):
        det_dict["box"] = (x0, y0, x1, y1)
        det_dict["det_conf"] = float(det_conf)

    ingredients = sorted({lbl.split("/")[0] for lbl in counts})

    return {
        "detections": per,
        "counts": counts,
        "ingredients": ingredients,
    }

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

# optional: ultralytics YOLOv8
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
    def __init__(self, model_path: str = "yolov8n.pt"):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("ultralytics not installed. pip install ultralytics or replace Detector with your detector.")
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640):
        """
        Returns list of detections where each detection is:
        (xmin, ymin, xmax, ymax, conf_score, class_id)
        Coordinates are integers in image pixel space.
        """
        results = self.model.predict(source=np.array(image), conf=conf, iou=iou, imgsz=imgsz)
        dets: List[Tuple[int, int, int, int, float, int]] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().reshape(-1)
                xmin, ymin, xmax, ymax = [int(float(x)) for x in xyxy[:4]]
                conf_score = float(box.conf.cpu().numpy())
                cls_id = int(box.cls.cpu().numpy())
                dets.append((xmin, ymin, xmax, ymax, conf_score, cls_id))
        return dets


def crop_boxes_from_image(image: Image.Image, boxes: List[Tuple[int, int, int, int]], pad: int = 5) -> List[Image.Image]:
    w, h = image.size
    crops: List[Image.Image] = []
    for (xmin, ymin, xmax, ymax) in boxes:
        x0 = max(0, xmin - pad)
        y0 = max(0, ymin - pad)
        x1 = min(w, xmax + pad)
        y1 = min(h, ymax + pad)
        crops.append(image.crop((x0, y0, x1, y1)))
    return crops


def load_finetuned_resnet(ckpt_path: str, num_classes: int, device: str = "cpu") -> torch.nn.Module:
    device = torch.device(device)
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state = torch.load(ckpt_path, map_location=device)
    # Accept multiple possible checkpoint formats
    if isinstance(state, dict):
        # If it's a dict with nested state_dict
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            sd = state["state_dict"]
        elif "model_state" in state and isinstance(state["model_state"], dict):
            sd = state["model_state"]
        else:
            sd = state
        # strip possible "module." prefixes
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if k.startswith("module."):
                nk = k[len("module."):]
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    else:
        # object with state_dict method
        try:
            model.load_state_dict(state.state_dict(), strict=False)
        except Exception:
            raise RuntimeError("Unsupported checkpoint format for resnet checkpoint.")
    model.to(device).eval()
    return model


def get_resnet_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def predict_batch_with_model(model: torch.nn.Module, crops: List[Image.Image], transform: transforms.Compose, device: str = "cpu", topk: int = 1):
    device = torch.device(device)
    tensors = [transform(img.convert("RGB")) for img in crops]
    if len(tensors) == 0:
        return np.zeros((0, topk), dtype=int), np.zeros((0, topk), dtype=float), np.zeros((0, model.fc.out_features), dtype=float)
    batch = torch.stack(tensors, dim=0).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    topk_idx = np.argsort(-probs, axis=1)[:, :topk]
    topk_scores = np.take_along_axis(probs, topk_idx, axis=1)
    return topk_idx, topk_scores, probs


def aggregate_predictions(topk_idx: np.ndarray, topk_scores: np.ndarray, id2label_map: Dict[int, str]):
    per: List[Dict[str, Any]] = []
    counts = Counter()
    for i in range(topk_idx.shape[0]):
        idx0 = int(topk_idx[i, 0])
        score0 = float(topk_scores[i, 0])
        label = id2label_map[idx0]
        per.append({"label": label, "score": score0, "index": idx0})
        counts[label] += 1
    return counts, per


def _read_recipes_json_if_exists() -> List[Dict[str, Any]]:
    p = Path("recipes.json")
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def full_pipeline(
    image_path: str,
    detector: Detector,
    resnet_model: torch.nn.Module,
    transform: transforms.Compose,
    id2label_map: Dict[int, str],
    device: str = "cpu",
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.45,
    topk: int = 1,
    recipe_category: str = "Dessert"
) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    dets = detector.detect(img, conf=yolo_conf, iou=yolo_iou)
    if len(dets) == 0:
        return {"detections": [], "recipes": []}

    boxes = [(xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax, conf, cls in dets]
    crops = crop_boxes_from_image(img, boxes, pad=6)

    topk_idx, topk_scores, probs = predict_batch_with_model(resnet_model, crops, transform, device=device, topk=topk)
    counts, per = aggregate_predictions(topk_idx, topk_scores, id2label_map)

    # prepare recipe queries: unique fruit names (no duplicates), lowercased
    ingredients = []
    for lbl in counts.keys():
        name = lbl.split('/')[0].strip().lower()
        if name not in ingredients:
            ingredients.append(name)

    recipes_for_ingredients: Dict[str, Any] = {}
    for ingredient in ingredients:
        msg = get_recipes_by_ingredient(ingredient, category_filter=recipe_category)
        json_recipes = _read_recipes_json_if_exists()
        if json_recipes:
            recipes_for_ingredients[ingredient] = json_recipes
        else:
            recipes_for_ingredients[ingredient] = {"message": msg}

    # return structured result
    return {
        "detections": per,             # per-detection list with label and score
        "counts": dict(counts),        # counts per joint label
        "recipes": recipes_for_ingredients
    }


if __name__ == "__main__":
    # usage example (loads models once)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = Detector(model_path="yolov8n.pt")  # put correct path if needed
    resnet = load_finetuned_resnet("finetuned_models/resnet50_fruits.pth", num_classes=10, device=device)
    transform = get_resnet_transform(224)

    out = full_pipeline(
        image_path="dataset_prepared/test/apple/fresh/FreshApple (17).jpg",
        detector=detector,
        resnet_model=resnet,
        transform=transform,
        id2label_map=id2label,
        device=device,
        yolo_conf=0.25,
        yolo_iou=0.45,
        topk=1,
        recipe_category="Dessert"
    )
    print(out)

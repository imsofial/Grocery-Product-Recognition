import os, json, time, csv, itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

DATA_DIR = r"C:\Users\Kistanov\Desktop\Grocery-Product-Recognition\dataset_prepared\test"
CKPT_PATH = r"C:\Users\Kistanov\Desktop\Grocery-Product-Recognition\finetuned_models\resnet50_fruits.pth"
BATCH_SIZE = 64
NUM_WORKERS = 2
OUT_DIR = "eval_outputs"

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

label2id = {'apple/fresh': 0, 'apple/rotten': 1, 'banana/fresh': 2, 'banana/rotten': 3,
            'orange/fresh': 4, 'orange/rotten': 5, 'potato/fresh': 6, 'potato/rotten': 7,
            'tomato/fresh': 8, 'tomato/rotten': 9}

class FruitConditionDataset(Dataset):
    def __init__(self, root_dir, transform=None, label2id=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_names = []

        for fruit in os.listdir(root_dir):
            fruit_path = os.path.join(root_dir, fruit)
            if not os.path.isdir(fruit_path):
                continue
            for condition in os.listdir(fruit_path):
                condition_path = os.path.join(fruit_path, condition)
                if not os.path.isdir(condition_path):
                    continue
                label_name = f"{fruit}/{condition}"
                self.class_names.append(label_name)
                label_id = label2id[label_name]
                for img_name in os.listdir(condition_path):
                    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.samples.append((os.path.join(condition_path, img_name), label_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_model(num_classes, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval().to(device)
    P, H, Y = [], [], []
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(1)
        P.append(probs)
        H.append(preds)
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(H), np.concatenate(Y)

def plot_confusion(cm, names, out_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(names))
    plt.xticks(ticks, names, rotation=45, ha="right")
    plt.yticks(ticks, names)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        v = cm[i, j]
        if v > 0:
            plt.text(j, i, str(v), ha="center", va="center",
                     color="white" if v > thresh else "black", fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_report(y_true, y_pred, class_names, out_csv):
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for c in class_names:
            r = rep[c]
            w.writerow([c, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", int(r['support'])])
        w.writerow(["accuracy", "", "", f"{rep['accuracy']:.4f}", sum(int(rep[c]['support']) for c in class_names)])

if __name__ == "__main__":
    ds = FruitConditionDataset(DATA_DIR, transform=data_transforms, label2id=label2id)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(len(ds.class_names), CKPT_PATH)

    print(f"Evaluating on {len(ds)} images ({len(ds.class_names)} classes)...")
    start = time.time()
    probs, preds, targets = evaluate(model, dl, device)
    elapsed = time.time() - start

    out_dir = Path(OUT_DIR) / "resnet50_finetuned_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = accuracy_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    plot_confusion(cm, ds.class_names, out_dir / "confusion_matrix.png")
    save_report(targets, preds, ds.class_names, out_dir / "class_report.csv")

    np.save(out_dir / "probs.npy", probs)
    np.save(out_dir / "preds.npy", preds)
    np.save(out_dir / "targets.npy", targets)

    summary = {
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "num_samples": len(ds),
        "num_classes": len(ds.class_names),
        "classes": ds.class_names,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[ResNet50 fruits] test acc={acc:.4f} | n={len(ds)} | {elapsed:.1f}s -> {out_dir.resolve()}")

# 🥦 Grocery Product Recognition & Recipe Recommendation

## 📌 Overview

This project aims to build a system that can **recognize grocery products** from images (e.g., photos of a shopping bag, fridge, or pantry) and provide **smart recipe recommendations**.

The system goes beyond simple recognition — it can also **detect spoiled or expired products** and exclude them from suggestions. All valid items are added to a **virtual basket**, which serves as the basis for personalized recipe recommendations.

Additionally, if a recipe requires extra ingredients, the system can **suggest complementary products** to help users prepare complete meals.

---

## ✨ Features

- 🛒 **Grocery Recognition** – Detect and classify food products from photos.
- ❌ **Spoilage Detection** – Identify spoiled or expired items and remove them from the basket.
- 🧺 **Virtual Basket** – Store recognized products for recipe generation.
- 🍳 **Recipe Recommendation** – Suggest recipes that match available products.
- ➕ **Complementary Suggestions** – Recommend missing ingredients to complete a recipe.

---

## 💡 Use Cases

- ♻️ Reduce **food waste** by suggesting meals with soon-to-expire products.
- 🍲 Help users discover **new recipes** from everyday groceries.
- 🛍️ Make grocery shopping smarter by showing what else to buy for a chosen meal.

---

## 🛠️ Tech Stack (Planned)

- **Computer Vision:** Object detection + classification models (e.g., YOLO, EfficientNet, ResNet).
- **Spoilage Detection:** Image quality analysis + freshness classification.
- **Recipe Recommendation:** NLP-based recommendation system with product-to-recipe matching.
- **Backend:** FastAPI / Flask for serving models.
- **Database:** PostgreSQL for product and recipe storage.

---

## 🚀 Future Improvements

- 📱 Mobile app integration (scan fridge/bag in real-time).
- 🌍 Multilingual recipe recommendations.
- 🧠 Personalized suggestions based on dietary preferences and history.

---

## 📷 Example Workflow

1. Upload a photo of your fridge or shopping bag.
2. System recognizes the grocery items.
3. Spoiled items are filtered out.
4. A recipe is suggested based on available products.
5. Missing ingredients are recommended for shopping.

 ---

 ## 🧪 What works now

Cleaned and prepared dataset (duplicates removed, 70/15/15 split, 224×224).

Baseline evaluations:

Fruit identity (5-class)

EfficientNet-B0 — 97.57% test accuracy (n=1936)

ResNet50 — 98.45% test accuracy (n=1936)

Freshness (2-class) (harder task; needs more curation/augs)

EfficientNet-B0 — 44.94% test accuracy

ResNet50 — 50.10% test accuracy

EfficientNet-B3 (Fine-tuned) — 81.82% validation accuracy 

(Fruit recognition is basically solved on our distribution; freshness requires better data/augs and light fine-tuning.)


## Project Structure

```

Grocery-Product-Recognition/
│
├── dataset_prepared/        # Dataset ready for training
│   ├── train/               # Training set
│   │   ├── apple/           # Class: apples
│   │   │   ├── fresh/       # Subclass: fresh apples
│   │   │   └── rotten/      # Subclass: rotten apples
│   │   ├── banana/          # Class: bananas
│   │   ├── orange/          # ...
│   │   ├── potato/          
│   │   └── tomato/          
│   ├── val/                 # Validation set
│   └── test/                # Test set
│
├── raw_dataset/             # Raw (unprocessed) data - ignored for saving memory
│
├── eval_outputs/          # Saved metrics, confusion matrices, summaries (created after runs)
|
├── finetuned_models/       # Folder for model weights savings
|
├── count_images.py          # Script to count the number of images per class/subclass
├── prepare_dataset.py       # Script to preprocess and split the dataset
|
|
├── evaluation_scripts/
│   ├── eval_torchvision.py                 # 5-class FRUIT evaluation
│   ├── eval_freshness.py                   # 2-class FRESH vs ROTTEN evaluation
│   ├── evaluate_finetuned_efficientnet.py  # 10-class FRUIT & FRESHNESS evaluation for EfficientNet_b0 and EfficientNet_b3 models 
│   └── evaluate_finetuned_resnet.py        # 10-class FRUIT & FRESHNESS evaluation for ResNet50 model
│
├── train_scripts/
│   ├── train_linear_probe.py  # quick head-only training
│   ├── train_efficientnet.py  # train pipeline for efficientnet_b0 and efficientnet_b3
│   ├── resnet50_train.py      # train pipeline for resnet50 
|
│  
└── README.md                # Project documentation

````



## Scripts

### 1. `prepare_dataset.py`
Prepares the dataset for training:
- Reads raw images from `raw_dataset/`
- Splits them into `train/`, `val/`, and `test/`
- Organizes images into `fresh/` and `rotten/` subfolders for each class

**Run:**
```bash
python prepare_dataset.py
````

### 2. `count_images.py`

Counts the number of images in each class and subclass inside `dataset_prepared/`.

**Run:**

```bash
python count_images.py
```

Output example:

```
train
apple/fresh: 857 photos
apple/rotten: 868 photos
banana/fresh: 972 photos
banana/rotten: 906 photos
orange/fresh: 935 photos
orange/rotten: 917 photos
potato/fresh: 802 photos
potato/rotten: 956 photos
tomato/fresh: 868 photos
tomato/rotten: 912 photos

val
apple/fresh: 183 photos
apple/rotten: 186 photos
banana/fresh: 208 photos
banana/rotten: 194 photos
orange/fresh: 200 photos 
   ...
```

### 3. `resnet50_train.py` and `train_efficientnet.py`

Starts train loop with fine-tuning of EfficientNet_b0/EfficientNet_b3/ResNet50 model to classify fresh/rotten groccery products.

**Run:**

```bash
python train_scripts/resnet50_train.py
python train_scripts/train_efficientnet.py
```

---

## Dataset Notes

* All images should be placed in `raw_dataset/` before preprocessing.
* After running `prepare_dataset.py`, the `dataset_prepared/` folder will be automatically generated and ready for model training.

## 📊 Evaluation

###  A) Fruit Identity (5-class)

Predict which fruit it is:
`apple | banana | orange | potato | tomato`

```bash
# EfficientNet-B0
python scripts/eval_torchvision.py \
  --data dataset_prepared --split test --model efficientnet_b0 --outdir eval_outputs

# ResNet50
python scripts/eval_torchvision.py \
  --data dataset_prepared --split test --model resnet50 --outdir eval_outputs
```

**Outputs** (in `eval_outputs/<model>_test/`):

* `summary.json` — overall accuracy, number of samples, class names
* `class_report.csv` — per-class precision / recall / F1
* `confusion_matrix.png` — visual confusion matrix
* `preds.npy`, `probs.npy`, `targets.npy` — raw arrays for further analysis

**Latest results:**

| Model           | Test Accuracy | Samples |
| :-------------- | :-----------: | :-----: |
| EfficientNet-B0 |  **97.57 %**  |   1936  |
| ResNet50        |  **98.45 %**  |   1936  |

---

### B) Freshness (2-class)

Binary task: `fresh | rotten`
(Fruit identity ignored; label comes from second-level folder names.)

```bash
# EfficientNet-B0
python scripts/eval_freshness.py \
  --data dataset_prepared --split test --model efficientnet_b0 --outdir eval_outputs

# ResNet50
python scripts/eval_freshness.py \
  --data dataset_prepared --split test --model resnet50 --outdir eval_outputs
```

**Latest results:**

| Model           | Test Accuracy |
| :-------------- | :-----------: |
| EfficientNet-B0 |  **44.94 %**  |
| ResNet50        |  **50.10 %**  |

### C) Fruit Identity and Freshness (10-class)

Predict if fruit belong one of categories:
`apple/fresh | apple/rotten | banana/fresh | banana/rotten | orange/fresh | orange/rotten | potato/fresh | potato/rotten | tomato/fresh | tomato/rotten`

**Latest results:**
| Model           | Test Accuracy |
| :-------------- | :-----------: |
| EfficientNet-B0 |  **80.05 %**  |
| EfficientNet-B3 |  **83.51 %**  |
| ResNet50        |  **89.35 %**  |

---

### Quick “Linear Probe” (Train Head Only)

Use this when you have only ImageNet weights — it quickly trains just the classifier head, often boosting accuracy dramatically.

```bash
# Train head on fruit 5-class (freeze backbone)
python scripts/train_linear_probe.py \
  --data dataset_prepared --model efficientnet_b0 --epochs 5

# Then evaluate on test with the saved checkpoint
python scripts/eval_torchvision.py \
  --data dataset_prepared --split test --model efficientnet_b0 \
  --ckpt checkpoints/efficientnet_b0_linear_probe_best.pth
```

You can do the same for ResNet:

```bash
python scripts/train_linear_probe.py \
  --data dataset_prepared --model resnet50 --epochs 5
```

## 👥 Team

* **Ekaterina Akimenko** — Evaluation pipeline, binary freshness evaluation, reporting
* **Sofia Goryunova** — Model development / fine-tuning / models evaluation
* **Yasmina Mamadalieva** — Data sourcing/curation, docs

---






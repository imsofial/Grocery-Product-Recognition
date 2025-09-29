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
├── count_images.py          # Script to count the number of images per class/subclass
├── prepare_dataset.py       # Script to preprocess and split the dataset
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

---

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

---

## Dataset Notes

* All images should be placed in `raw_dataset/` before preprocessing.
* After running `prepare_dataset.py`, the `dataset_prepared/` folder will be automatically generated and ready for model training.






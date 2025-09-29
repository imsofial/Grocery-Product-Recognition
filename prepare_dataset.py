import os
import random
import hashlib
from PIL import Image
import shutil

INPUT_DIR = "raw_dataset"
OUTPUT_DIR = "dataset_prepared"
IMG_SIZE = (224, 224)
SPLIT = [0.7, 0.15, 0.15]

# Hashing for searching duplicates
def get_image_hash(path):
    with Image.open(path).convert("RGB") as img:
        img = img.resize((64, 64))   # for stabilizing
        return hashlib.md5(img.tobytes()).hexdigest()

# Saving images
def process_and_save(src_path, dst_path):
    img = Image.open(src_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img.save(dst_path)

def prepare_dataset():
    seen_hashes = set()
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    classes = os.listdir(INPUT_DIR)
    for cls in classes:  # apple, banana, potato..
        sub_classes = os.listdir(os.path.join(INPUT_DIR, cls))  # fresh, rotten
        for sub in sub_classes:
            src_folder = os.path.join(INPUT_DIR, cls, sub) # constructing input path
            images = os.listdir(src_folder)
            random.shuffle(images) # mix photos

            # duplicates filtering
            unique_images = []
            for f in images:
                src_path = os.path.join(src_folder, f)
                try:
                    h = get_image_hash(src_path)
                    if h not in seen_hashes: #checking for duplicates
                        seen_hashes.add(h)
                        unique_images.append(f)
                except Exception as e:
                    print(f"Error while preprocessing {src_path}: {e}")

            # splitting data
            n_total = len(unique_images)
            n_train = int(n_total * SPLIT[0])
            n_val = int(n_total * SPLIT[1])

            splits = {
                "train": unique_images[:n_train],
                "val": unique_images[n_train:n_train+n_val],
                "test": unique_images[n_train+n_val:]
            }
            # for each group test/train/val create the folder
            for split_name, files in splits.items():
                dst_folder = os.path.join(OUTPUT_DIR, split_name, cls, sub)
                os.makedirs(dst_folder, exist_ok=True)
                for f in files:
                    src_path = os.path.join(src_folder, f)
                    dst_path = os.path.join(dst_folder, f)
                    try:
                        # copy preprocessed data to the folder
                        process_and_save(src_path, dst_path)
                    except Exception as e:
                        print(f"Error with{src_path}: {e}")

if __name__ == "__main__":
    prepare_dataset()
    print("Done")


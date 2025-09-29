import os

base_dir = "dataset_prepared"

for split in ["train", "val", "test"]:
    split_path = os.path.join(base_dir, split)
    if not os.path.exists(split_path):
        continue
    print(f"\n{split}")
    for product in os.listdir(split_path):
        product_path = os.path.join(split_path, product)
        if os.path.isdir(product_path):
            for state in os.listdir(product_path):
                state_path = os.path.join(product_path, state)
                if os.path.isdir(state_path):
                    num_files = len([
                        f for f in os.listdir(state_path)
                        if os.path.isfile(os.path.join(state_path, f))
                    ])
                    print(f"{product}/{state}: {num_files} photos")

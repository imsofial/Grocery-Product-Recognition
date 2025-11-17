# main.py

import torch
from pipeline import full_pipeline


if __name__ == "__main__":

    image_path = "dataset_prepared/test/apple/fresh/FreshApple (17).jpg"

    result = full_pipeline(
        image_path=image_path,
        category="Dessert",
        topk=1
    )

    print(result)

# %%
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# %%
# !pip install ultralytics -q

# %%
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import os
import json

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

# %%
KARPATHY_JSON = "data/coco/dataset_coco.json"
TRAIN_DIR = "data/coco/train2014"
VAL_DIR = "data/coco/val2014"
OUTPUT_DIR = "data/coco/detections/yolo"

CONFIDENCE_THRESHOLD = 0.6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
with open(KARPATHY_JSON, "r") as f:
    data = json.load(f)

splits = {"train": [], "restval": [], "val": [], "test": []}
for img in data["images"]:
    if img["split"] in splits:
        splits[img["split"]].append(img)

train_images = splits["train"] + splits["restval"]
val_images = splits["val"]
test_images = splits["test"]

print(f"Karpathy Split:")
print(f"   Train+Restval: {len(train_images):,} images")
print(f"   Val:           {len(val_images):,} images")
print(f"   Test:          {len(test_images):,} images")

# %%
model = YOLO("yolo12x.pt")
print("YOLO12x model loaded")

def find_image_path(img_info):
    filename = img_info["filename"]
    if "train2014" in filename:
        return f"{TRAIN_DIR}/{filename}"
    return f"{VAL_DIR}/{filename}"

def detect_and_count(image_list, split_name):
    results_list = []

    print(f"Processing {split_name} set...")

    for img_info in tqdm(image_list, desc=split_name):
        img_path = find_image_path(img_info)
        filename = img_info["filename"]

        if not os.path.exists(img_path):
            continue

        results = model.predict(
            source=img_path, conf=CONFIDENCE_THRESHOLD, verbose=False
        )

        boxes = results[0].boxes
        labels_in_image = []
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            labels_in_image.append(class_name)

        counts = Counter(labels_in_image)
        # Format: "person 2, dog 1" — sorted by frequency (most prominent first)
        objects_str = ", ".join([f"{label} {count}" for label, count in counts.most_common()])
        if not objects_str:
            objects_str = "none"

        results_list.append({"image": filename, "objects": objects_str})
        
    df = pd.DataFrame(results_list)
    return df

def display_sample_images(image_list, split_name, n=20):
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for i, img_info in enumerate(image_list[:n]):
        img_path = find_image_path(img_info)

        if not os.path.exists(img_path):
            axes[i].set_title("Not found", fontsize=9)
            axes[i].axis("off")
            continue

        results = model.predict(
            source=img_path, conf=CONFIDENCE_THRESHOLD, verbose=False
        )

        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        axes[i].imshow(annotated)
        axes[i].set_title(img_info["filename"][:20] + "...", fontsize=9)
        axes[i].axis("off")

    plt.suptitle(f" First 20 images - {split_name} set", fontsize=14)
    plt.tight_layout()
    plt.show()


# %%
display_sample_images(test_images, "TEST", n=20)
df_test = detect_and_count(test_images, "TEST")

df_test.to_csv(os.path.join(OUTPUT_DIR, "coco_test_detection.csv"), index=False)
print("Saved: coco_test_detection.csv")

print("First 20 rows of coco_test_detection.csv:")
print(df_test.head())

# %%
display_sample_images(train_images, "TRAIN", n=20)

df_train = detect_and_count(train_images, "TRAIN")

df_train.to_csv(os.path.join(OUTPUT_DIR, "coco_train_detection.csv"), index=False)
print("Saved: coco_train_detection.csv")

print("First 20 rows of coco_train_detection.csv:")
print(df_train.head())

# %%
display_sample_images(val_images, "VAL", n=20)

df_val = detect_and_count(val_images, "VAL")

df_val.to_csv(os.path.join(OUTPUT_DIR, "coco_val_detection.csv"), index=False)
print("Saved: coco_val_detection.csv")

print("First 20 rows of coco_val_detection.csv:")
print(df_val.head())

# %%
print("=" * 70)
print("SUMMARY - MS COCO YOLO Detection")
print("=" * 70)

print(f"TRAIN: {len(train_images):,} images → {len(df_train):,} results")
print(f"VAL:   {len(val_images):,} images → {len(df_val):,} results")
print(f"TEST:  {len(test_images):,} images → {len(df_test):,} results")

print("Top 10 most common objects (all sets):")
all_objects = []
for df in [df_train, df_val, df_test]:
    for obj_str in df["objects"]:
        if obj_str != "none":
            for item in obj_str.split(", "):
                parts = item.rsplit(" ", 1)
                if len(parts) == 2:
                    label, count = parts
                    all_objects.extend([label] * int(count))

object_counts = Counter(all_objects)
for i, (obj, count) in enumerate(object_counts.most_common(10), 1):
    print(f"   {i:2d}. {obj}: {count:,}")

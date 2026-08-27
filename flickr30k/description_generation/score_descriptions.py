# %%
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# %%
# !pip install -q transformers accelerate

# %%
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
import os
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict, List
from transformers import BlipProcessor, BlipForImageTextRetrieval
from transformers import logging as hf_logging

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

# %%
KARPATHY_JSON  = "data/flickr30k/dataset_flickr30k.json"
IMAGE_DIR      = "data/flickr30k/flickr30k_images"
DESC_DIR       = "data/flickr30k/descriptions"

BLIP_MODEL_ID    = "Salesforce/blip-itm-base-coco"
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

TARGET = "train_A"
assert TARGET in ("val", "test", "train_A", "train_B", "train_C"), \
    "TARGET must be 'val', 'test', 'train_A', 'train_B', or 'train_C'"

_input_map = {
    "val":     "flickr30k_descs_split0_val.csv",
    "test":    "flickr30k_descs_split0_test.csv",
    "train_A": "flickr30k_descs_train_merged_A.csv",
    "train_B": "flickr30k_descs_train_merged_B.csv",
    "train_C": "flickr30k_descs_train_merged_C.csv",
}
INTERMEDIATE_CSV = os.path.join(DESC_DIR, _input_map[TARGET])
OUTPUT_CSV       = os.path.join(DESC_DIR, f"flickr30k_descs_{TARGET}_scored.csv")

print(f"Device: {DEVICE}")

# %%
with open(KARPATHY_JSON, "r") as f:
    karpathy_data = json.load(f)

filename_to_path: Dict[str, str] = {}
for img_info in karpathy_data["images"]:
    filename = img_info["filename"]
    filename_to_path[filename] = os.path.join(IMAGE_DIR, filename)

print(f"Total images in Karpathy: {len(filename_to_path):,}")

# %%
print(f"\nLoading BLIP ITM: {BLIP_MODEL_ID}...")
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
blip_model     = BlipForImageTextRetrieval.from_pretrained(BLIP_MODEL_ID)
blip_model.to(DEVICE)
blip_model.eval()
print("BLIP loaded!")

hf_logging.set_verbosity_error()

intermediate_df = pd.read_csv(INTERMEDIATE_CSV)
print(f"Loaded intermediate CSV: {len(intermediate_df):,} images")


@torch.no_grad()
def score_descs_blip(image_path: str, descs: List[str]) -> List[float]:
    """Return the ITM score (match probability) of each desc against the image."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return [0.0] * len(descs)

    scores = []
    for desc in descs:
        if not desc:
            scores.append(0.0)
            continue
        try:
            inputs = blip_processor(images=image, text=desc, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            output = blip_model(**inputs, use_itm_head=True)
            prob = F.softmax(output.itm_score, dim=-1)[0, 1].item()
            scores.append(prob)
        except Exception:
            scores.append(0.0)
    return scores


final_rows = []
for _, row in tqdm(intermediate_df.iterrows(), total=len(intermediate_df), desc="Scoring BLIP ITM"):
    img      = row["image"]
    descs     = [str(row.get(f"desc_{j+1}", "")) for j in range(5)]
    img_path = filename_to_path.get(img, "")

    if img_path and os.path.exists(img_path):
        scores = score_descs_blip(img_path, descs)
    else:
        scores = [0.0] * 5

    best_idx   = int(np.argmax(scores))
    best_desc   = descs[best_idx]
    best_score = scores[best_idx]

    final_row = {"image": img}
    for j in range(5):
        final_row[f"desc_{j+1}"]   = descs[j]
        final_row[f"score_{j+1}"] = round(scores[j], 4)
    final_row["best_desc"]   = best_desc
    final_row["best_score"] = round(best_score, 4)
    final_rows.append(final_row)

final_df = pd.DataFrame(final_rows)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved final CSV: {OUTPUT_CSV}")
print(f"Total images: {len(final_df):,}")

print("\n=== STATISTICS ===")
print(f"Images with best_score > 0.5: {(final_df['best_score'] > 0.5).sum():,}")
print(f"Images with best_score > 0.3: {(final_df['best_score'] > 0.3).sum():,}")
print(f"Average best score:   {final_df['best_score'].mean():.4f}")
print(f"\n10 sample images:")
print(final_df[["image", "best_desc", "best_score"]].head(10).to_string(index=False))

# %%
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# %%
KARPATHY_JSON = "data/flickr30k/dataset_flickr30k.json"
IMAGE_DIR = "data/flickr30k/flickr30k_images"
OUTPUT_DIR = "data/flickr30k/detections/hoi"

CHECKPOINT_PATH = "checkpoints/hoiclip_checkpoint.pth"

HOI_THRESHOLD = 0.25
TOP_K = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
import os
import sys

# !git clone https://github.com/Artanic30/HOICLIP.git
# !pip install -q ftfy regex tqdm opencv-python scipy cython pycocotools
# !pip install -q git+https://github.com/openai/CLIP.git

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import argparse
import pandas as pd
import json
from tqdm import tqdm
from collections import Counter

hoiclip_path = os.path.join(os.getcwd(), 'HOICLIP')
sys.path.insert(0, hoiclip_path)

from models import build_model
import util.misc as utils
from datasets.static_hico import ACT_IDX_TO_ACT_NAME, OBJ_IDX_TO_OBJ_NAME, HOI_IDX_TO_ACT_IDX, HOI_IDX_TO_OBJ_IDX
from ModifiedCLIP import clip

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

# %%
def get_args():
    parser = argparse.ArgumentParser('HOICLIP Inference')
    parser.add_argument('--model_name', default='HOICLIP', type=str)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=3, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=64, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--hoi', action='store_true', default=True)
    parser.add_argument('--num_obj_classes', type=int, default=80)
    parser.add_argument('--num_verb_classes', type=int, default=117)
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--dataset_file', default='hico', type=str)
    parser.add_argument('--dataset_root', default='GEN', type=str)
    parser.add_argument('--with_clip_label', action='store_true', default=True)
    parser.add_argument('--with_obj_clip_label', action='store_true', default=True)
    parser.add_argument('--clip_model', default='ViT-B/32', type=str)
    parser.add_argument('--clip_embed_dim', default=512, type=int)
    parser.add_argument('--fix_clip', action='store_true')
    parser.add_argument('--no_clip_cls_init', action='store_true')
    parser.add_argument('--fix_clip_label', action='store_true', default=False)
    parser.add_argument('--aux_loss', action='store_true', default=True)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--verb_pth', default='./HOICLIP/tmp/verb.pth', type=str)
    parser.add_argument('--verb_weight', default=0.5, type=float)
    parser.add_argument('--zero_shot_type', default='default', type=str)
    parser.add_argument('--with_mimic', action='store_true')
    parser.add_argument('--with_rec_loss', action='store_true')
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--hoi_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--mimic_loss_coef', default=20, type=float)
    parser.add_argument('--rec_loss_coef', default=2, type=float)
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=2.5, type=float)
    parser.add_argument('--set_cost_giou', default=1, type=float)
    parser.add_argument('--set_cost_obj_class', default=1, type=float)
    parser.add_argument('--set_cost_verb_class', default=1, type=float)
    parser.add_argument('--set_cost_hoi', default=1, type=float)
    parser.add_argument('--verb_loss_type', default='focal', type=str)
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--masks', action='store_true', default=False)
    parser.add_argument('--frozen_weights', default=None, type=str)
    parser.add_argument('--num_feature_levels', default=1, type=int)
    parser.add_argument('--with_box_refine', action='store_true', default=False)
    parser.add_argument('--two_stage', action='store_true', default=False)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--use_insadapter', action='store_true', default=False)
    parser.add_argument('--prior_method', default='', type=str)
    parser.add_argument('--LA', action='store_true', default=False)
    parser.add_argument('--LA_weight', default=0.5, type=float)
    parser.add_argument('--box_proj', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--inter_dec_layers', default=3, type=int)
    parser.add_argument('--use_spatial', action='store_true', default=False)
    parser.add_argument('--spatial_dim', default=64, type=int)
    parser.add_argument('--use_verb_temp', action='store_true', default=False)
    parser.add_argument('--verb_temp', default=1.0, type=float)
    parser.add_argument('--use_obj_temp', action='store_true', default=False)
    parser.add_argument('--obj_temp', default=1.0, type=float)
    parser.add_argument('--hoi_aux_loss', action='store_true', default=False)
    parser.add_argument('--use_matching', action='store_true', default=False)
    parser.add_argument('--dec_layers_hopd', default=3, type=int)
    parser.add_argument('--dec_layers_interaction', default=3, type=int)
    parser.add_argument('--use_nms_filter', action='store_true', default=False)
    parser.add_argument('--nms_thresh', default=0.7, type=float)
    parser.add_argument('--zero_shot_setting', default='default', type=str)
    parser.add_argument('--rare_first_setting', default='default', type=str)
    parser.add_argument('--unseen_idx_file', default=None, type=str)
    parser.add_argument('--del_unseen', action='store_true', default=False)
    args = parser.parse_args(args=[])
    return args

args = get_args()

# %%
def load_model(checkpoint_path, args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Building model...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)

    model.eval()
    print("Model loaded successfully!")

    return model, postprocessors, device

model, postprocessors, device = load_model(CHECKPOINT_PATH, args)

# %%
with open(KARPATHY_JSON, "r") as f:
    data = json.load(f)

splits = {"train": [], "val": [], "test": []}
for img in data["images"]:
    if img["split"] in splits:
        splits[img["split"]].append(img)

train_images = splits["train"]
val_images = splits["val"]
test_images = splits["test"]

print(f"Karpathy Split:")
print(f"   Train: {len(train_images):,} images")
print(f"   Val:   {len(val_images):,} images")
print(f"   Test:  {len(test_images):,} images")

# %%
print("Loading CLIP preprocessor...")
_, clip_preprocess = clip.load(args.clip_model, device=device)
print("CLIP loaded!")

image_transform = transforms.Compose([
    transforms.Resize((800, 1333)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def find_image_path(img_info):
    filename = img_info["filename"]
    return f"{IMAGE_DIR}/{filename}"

def preprocess_image(image_path, device):
    original_image = Image.open(image_path).convert('RGB')
    orig_size = original_image.size

    image_tensor = image_transform(original_image).unsqueeze(0).to(device)
    clip_input = clip_preprocess(original_image).unsqueeze(0).to(device)

    orig_size_tensor = torch.tensor([[orig_size[1], orig_size[0]]]).to(device)

    return image_tensor, clip_input, original_image, orig_size_tensor


@torch.no_grad()
def inference(model, postprocessors, image_tensor, clip_input, orig_size, device):
    model.eval()
    outputs = model(image_tensor, is_training=False, clip_input=clip_input, targets=None)
    results = postprocessors['hoi'](outputs, orig_size)
    return results[0]


def get_hoi_predictions(result, threshold=HOI_THRESHOLD, top_k=TOP_K):
    hoi_scores = result['hoi_scores']

    hoi_scores_max, hoi_classes = hoi_scores.max(dim=-1)
    sorted_indices = torch.argsort(hoi_scores_max, descending=True, stable=True)

    unique_predictions = {}

    for idx in sorted_indices:
        hoi_score = hoi_scores_max[idx].item()

        if hoi_score < threshold:
            continue

        hoi_id = hoi_classes[idx].item()
        verb_id = HOI_IDX_TO_ACT_IDX[hoi_id]
        obj_id = HOI_IDX_TO_OBJ_IDX[hoi_id]

        verb_name = ACT_IDX_TO_ACT_NAME[verb_id]
        obj_name = OBJ_IDX_TO_OBJ_NAME[obj_id]

        key = (verb_name, obj_name)

        if key not in unique_predictions:
            unique_predictions[key] = hoi_score

    predictions = [(verb, obj, score) for (verb, obj), score in unique_predictions.items()]
    predictions.sort(key=lambda x: x[2], reverse=True)

    return predictions[:top_k]

print(f"   HOI threshold: {HOI_THRESHOLD}")
print(f"   Top K: {TOP_K}")

# %%
def detect_and_save_csv(image_list, split_name, csv_filename):
    results_list = []

    print(f"Processing {split_name} set...")

    for img_info in tqdm(image_list, desc=split_name):
        img_path = find_image_path(img_info)
        filename = img_info["filename"]

        if not os.path.exists(img_path):
            continue

        try:
            image_tensor, clip_input, _, orig_size = preprocess_image(img_path, device)
            result = inference(model, postprocessors, image_tensor, clip_input, orig_size, device)
            predictions = get_hoi_predictions(result)

            if predictions:
                for verb, obj, score in predictions:
                    results_list.append({
                        'image': filename,
                        'prediction': f"Human - {verb} - {obj}",
                        'score': round(score, 4)
                    })
            else:
                results_list.append({
                    'image': filename,
                    'prediction': 'no_interaction',
                    'score': 0.0
                })

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    df = pd.DataFrame(results_list)
    df.to_csv(csv_filename, index=False)

    print(f"Saved: {csv_filename}")
    print(f"   Total rows: {len(df)}")
    print(f"   Number of images: {df['image'].nunique()}")

    return df

def display_sample_with_predictions(image_list, df, split_name, n=20):

    print(f"\n{'='*70}")
    print(f"FIRST 20 IMAGES - {split_name} SET")
    print(f"{'='*70}")

    sample_images = image_list[:n]

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))

    for i, img_info in enumerate(sample_images):
        img_path = find_image_path(img_info)

        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[i].imshow(img)

        axes[i].set_title(f"{i+1}. {img_info['filename'][:20]}...", fontsize=9)
        axes[i].axis('off')

    plt.suptitle(f"First 20 images - {split_name} set", fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"PREDICTIONS FOR THE FIRST 20 IMAGES:")
    print("-"*70)

    for i, img_info in enumerate(sample_images):
        filename = img_info["filename"]
        print(f" Image {i+1}: {filename}")

        img_predictions = df[df['image'] == filename]

        if len(img_predictions) == 0:
            print("   (No predictions)")
        else:
            for j, row in img_predictions.iterrows():
                print(f"   • [{row['prediction']}] - Score: {row['score']}")

    print("-"*70)


# %%
df_val = detect_and_save_csv(val_images, "VAL", os.path.join(OUTPUT_DIR, "flickr30k_val_hoi_detection.csv"))

display_sample_with_predictions(val_images, df_val, "VAL", n=20)

print("Preview flickr30k_val_hoi_detection.csv (first 20 rows):")
print(df_val.head())

# %%
df_test = detect_and_save_csv(test_images, "TEST", os.path.join(OUTPUT_DIR, "flickr30k_test_hoi_detection.csv"))

display_sample_with_predictions(test_images, df_test, "TEST", n=20)

print("Preview flickr30k_test_hoi_detection.csv (first 20 rows):")
print(df_test.head())

# %%
df_train = detect_and_save_csv(train_images, "TRAIN", os.path.join(OUTPUT_DIR, "flickr30k_train_hoi_detection.csv"))

display_sample_with_predictions(train_images, df_train, "TRAIN", n=20)

print("Preview flickr30k_train_hoi_detection.csv (first 20 rows):")
print(df_train.head())

# %%
print(f"TRAIN: {len(train_images):,} images | {len(df_train):,} predictions | {df_train['prediction'].nunique()} HOI types")
print(f" TEST:  {len(test_images):,} images | {len(df_test):,} predictions | {df_test['prediction'].nunique()} HOI types")
print(f" VAL:   {len(val_images):,} images | {len(df_val):,} predictions | {df_val['prediction'].nunique()} HOI types")

print("Top 10 most common HOIs (all sets):")
all_predictions = pd.concat([df_train, df_test, df_val])
top_hoi = all_predictions['prediction'].value_counts().head(10)
for i, (hoi, count) in enumerate(top_hoi.items(), 1):
    print(f"   {i:2d}. {hoi}: {count:,}")

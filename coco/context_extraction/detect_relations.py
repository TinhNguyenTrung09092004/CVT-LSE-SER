# %%
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# %%
KARPATHY_JSON = "data/coco/dataset_coco.json"
TRAIN_DIR = "data/coco/train2014"
VAL_DIR = "data/coco/val2014"
OUTPUT_DIR = "data/coco/detections/reltr"
CONFIDENCE_THRESHOLD = 0.25
TOP_K = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# !git clone https://github.com/yrcong/RelTR.git

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import json
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.getcwd(), 'RelTR'))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

# %%
# VG 150 entity classes
CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
           'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
           'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
           'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
           'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
           'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
           'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
           'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
           'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
           'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
           'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
           'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
           'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
           'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

# VG 50 relationship classes
REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

print(f"✅ Loaded {len(CLASSES)} entity classes and {len(REL_CLASSES)} relation classes")

# %%
from models.backbone import Backbone, Joiner
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.reltr import RelTR

position_embedding = PositionEmbeddingSine(128, normalize=True)
backbone = Backbone('resnet50', False, False, False)
backbone = Joiner(backbone, position_embedding)
backbone.num_channels = 2048

transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                          dim_feedforward=2048,
                          num_encoder_layers=6,
                          num_decoder_layers=6,
                          normalize_before=False,
                          return_intermediate_dec=True)

model = RelTR(backbone, transformer, num_classes=151, num_rel_classes=51,
              num_entities=100, num_triplets=200)

checkpoint_path = "checkpoints/reltr_checkpoint0149.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()

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
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def find_image_path(img_info):
    filename = img_info["filename"]
    if "train2014" in filename:
        return f"{TRAIN_DIR}/{filename}"
    return f"{VAL_DIR}/{filename}"


def get_predictions(img_path, threshold=CONFIDENCE_THRESHOLD, top_k=TOP_K):
    im = Image.open(img_path).convert('RGB')
    img = transform(im).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)

    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]

    keep = torch.logical_and(
        probas.max(-1).values > threshold,
        torch.logical_and(
            probas_sub.max(-1).values > threshold,
            probas_obj.max(-1).values > threshold
        )
    )

    keep_queries = torch.nonzero(keep, as_tuple=True)[0]

    if len(keep_queries) == 0:
        return []

    combined_scores = (probas[keep_queries].max(-1)[0] *
                       probas_sub[keep_queries].max(-1)[0] *
                       probas_obj[keep_queries].max(-1)[0])
    indices = torch.argsort(-combined_scores, stable=True)
    keep_queries = keep_queries[indices]

    unique_predictions = {}

    for idx in keep_queries:
        subject = CLASSES[probas_sub[idx].argmax()]
        predicate = REL_CLASSES[probas[idx].argmax()]
        obj = CLASSES[probas_obj[idx].argmax()]

        sub_conf = probas_sub[idx].max().item()
        rel_conf = probas[idx].max().item()
        obj_conf = probas_obj[idx].max().item()
        avg_score = (sub_conf + rel_conf + obj_conf) / 3

        key = (subject, predicate, obj)

        if key not in unique_predictions:
            unique_predictions[key] = {
                'subject': subject,
                'relation': predicate,
                'object': obj,
                'score': round(avg_score, 4)
            }

    predictions = list(unique_predictions.values())
    predictions.sort(key=lambda x: x['score'], reverse=True)

    return predictions[:top_k]


print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
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
            predictions = get_predictions(img_path)

            if predictions:
                for pred in predictions:
                    results_list.append({
                        'image': filename,
                        'prediction': f"{pred['subject']} - {pred['relation']} - {pred['object']}",
                        'score': pred['score']
                    })
            else:
                results_list.append({
                    'image': filename,
                    'prediction': 'no_relation',
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
        print(f"Image {i+1}: {filename}")

        img_predictions = df[df['image'] == filename]

        if len(img_predictions) == 0:
            print("   (No predictions)")
        else:
            for j, row in img_predictions.iterrows():
                print(f"   • [{row['prediction']}] - Score: {row['score']}")

# %%
df_test = detect_and_save_csv(test_images, "TEST", os.path.join(OUTPUT_DIR, "coco_test_reltr_detection.csv"))

display_sample_with_predictions(test_images, df_test, "TEST", n=20)

print("Preview coco_test_reltr_detection.csv (first 20 rows):")
print(df_test.head())

# %%
df_val = detect_and_save_csv(val_images, "VAL", os.path.join(OUTPUT_DIR, "coco_val_reltr_detection.csv"))

display_sample_with_predictions(val_images, df_val, "VAL", n=20)

print("Preview coco_val_reltr_detection.csv (first 20 rows):")
print(df_val.head())

# %%
df_train = detect_and_save_csv(train_images, "TRAIN", os.path.join(OUTPUT_DIR, "coco_train_reltr_detection.csv"))

display_sample_with_predictions(train_images, df_train, "TRAIN", n=20)

print("Preview coco_train_reltr_detection.csv (first 20 rows):")
print(df_train.head())
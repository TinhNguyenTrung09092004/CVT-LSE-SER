# Flickr30k Pipeline

Dataset-specific paths, split/part schemes, and run counts for **Flickr30k** (Karpathy split). For the project overview, shared model details, architecture, environment setup, and external checkpoints, see the [root README](../README.md). For the MS COCO counterpart, see [coco/README.md](../coco/README.md).

All commands below are run from the **repository root**, and every path variable in the scripts is relative to it.

## Dataset

- **Flickr30k**: `flickr30k_images/` (31,783 images in the official release)
- **Karpathy split** (`dataset_flickr30k.json`): 29,000 train / 1,014 val / 1,000 test — 31,014 of those images

### Download the images (local machine)

1. Open <http://shannon.cs.illinois.edu/DenotationGraph/> and submit the [access request form](https://illinois.edu/fb/sec/229675). Use of the images must abide by the [Flickr Terms of Use](http://www.flickr.com/help/terms/).
2. A download link for the image archive is e-mailed back to you. Download and extract it.
3. Move (and rename) that folder to `data/flickr30k/flickr30k_images/`, so the `.jpg` files sit **directly** inside it with no intermediate sub-directory — every script builds an image path as `IMAGE_DIR` + the bare `filename` field of the Karpathy JSON (e.g. `data/flickr30k/flickr30k_images/1000092795.jpg`).

### Download the Karpathy split (local machine)

```bash
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip dataset_flickr30k.json -d data/flickr30k/
```

---

## Data Directory Layout (local machine)

Organize as below, or update the path variables in each script:

```
data/flickr30k/
├── flickr30k_images/           # 31,783 .jpg files, flat (requested via the Illinois form)
├── dataset_flickr30k.json      # Karpathy split (downloaded)
├── train.txt                   # image lists shipped with this repository
├── val.txt
├── test.txt
├── detections/                 # Step 1 output
│   ├── yolo/
│   ├── hoi/
│   └── reltr/
├── descriptions/               # Step 2 output
│   ├── flickr30k_descs_train_scored.csv
│   ├── flickr30k_descs_val_scored.csv
│   └── flickr30k_descs_test_scored.csv
└── features/                   # Step 3 output
    ├── train_part1.h5
    ├── train_part2.h5
    ├── val.h5
    └── test.h5
```

Step 4 and Step 5 write to `outputs/flickr30k/`.

---

## Step 1 – Context Generation

Model choices, thresholds, and detector setup are in the [root README](../README.md#step-1--context-generation). Default path variables:

```python
# context_extraction/detect_objects.py
KARPATHY_JSON = "data/flickr30k/dataset_flickr30k.json"
IMAGE_DIR     = "data/flickr30k/flickr30k_images"
OUTPUT_DIR    = "data/flickr30k/detections/yolo"
```

```python
# context_extraction/detect_interactions.py
KARPATHY_JSON   = "data/flickr30k/dataset_flickr30k.json"
IMAGE_DIR       = "data/flickr30k/flickr30k_images"
OUTPUT_DIR      = "data/flickr30k/detections/hoi"
CHECKPOINT_PATH = "checkpoints/hoiclip_checkpoint.pth"
```

```python
# context_extraction/detect_relations.py
KARPATHY_JSON   = "data/flickr30k/dataset_flickr30k.json"
IMAGE_DIR       = "data/flickr30k/flickr30k_images"
OUTPUT_DIR      = "data/flickr30k/detections/reltr"
checkpoint_path = "checkpoints/reltr_checkpoint0149.pth"
```

```bash
python flickr30k/context_extraction/detect_objects.py
python flickr30k/context_extraction/detect_interactions.py
python flickr30k/context_extraction/detect_relations.py
```

---

## Step 2 – Semantic Description Generation (LLM Pipeline)

### 2a. Generate descriptions with Qwen3-4B

[description_generation/generate_descriptions.py](description_generation/generate_descriptions.py):

```python
KARPATHY_JSON = "data/flickr30k/dataset_flickr30k.json"
IMAGE_DIR     = "data/flickr30k/flickr30k_images"
DETECTION_DIR = "data/flickr30k/detections"
DESC_DIR      = "data/flickr30k/descriptions"
SPLIT         = 0.1    # see table below
```

**SPLIT values**:

| Value | Subset |
|-------|--------|
| 0.1   | test |
| 0.2   | val |
| X     | train part X/10 (X = 1..10) |

Run 12 times to cover all splits.

---

### 2b. Merge split CSV files

[description_generation/merge_descriptions.py](description_generation/merge_descriptions.py) reads the 10 train-split CSVs from step 2a out of `INPUT_DIR = "data/flickr30k/descriptions"`. The `test`/`val` outputs (`flickr30k_descs_split0_test.csv`, `flickr30k_descs_split0_val.csv`) need no merging — each is already a single file.

Groups merged:

| Group | Train parts |
|---|---|
| A | 1, 2, 3 |
| B | 4, 5, 6 |
| C | 7, 8, 9, 10 |

Output (same directory): `flickr30k_descs_train_merged_{A,B,C}.csv`

---

### 2c. Score descriptions with BLIP-ITM

[description_generation/score_descriptions.py](description_generation/score_descriptions.py):

```python
KARPATHY_JSON = "data/flickr30k/dataset_flickr30k.json"
IMAGE_DIR     = "data/flickr30k/flickr30k_images"
DESC_DIR      = "data/flickr30k/descriptions"
TARGET        = "train_A"   # "val" | "test" | "train_A" | "train_B" | "train_C"
```

Run 5 times (val, test, train_A, train_B, train_C).

---

### 2d. Merge scored files

[description_generation/merge_scored_descriptions.py](description_generation/merge_scored_descriptions.py):

```python
SCORED_A = "data/flickr30k/descriptions/flickr30k_descs_train_A_scored.csv"
SCORED_B = "data/flickr30k/descriptions/flickr30k_descs_train_B_scored.csv"
SCORED_C = "data/flickr30k/descriptions/flickr30k_descs_train_C_scored.csv"
OUTPUT   = "data/flickr30k/descriptions/flickr30k_descs_train_scored.csv"
```

---

## Step 3 – Feature Extraction

Backbone details are in the [root README](../README.md#step-3--feature-extraction). [feature_extraction/extract_features.py](feature_extraction/extract_features.py):

```python
PART          = "1.1"    # "1.1"-"1.2" = train, "2" = test, "3" = val
OUTPUT_DIR    = "data/flickr30k/features"
KARPATHY_JSON = "data/flickr30k/dataset_flickr30k.json"
IMAGE_DIR     = "data/flickr30k/flickr30k_images"
```

Run with `PART` set to `"1.1"`, `"1.2"`, `"2"`, `"3"` (4 runs total).

---

## Step 4 – Model Training

Architecture and default hyperparameters are in the [root README](../README.md#step-4--model-training). `Config` in [training/train.py](training/train.py):

```python
karpathy_json  = "data/flickr30k/dataset_flickr30k.json"
image_dir      = "data/flickr30k/flickr30k_images"
train_h5_parts = (
    "data/flickr30k/features/train_part1.h5",
    "data/flickr30k/features/train_part2.h5",
)
val_h5         = "data/flickr30k/features/val.h5"
test_h5        = "data/flickr30k/features/test.h5"
train_desc_csv = "data/flickr30k/descriptions/flickr30k_descs_train_scored.csv"
val_desc_csv   = "data/flickr30k/descriptions/flickr30k_descs_val_scored.csv"
test_desc_csv  = "data/flickr30k/descriptions/flickr30k_descs_test_scored.csv"
```

```bash
python flickr30k/training/train.py
```

Outputs:

| File | Written | Purpose |
|---|---|---|
| `outputs/flickr30k/checkpoint_latest.pt` | after every epoch, overwritten | resume training (`RESUME_FROM`) |
| `outputs/flickr30k/model_checkpoint_epoch_{N}.pth` | after every epoch, one file each | weights only |
| `outputs/flickr30k/beam_candidates_epoch_{N}.csv` | after every epoch, one file each | Step 5 input |
| `outputs/flickr30k/word2idx.json` | before training | vocabulary, required to rebuild the model |

---

## Step 5 – Caption Re-ranking

[caption_selection/rerank.py](caption_selection/rerank.py):

```python
CSV_PATH              = "outputs/flickr30k/beam_candidates_epoch_<EPOCH>.csv"    # from Step 4
DESC_CSV_PATH         = "data/flickr30k/descriptions/flickr30k_descs_test_scored.csv"
DETECTION_DATASET_DIR = "data/flickr30k/detections"
```

```bash
python flickr30k/caption_selection/rerank.py
```

Every image is re-ranked by the LLM; an image whose LLM response cannot be parsed falls back to the top-1 beam candidate (marked `beam_fallback` in the `selection_source` column).

LLM-selected captions are saved to `outputs/flickr30k/flickr30k_llm_rerank_selections.csv`.

# MS COCO Pipeline

Dataset-specific paths, split/part schemes, and run counts for **MS COCO 2014** (Karpathy split). For the project overview, shared model details, architecture, environment setup, and external checkpoints, see the [root README](../README.md). For the Flickr30k counterpart, see [flickr30k/README.md](../flickr30k/README.md).

All commands below are run from the **repository root**, and every path variable in the scripts is relative to it.

## Dataset

- **MS COCO 2014**: train2014, val2014 images
- **Karpathy splits** (`dataset_coco.json`): 113287 train+restval / 5k val / 5k test

### Download (local machine)

```bash
# COCO 2014 images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip train2014.zip -d data/coco/
unzip val2014.zip   -d data/coco/

# Karpathy splits
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip dataset_coco.json -d data/coco/

# COCO 2014 instance annotations (Step 6 CHAIR only)
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip -j annotations_trainval2014.zip annotations/instances_val2014.json -d data/coco/annotations/
```

---

## Data Directory Layout (local machine)

Organize as below, or update the path variables in each script:

```
data/coco/
├── train2014/                  # images (downloaded)
├── val2014/                    # images (downloaded)
├── dataset_coco.json           # Karpathy split (downloaded)
├── train_restval.txt           # image lists shipped with this repository
├── val.txt
├── test.txt
├── annotations/                # instances_val2014.json (downloaded, Step 6 CHAIR only)
├── detections/                 # Step 1 output
│   ├── yolo/
│   ├── hoi/
│   └── reltr/
├── descriptions/               # Step 2 output
│   ├── coco_descs_train_scored.csv
│   ├── coco_descs_val_scored.csv
│   └── coco_descs_test_scored.csv
└── features/                   # Step 3 output
    ├── train_part1.h5
    ├── train_part2.h5
    ├── train_part3.h5
    ├── train_part4.h5
    ├── train_part5.h5
    ├── train_part6.h5
    ├── val.h5
    └── test.h5
```

Steps 4 and 5 write to `outputs/coco/`; Step 6 only prints to stdout.

---

## Step 1 – Context Generation

Model choices, thresholds, and detector setup are in the [root README](../README.md#step-1--context-generation). Default path variables:

```python
# context_extraction/detect_objects.py
KARPATHY_JSON = "data/coco/dataset_coco.json"
TRAIN_DIR     = "data/coco/train2014"
VAL_DIR       = "data/coco/val2014"
OUTPUT_DIR    = "data/coco/detections/yolo"
```

```python
# context_extraction/detect_interactions.py
KARPATHY_JSON   = "data/coco/dataset_coco.json"
TRAIN_DIR       = "data/coco/train2014"
VAL_DIR         = "data/coco/val2014"
OUTPUT_DIR      = "data/coco/detections/hoi"
CHECKPOINT_PATH = "checkpoints/hoiclip_checkpoint.pth"
```

```python
# context_extraction/detect_relations.py
KARPATHY_JSON   = "data/coco/dataset_coco.json"
TRAIN_DIR       = "data/coco/train2014"
VAL_DIR         = "data/coco/val2014"
OUTPUT_DIR      = "data/coco/detections/reltr"
checkpoint_path = "checkpoints/reltr_checkpoint0149.pth"
```

```bash
python coco/context_extraction/detect_objects.py
python coco/context_extraction/detect_interactions.py
python coco/context_extraction/detect_relations.py
```

---

## Step 2 – Semantic Description Generation (LLM Pipeline)

### 2a. Generate descriptions with Qwen3-4B

[description_generation/generate_descriptions.py](description_generation/generate_descriptions.py):

```python
KARPATHY_JSON  = "data/coco/dataset_coco.json"
TRAIN_DIR      = "data/coco/train2014"
VAL_DIR        = "data/coco/val2014"
DETECTION_DIR  = "data/coco/detections"
DESC_DIR       = "data/coco/descriptions"
SPLIT          = 2.2    # see table below
```

**SPLIT values**:

| Value | Subset |
|-------|--------|
| 0.1   | test – first half |
| 0.2   | test – second half |
| 0.3   | val – first half |
| 0.4   | val – second half |
| X.1   | train part X/12, first half  (X = 1..12) |
| X.2   | train part X/12, second half (X = 1..12) |

Run 28 times to cover all splits.

---

### 2b. Merge split CSV files

[description_generation/merge_descriptions.py](description_generation/merge_descriptions.py) reads the 28 CSV files from step 2a out of `INPUT_DIR = "data/coco/descriptions"`.

Output (same directory):
- `coco_descs_train_merged_{A,B,C}.csv`
- `coco_descs_test_merged.csv`
- `coco_descs_val_merged.csv`

---

### 2c. Score descriptions with BLIP-ITM

[description_generation/score_descriptions.py](description_generation/score_descriptions.py):

```python
KARPATHY_JSON = "data/coco/dataset_coco.json"
TRAIN_DIR     = "data/coco/train2014"
VAL_DIR       = "data/coco/val2014"
DESC_DIR      = "data/coco/descriptions"
TARGET        = "train_A"   # "val" | "test" | "train_A" | "train_B" | "train_C"
```

Run 5 times (val, test, train_A, train_B, train_C).

---

### 2d. Merge scored files

[description_generation/merge_scored_descriptions.py](description_generation/merge_scored_descriptions.py):

```python
SCORED_A = "data/coco/descriptions/coco_descs_train_A_scored.csv"
SCORED_B = "data/coco/descriptions/coco_descs_train_B_scored.csv"
SCORED_C = "data/coco/descriptions/coco_descs_train_C_scored.csv"
OUTPUT   = "data/coco/descriptions/coco_descs_train_scored.csv"
```

---

## Step 3 – Feature Extraction

Backbone details are in the [root README](../README.md#step-3--feature-extraction). [feature_extraction/extract_features.py](feature_extraction/extract_features.py):

```python
PART          = "1.1"    # "1.1"-"1.6" = train, "2" = test, "3" = val
OUTPUT_DIR    = "data/coco/features"
KARPATHY_JSON = "data/coco/dataset_coco.json"
TRAIN_DIR     = "data/coco/train2014"
VAL_DIR       = "data/coco/val2014"
```

Run with `PART` set to `"1.1"`, `"1.2"`, `"1.3"`, `"1.4"`, `"1.5"`, `"1.6"`, `"2"`, `"3"` (8 runs total).

---

## Step 4 – Model Training

Architecture and default hyperparameters are in the [root README](../README.md#step-4--model-training). `Config` in [training/train.py](training/train.py):

```python
karpathy_json  = "data/coco/dataset_coco.json"
train_dir      = "data/coco/train2014"
val_dir        = "data/coco/val2014"
train_h5_parts = (
    "data/coco/features/train_part1.h5",
    "data/coco/features/train_part2.h5",
    "data/coco/features/train_part3.h5",
    "data/coco/features/train_part4.h5",
    "data/coco/features/train_part5.h5",
    "data/coco/features/train_part6.h5",
)
val_h5         = "data/coco/features/val.h5"
test_h5        = "data/coco/features/test.h5"
train_desc_csv = "data/coco/descriptions/coco_descs_train_scored.csv"
val_desc_csv   = "data/coco/descriptions/coco_descs_val_scored.csv"
test_desc_csv  = "data/coco/descriptions/coco_descs_test_scored.csv"
```

```bash
python coco/training/train.py
```

Outputs:

| File | Written | Purpose |
|---|---|---|
| `outputs/coco/checkpoint_latest.pt` | after every epoch, overwritten | resume training (`RESUME_FROM`) |
| `outputs/coco/model_checkpoint_epoch_{N}.pth` | after every epoch, one file each | weights only |
| `outputs/coco/beam_candidates_epoch_{N}.csv` | after every epoch, one file each | Step 5 / Step 6 input |
| `outputs/coco/word2idx.json` | before training | vocabulary, required to rebuild the model |


---

## Step 5 – Caption Re-ranking

[caption_selection/rerank.py](caption_selection/rerank.py):

```python
CSV_PATH      = "outputs/coco/beam_candidates_epoch_<EPOCH>.csv"    # from Step 4
DESC_CSV_PATH = "data/coco/descriptions/coco_descs_test_scored.csv"
DETECTION_DIR = "data/coco/detections"
```

```bash
python coco/caption_selection/rerank.py
```

Every image is re-ranked by the LLM; an image whose LLM response cannot be parsed falls back to the top-1 beam candidate (marked `beam_fallback` in the `selection_source` column).

LLM-selected captions are saved to `outputs/coco/coco_llm_rerank_selections.csv`.

---

## Step 6 – Analysis

MS COCO only. What each script measures is described in the [root README](../README.md#step-6--analysis-ms-coco-only).

### 6a. CHAIR object hallucination

[analysis/evaluate_chair.py](analysis/evaluate_chair.py):

```python
INPUT_TYPE     = "both"    # "type1" = candidates only, "type2" = selections only
CANDIDATES_CSV = "outputs/coco/beam_candidates_epoch_<EPOCH>.csv"          # from Step 4
SELECTIONS_CSV = "outputs/coco/coco_llm_rerank_selections.csv"       # from Step 5
INSTANCES_JSON = "data/coco/annotations/instances_val2014.json"
```

```bash
python coco/analysis/evaluate_chair.py
```

---

### 6b. Candidate order sensitivity

[analysis/evaluate_order_sensitivity.py](analysis/evaluate_order_sensitivity.py) re-runs the Step 5 re-ranker `N_ORDERINGS` times over the same candidates, each time presenting them in a different order:

```python
CSV_PATH       = "outputs/coco/beam_candidates_epoch_<EPOCH>.csv"          # from Step 4
DESC_CSV_PATH  = "data/coco/descriptions/coco_descs_test_scored.csv" # from Step 2
DETECTION_DIR  = "data/coco/detections"                              # from Step 1
LLM_MODEL_ID   = "Qwen/Qwen3-4B"
N_ORDERINGS    = 5      # P1 = shuffle at ORDER_SEED, P2-P5 = cyclic rotations of P1
ORDER_SEED     = 42
```

```bash
python coco/analysis/evaluate_order_sensitivity.py
```

---

### 6c. Paired significance test

[analysis/compare_significance.py](analysis/compare_significance.py) compares two caption sets on the same images:

```python
METHOD_A_CSV = "outputs/coco/coco_llm_rerank_selections.csv"   # from Step 5
METHOD_B_CSV = "outputs/coco/beam_candidates_epoch_<EPOCH>.csv"      # from Step 4, best rank per image
N_BOOTSTRAP  = 2000
```

```bash
python coco/analysis/compare_significance.py
```

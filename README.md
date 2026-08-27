# Evidence-Guided LLM-Based Semantic Enhancement and Caption Selection for Image Captioning

Image captioning requires representing objects, actions, relationships, and context, yet existing methods often underuse image-derived semantic evidence for decoder conditioning and rank beam candidates mainly by language probability. This paper proposes a Transformer-based method that fuses CLIP-ViT features with semantic representations encoded from LLM-generated descriptions. Rather than generating captions directly, the LLM performs two controlled roles: verbalising evidence about objects, relations, and human-object interactions into sentence-level descriptions and selecting the final caption from decoder-generated candidates. BLIP-ITM scores and softly weights descriptions before fusion, while semantic evidence guides final caption selection. Experiments on MS COCO show the method achieves mean BLEU-4/CIDEr scores of 38.7/129.0 over three training seeds; the corresponding Flickr30K scores are 30.9/76.3. Component ablations show complementary contributions from semantic enhancement and evidence-guided selection. Further analyses show lower CHAIR scores, limited seed variation, and statistically significant gains over beam-rank-1 selection on several metrics, while quantifying candidate-order sensitivity.

```
image
  │
  ├─▶ YOLO12x / HOICLIP / RelTR ──▶ objects, actions, relations
  │                                        │
  │                                        ▼
  │                              Qwen3-4B description generation
  │                                        │
  │                                        ▼
  │                                BLIP-ITM description scoring
  │                                        │
  ▼                                        ▼
CLIP ViT-L/14 features ───────▶ Encoder–Decoder training ──▶ beam search captions
                                                                     │
                                                                     ▼
                                                    Qwen3-4B re-ranking ──▶ final caption
```

Both **MS COCO 2014** and **Flickr30k** (Karpathy splits) run this exact pipeline. Model choices, hyperparameters, architecture, and external checkpoints are identical across datasets and documented once below; each dataset has its own subdirectory and README covering only what differs - paths, split/part schemes, and run counts.

## Repository Layout

| Directory | Dataset | Karpathy Split | Docs |
|---|---|---|---|
| [`coco/`](coco/) | MS COCO 2014 | 113287 train+restval / 5k val / 5k test | [coco/README.md](coco/README.md) |
| [`flickr30k/`](flickr30k/) | Flickr30k | 29000 train / 1014 val / 1k test | [flickr30k/README.md](flickr30k/README.md) |

Both datasets share the same directory structure for Steps 1-5; Step 6 (`analysis/`) exists for MS COCO only:

```
<dataset>/
    │
    ├── context_extraction/                   # Step 1 - visual context extraction
    │       ├── detect_objects.py             # YOLO12x  → detected objects
    │       ├── detect_interactions.py        # HOICLIP  → human–object interactions
    │       └── detect_relations.py           # RelTR    → scene graph relations
    │
    ├── description_generation/               # Step 2 - LLM semantic descriptions
    │       ├── generate_descriptions.py      # Qwen3-4B generates 5 descriptions/image from context
    │       ├── merge_descriptions.py         # Merge train description splits
    │       ├── score_descriptions.py         # BLIP-ITM scores each description
    │       └── merge_scored_descriptions.py  # Merge scored CSV files
    │
    ├── feature_extraction/                   # Step 3
    │       └── extract_features.py           # CLIP ViT-L/14 vision encoder → HDF5 (256, 1024)
    │
    ├── training/                             # Step 4
    │       └── train.py                      # Train Encoder–Decoder + beam search inference
    │
    ├── caption_selection/                    # Step 5
    │       └── rerank.py                     # Qwen3-4B re-ranks beam search candidates
    │
    └── analysis/                             # Step 6 - MS COCO only
            ├── evaluate_chair.py             # CHAIR_s / CHAIR_i object hallucination
            ├── evaluate_order_sensitivity.py # re-ranking stability across candidate orderings
            └── compare_significance.py       # paired bootstrap test, LLM re-rank vs beam top-1
```


All scripts use paths relative to the repository root, so run them from there:

```bash
python coco/training/train.py
python flickr30k/training/train.py
```

Generated artifacts follow the same convention for both datasets:

```
data/<dataset>/
    ├── dataset_<dataset>.json    # Karpathy split (downloaded)
    ├── <images>/                 # dataset images (downloaded)
    ├── annotations/                   # COCO instance annotations (downloaded, Step 6 only)
    ├── detections/{yolo,hoi,reltr}/   # Step 1 output
    ├── descriptions/                  # Step 2 output
    └── features/                      # Step 3 output

checkpoints/                      # HOICLIP / RelTR external checkpoints
outputs/<dataset>/                # Step 4-5 output: checkpoints, beam candidates, re-rank selections
```

---

## Environment

Tested on:

| Component | Version |
|---|---|
| Python | 3.12.12 |
| CUDA | 12.6 |
| GPU | Tesla T4 × 2 (14 911 MB each) |
| PyTorch | 2.9.0+cu126 |
| torchvision | 0.24.0+cu126 |
| transformers | 5.2.0 |
| accelerate | 1.12.0 |
| bitsandbytes | 0.50.1 |
| ultralytics | 8.4.129 |

### Install all dependencies

```bash
# 1. PyTorch with CUDA 12.6 (must be installed first)
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126

# 2. Everything else, including the git-only packages (CLIP, pycocoevalcap)
pip install -r requirements.txt
```

> For a different CUDA version replace `cu126` with e.g. `cu121`.
> Full version list is in [requirements.txt](requirements.txt). This single file covers both the MS COCO and Flickr30k pipelines.

> **Java is also required** (system dependency, not installable via pip): `pycocoevalcap`'s METEOR scorer and PTBTokenizer shell out to a JRE. Tested with OpenJDK 17.0.17. Install it via your system package manager before running Step 5 re-ranking/evaluation.

---

## Shared Pipeline Details

The steps below are identical for both datasets — only file paths, `SPLIT`/`PART` values, and run counts differ (see each dataset's README).

### Step 1 – Context Generation

| Script | Model | Key thresholds | Output |
|---|---|---|---|
| `context_extraction/detect_objects.py` | YOLO12x (`ultralytics`) | `CONFIDENCE_THRESHOLD = 0.6` | `detections/yolo/{dataset}_{train,val,test}_detection.csv` |
| `context_extraction/detect_interactions.py` | [HOICLIP](https://github.com/Artanic30/HOICLIP), pretrained on HICO-DET | `HOI_THRESHOLD = 0.25`, `TOP_K = 10` | `detections/hoi/{dataset}_{train,val,test}_hoi_detection.csv` |
| `context_extraction/detect_relations.py` | [RelTR](https://github.com/yrcong/RelTR), pretrained on Visual Genome | `CONFIDENCE_THRESHOLD = 0.25`, `TOP_K = 10` | `detections/reltr/{dataset}_{train,val,test}_reltr_detection.csv` |

Clone the detector repos into the repository root before running:

```bash
git clone https://github.com/Artanic30/HOICLIP.git
git clone https://github.com/yrcong/RelTR.git
```

Each script writes straight into `data/{dataset}/detections/{yolo,hoi,reltr}/`, which is where Steps 2 and 5 read them from.

### Step 2 – Semantic Description Generation (LLM Pipeline)

- **2a.** `description_generation/generate_descriptions.py` — `Qwen/Qwen3-4B` generates 5 descriptions per image from the merged detection context. `SPLIT` values and the number of runs needed differ per dataset (see dataset README).
- **2b.** `description_generation/merge_descriptions.py` — merges the per-split description CSVs from 2a into train groups (grouping scheme is dataset-specific).
- **2c.** `description_generation/score_descriptions.py` — `Salesforce/blip-itm-base-coco` scores each of the 5 descriptions against its image (ITM match probability).
- **2d.** `description_generation/merge_scored_descriptions.py` — merges the 3 scored train groups (`train_A`/`train_B`/`train_C`) into one `{dataset}_descs_train_scored.csv`.

All four scripts read and write inside `data/{dataset}/descriptions/`.

### Step 3 – Feature Extraction

**Backbone**: CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) vision encoder → patch embeddings → shape `(256, 1024)`, identical for both datasets. Weights download automatically via `from_pretrained` on first run — no manual checkpoint download required.


Only the number of `PART` values differs, since MS COCO's larger train set is chunked into more shards (see dataset README).

### Step 4 – Model Training

`training/train.py` trains the same Encoder–Decoder architecture for both datasets:

| Component | Details |
|---|---|
| Encoder | RoPE-2D + Learned-2D PE + Sinusoidal-2D PE → TransformerEncoder (2 layers) → attention pooling |
| DescVisualCrossAttention | Qwen3-4B descriptions cross-attend to visual features |
| DescTokenEncoder | Shared embedding, weighted by BLIP scores |
| Decoder | RoPE + TransformerDecoder (3 layers) |
| DescAuxHead | Predicts description embedding (cosine loss) |
| Loss | CrossEntropy (label_smoothing=0.1) + λ · desc_loss |

**Default hyperparameters** (identical for both datasets):

```python
feature_dim     = 1024   # CLIP ViT-L/14 patch embedding dim
grid_h          = 16     # 224 / 14 patch grid
grid_w          = 16
d_model         = 768
n_heads_encoder = 12
n_heads_decoder = 12
batch_size      = 64
learning_rate   = 1e-4
warmup_steps    = 2000
lambda_desc     = 0.3
BEAM_WIDTH      = 5
```


### Step 5 – Caption Re-ranking

`caption_selection/rerank.py` uses `Qwen/Qwen3-4B` to pick the best beam-search candidate per image, using the same YOLO/HOI/RelTR detection evidence and BLIP-scored descriptions as context. Every image goes through the LLM; only an unparsable LLM response falls back to the top-1 beam candidate. The selected captions are written to `outputs/{dataset}/{dataset}_llm_rerank_selections.csv`, and BLEU-1/2/3/4, CIDEr, METEOR, and ROUGE-L are printed to stdout.

### Step 6 – Analysis (MS COCO only)

Three analyses beyond the BLEU/CIDEr/METEOR/ROUGE-L numbers of Step 5. All three read the Step 4 beam candidates and the Step 5 selections, so run them after Step 5; only MS COCO is covered (see [coco/README.md](coco/README.md#step-6--analysis)).

| Script | Measures |
|---|---|
| `analysis/evaluate_chair.py` | CHAIR_s / CHAIR_i object hallucination of the beam top-1 candidate and of the LLM-selected caption |
| `analysis/evaluate_order_sensitivity.py` | How often the re-ranker's choice changes when the candidates are presented in a different order - 5 cyclic orderings of the same candidate set, re-ranked independently |
| `analysis/compare_significance.py` | Paired bootstrap significance test (2000 resamples) of LLM re-ranking vs beam top-1 on all caption metrics |

### Computational Cost

`analysis/measure_complexity.py` reports parameters, FLOPs, and latency **per image** (batch = 1) for every model in the pipeline.

```bash
python analysis/measure_complexity.py
```

---

## External Checkpoints

Shared across both datasets:

| Checkpoint | Source | Expected local path |
|---|---|---|
| YOLO12x [`yolo12x.pt`](https://docs.ultralytics.com/models/yolo12/) | [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | downloaded automatically |
| HOICLIP [`checkpoint_default.pth`](https://drive.google.com/file/d/1q3JuEzICoppij3Wce9QfwZ1k9a4HZ9or/view)| [github.com/Artanic30/HOICLIP](https://github.com/Artanic30/HOICLIP) | `checkpoints/hoiclip_checkpoint.pth` |
| RelTR [`checkpoint0149.pth`](https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD/view) | [github.com/yrcong/RelTR](https://github.com/yrcong/RelTR) | `checkpoints/reltr_checkpoint0149.pth` |
| Qwen3-4B [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B) | HuggingFace | downloaded automatically |
| BLIP-ITM [`Salesforce/blip-itm-base-coco`](https://huggingface.co/Salesforce/blip-itm-base-coco) | HuggingFace | downloaded automatically |
| CLIP ViT-L/14 [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | HuggingFace | downloaded automatically |

## Trained Checkpoints

Step 4 weights and vocabularies for both datasets: [Google Drive](https://drive.google.com/drive/folders/13scQJGQdkFjn6ebY5jKmO-KzwofJr50G)

---

## Qualitative Examples

See [examples/README.md](examples/README.md) for four annotated images from the MS COCO Karpathy test split (5k images) comparing the baseline (`CVT`) against the proposed semantic-evidence fusion and LLM re-ranking components (`CVT-SER`, `CVT-LSE`, `CVT-LSE-SER`), with generated evidence, candidate captions, and ground truths.

---

## Getting Started

1. Install dependencies (see [Environment](#environment) above).
2. Pick a dataset and follow its README for data download, directory layout, and dataset-specific `SPLIT`/`PART` values, run counts, and path variables:
   - [coco/README.md](coco/README.md)
   - [flickr30k/README.md](flickr30k/README.md)

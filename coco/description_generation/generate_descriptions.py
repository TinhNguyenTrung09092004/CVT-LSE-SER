# %%
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# %%
# !pip install -q transformers accelerate bitsandbytes

# %%
import torch
import pandas as pd
import json
import os
import re
from tqdm.auto import tqdm
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)

# %%
KARPATHY_JSON  = "data/coco/dataset_coco.json"
TRAIN_DIR      = "data/coco/train2014"
VAL_DIR        = "data/coco/val2014"
DETECTION_DIR  = "data/coco/detections"
DESC_DIR       = "data/coco/descriptions"

QWEN_MODEL_ID  = "Qwen/Qwen3-4B"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

QWEN_BATCH_SIZE  = 5
K_PROMPT         = 3

# === SPLIT ===
# 0.1  → test first half
# 0.2  → test second half
# 0.3  → val first half
# 0.4  → val second half
# 1.1  → train part 1/12, first half
# 1.2  → train part 1/12, second half
# 2.1  → train part 2/12, first half
# 2.2  → train part 2/12, second half
# ...
# 12.1 → train part 12/12, first half
# 12.2 → train part 12/12, second half
SPLIT = 2.2

def _parse_split(s):
    """Return (main_part, sub_part): sub_part=0 means the whole part."""
    if s == 0 or s == int(s):
        return int(s), 0
    main = int(s)
    sub  = round(s * 10) % 10
    return main, sub

_main, _sub = _parse_split(SPLIT)
assert (_main == 0 and _sub in [1, 2, 3, 4]) or (1 <= _main <= 12 and _sub in [1, 2]), \
    "SPLIT must be 0.1/0.2 (test first/second half), 0.3/0.4 (val first/second half), or X.1/X.2 with X from 1-12"

if _main == 0:
    if _sub == 1:   _split_label = "split0_test_1"
    elif _sub == 2: _split_label = "split0_test_2"
    elif _sub == 3: _split_label = "split0_val_1"
    else:           _split_label = "split0_val_2"
elif _sub == 0:
    _split_label = f"split{_main}_train"
else:
    _split_label = f"split{_main}_{_sub}_train"
os.makedirs(DESC_DIR, exist_ok=True)
INTERMEDIATE_CSV = os.path.join(DESC_DIR, f"coco_descs_{_split_label}.csv")

PREVIEW_MODE = False
PREVIEW_N    = 25

print(f"Device: {DEVICE}")


# %%
with open(KARPATHY_JSON, "r") as f:
    karpathy_data = json.load(f)

filename_to_path: Dict[str, str] = {}
for img_info in karpathy_data["images"]:
    filename = img_info["filename"]
    if "train2014" in filename:
        filename_to_path[filename] = os.path.join(TRAIN_DIR, filename)
    else:
        filename_to_path[filename] = os.path.join(VAL_DIR, filename)

print(f"Total images in Karpathy: {len(filename_to_path):,}")

# %%
def load_detection_for_split(split: str) -> Dict[str, Dict]:
    yolo_path  = os.path.join(DETECTION_DIR, "yolo",  f"coco_{split}_detection.csv")
    hoi_path   = os.path.join(DETECTION_DIR, "hoi",   f"coco_{split}_hoi_detection.csv")
    reltr_path = os.path.join(DETECTION_DIR, "reltr", f"coco_{split}_reltr_detection.csv")

    yolo_df  = pd.read_csv(yolo_path)  if os.path.exists(yolo_path)  else pd.DataFrame()
    hoi_df   = pd.read_csv(hoi_path)   if os.path.exists(hoi_path)   else pd.DataFrame()
    reltr_df = pd.read_csv(reltr_path) if os.path.exists(reltr_path) else pd.DataFrame()

    detection: Dict[str, Dict] = {}

    for _, row in yolo_df.iterrows():
        img = row["image"]
        detection[img] = {
            "yolo":  row["objects"] if pd.notna(row["objects"]) else "",
            "hoi":   [],
            "reltr": []
        }

    _BODY_PARTS = {
        'head', 'leg', 'legs', 'ear', 'ears', 'tail', 'mouth', 'nose', 'eye', 'eyes',
        'face', 'hand', 'hands', 'arm', 'arms', 'paw', 'hair', 'neck', 'foot', 'feet',
        'finger', 'fingers', 'wing', 'wings', 'toe', 'toes', 'teeth', 'tongue',
        'beak', 'claw', 'claws', 'forehead', 'chin', 'cheek', 'elbow', 'knee',
        'shoulder', 'wrist', 'ankle', 'hip', 'chest', 'stomach', 'back', 'waist', 'thumb',
    }
    _TRIVIAL_RELS  = {'has', 'of', 'part of', 'belong to', 'belonging to'}
    _USEFUL_RELS   = {
        'behind', 'in', 'near', 'under', 'above', 'between', 'in front of',
        'next to', 'beside', 'along', 'across', 'around', 'inside',
        'holding', 'riding', 'sitting on', 'standing on', 'walking on',
        'carrying', 'watching', 'looking at', 'laying on', 'lying on',
        'using', 'eating', 'playing', 'covered in', 'mounted on',
        'parked on', 'hanging from', 'attached to', 'sitting in',
        'standing in', 'walking in', 'flying in', 'growing on',
    }
    _PERSON_WORDS  = {'man', 'boy', 'woman', 'girl', 'person', 'people'}
    _USEFUL_WEAR   = {
        'hat', 'helmet', 'backpack', 'glasses', 'sunglasses', 'mask',
        'jacket', 'coat', 'dress', 'uniform', 'suit', 'tie',
        'shirt', 'short', 'shorts', 'pant', 'pants', 'jean', 'jeans',
        'shoe', 'shoes', 'boot', 'boots', 'glove', 'gloves',
        'skirt', 'sweater', 'hoodie', 'vest', 'scarf', 'apron',
        'jersey', 'cap', 'beanie', 'bandana', 'goggles',
    }

    def _keep_reltr(pred: str, score: float) -> bool:
        parts = pred.split(' - ')
        if len(parts) != 3:
            return False
        subj, rel, obj = parts[0].lower(), parts[1].lower(), parts[2].lower()
        if subj in _BODY_PARTS and obj in _BODY_PARTS:
            return False
        if rel in _TRIVIAL_RELS:
            if subj in _BODY_PARTS or obj in _BODY_PARTS:
                return False
            return score > 0.65
        if rel == 'on':
            if obj in _BODY_PARTS or obj in _PERSON_WORDS or subj in _BODY_PARTS:
                return False
        if rel in _USEFUL_RELS:
            return True
        if rel == 'wearing':
            return obj in _USEFUL_WEAR or score > 0.60
        return score > 0.55

    for _, row in hoi_df.iterrows():
        img  = row["image"]
        pred = str(row["prediction"])
        if "no_interaction" in pred or "no_relation" in pred or float(row["score"]) < 0.4:
            continue
        if img not in detection:
            detection[img] = {"yolo": "", "hoi": [], "reltr": []}
        action = pred.replace("Human - ", "").replace("_", " ").strip()
        detection[img]["hoi"].append((action, float(row["score"])))

    for _, row in reltr_df.iterrows():
        img  = row["image"]
        pred = str(row["prediction"])
        if pred == "no_relation" or float(row["score"]) < 0.4:
            continue
        if not _keep_reltr(pred, float(row["score"])):
            continue
        if img not in detection:
            detection[img] = {"yolo": "", "hoi": [], "reltr": []}
        detection[img]["reltr"].append((pred, float(row["score"])))

    for img in detection:
        seen = set()
        hoi_sorted = sorted(detection[img]["hoi"], key=lambda x: x[1], reverse=True)
        hoi_dedup  = []
        for action, _ in hoi_sorted:
            if action not in seen:
                seen.add(action)
                hoi_dedup.append(action)
        detection[img]["hoi"] = hoi_dedup

        seen = set()
        rel_sorted = sorted(detection[img]["reltr"], key=lambda x: x[1], reverse=True)
        rel_dedup  = []
        for pred, _ in rel_sorted:
            if pred not in seen:
                seen.add(pred)
                rel_dedup.append(pred)
        detection[img]["reltr"] = rel_dedup

    print(f"  [{split:5s}] {len(detection):,} images")
    return detection

print("Loading detection CSVs...")
train_detection = load_detection_for_split("train")
val_detection   = load_detection_for_split("val")
test_detection  = load_detection_for_split("test")
all_detection   = {**train_detection, **val_detection, **test_detection}

train_imgs_sorted = sorted([img for img in train_detection if img in filename_to_path])
test_imgs         = [img for img in test_detection if img in filename_to_path]
val_imgs          = [img for img in val_detection  if img in filename_to_path]

if _main == 0:
    if _sub in [1, 2]:
        pool = sorted(test_imgs)
        mid  = len(pool) // 2
        if _sub == 1:
            all_images = pool[:mid]
            print(f"Split 0.1 (test first half): {len(all_images):,} images")
        else:
            all_images = pool[mid:]
            print(f"Split 0.2 (test second half): {len(all_images):,} images")
    else:
        pool = sorted(val_imgs)
        mid  = len(pool) // 2
        if _sub == 3:
            all_images = pool[:mid]
            print(f"Split 0.3 (val first half): {len(all_images):,} images")
        else:
            all_images = pool[mid:]
            print(f"Split 0.4 (val second half): {len(all_images):,} images")
else:
    total     = len(train_imgs_sorted)
    part_size = (total + 11) // 12
    start     = (_main - 1) * part_size
    end       = min(_main * part_size, total)
    mid       = start + (end - start) // 2
    if _sub == 1:
        all_images = train_imgs_sorted[start:mid]
        print(f"Split {_main}.1 (train part {_main}/12, first half): images [{start:,}-{mid:,}), total {len(all_images):,} images")
    else:
        all_images = train_imgs_sorted[mid:end]
        print(f"Split {_main}.2 (train part {_main}/12, second half): images [{mid:,}-{end:,}), total {len(all_images):,} images")

if PREVIEW_MODE:
    all_images = all_images[:PREVIEW_N]
print(f"Total images to generate descriptions for: {len(all_images):,}")

# %%
print(f"\nLoading Qwen: {QWEN_MODEL_ID}...")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": DEVICE},
)
print("Qwen loaded!")


def build_desc_prompt(batch_images: List[str], detection: Dict) -> str:
    lines = []
    for local_id, img in enumerate(batch_images):
        det       = detection.get(img, {"yolo": "", "hoi": [], "reltr": []})
        yolo_str  = det["yolo"]                          if det["yolo"]   else "none"
        hoi_str   = "; ".join(det["hoi"][:K_PROMPT])     if det["hoi"]    else "none"
        reltr_str = "; ".join(det["reltr"][:K_PROMPT])   if det["reltr"]  else "none"
        lines.append(f"Image {local_id}:")
        lines.append(f"  Objects  : {yolo_str}")
        lines.append(f"  Actions  : {hoi_str}")
        lines.append(f"  Relations: {reltr_str}")

    return (
        "For image below, generate exactly 5 semantic descriptions. "
        "Each description must contain more than 5 words, be specific, diverse, and faithful to the visual context. \n\n"
        + "\n".join(lines)
        + "\n\nOUTPUT FORMAT (strictly follow, one line per image):\n"
        + "0: description1 | description2 | description3 | description4 | description5\n"
        + "1: description1 | description2 | description3 | description4 | description5\n"
        + "...\n"
        + "Output ONLY the numbered lines, nothing else."
    )


def call_qwen(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise image scene describer. Follow the output format exactly."},
        {"role": "user",   "content": prompt}
    ]
    inputs = qwen_tokenizer.apply_chat_template(
        messages, return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids      = (inputs.input_ids if hasattr(inputs, "input_ids") else inputs).to(qwen_model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = qwen_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id
        )
    return qwen_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)


def parse_descs(response: str, batch_size: int) -> List[List[str]]:
    result = [[""] * 5 for _ in range(batch_size)]
    for line in response.strip().split("\n"):
        line = line.strip()
        m = re.match(r'^(\d+)\s*[:\.]?\s*(.+)$', line)
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= batch_size:
            continue
        raw_descs = [t.strip() for t in m.group(2).split("|") if t.strip()]
        for i in range(min(5, len(raw_descs))):
            result[idx][i] = raw_descs[i]
    return result


intermediate_rows = []
for batch_start in tqdm(range(0, len(all_images), QWEN_BATCH_SIZE), desc="Generating descriptions (Qwen)"):
    batch  = all_images[batch_start: batch_start + QWEN_BATCH_SIZE]
    prompt = build_desc_prompt(batch, all_detection)
    descs_batch = None
    for attempt in range(3):
        try:
            torch.cuda.empty_cache()
            response    = call_qwen(prompt)
            descs_batch = parse_descs(response, len(batch))
            break
        except Exception as e:
            print(f"Qwen error at batch {batch_start} (attempt {attempt+1}): {e}")
            torch.cuda.empty_cache()
            if attempt == 2:
                descs_batch = [[""] * 5 for _ in batch]

    for img, descs in zip(batch, descs_batch):
        row = {"image": img}
        for j, desc in enumerate(descs):
            row[f"desc_{j+1}"] = desc
        intermediate_rows.append(row)

    if (batch_start // QWEN_BATCH_SIZE + 1) % 500 == 0:
        pd.DataFrame(intermediate_rows).to_csv(INTERMEDIATE_CSV, index=False)
        print(f"  Checkpoint saved at batch {batch_start}")

intermediate_df = pd.DataFrame(intermediate_rows)
intermediate_df.to_csv(INTERMEDIATE_CSV, index=False)
print(f"Saved intermediate descriptions: {len(intermediate_df):,} images -> {INTERMEDIATE_CSV}")

del qwen_model, qwen_tokenizer
torch.cuda.empty_cache()
print("Qwen has been released.")
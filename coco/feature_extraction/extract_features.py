import json
import math
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, CLIPVisionModel

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)


# 1.1 / 1.2 / 1.3 / 1.4 / 1.5 / 1.6  →  train part 1-6
# 2                                  →  test
# 3                                  →  val
PART = "1.2"

BATCH_SIZE = 256
OUTPUT_DIR = "data/coco/features"

KARPATHY_JSON  = "data/coco/dataset_coco.json"
TRAIN_DIR      = "data/coco/train2014"
VAL_DIR        = "data/coco/val2014"
CLIP_MODEL_ID  = "openai/clip-vit-large-patch14"


class CLIPFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        outputs = self.clip(pixel_values=image)
        return outputs.last_hidden_state[:, 1:, :]


NUM_WORKERS = 4


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            pixel = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        except Exception:
            pixel = torch.zeros(3, 224, 224)
        return pixel, path


def get_image_path(img_info: dict) -> str:
    fn = img_info["filename"]
    return os.path.join(TRAIN_DIR if "train2014" in fn else VAL_DIR, fn)


def load_split_paths() -> Dict[str, List[str]]:
    with open(KARPATHY_JSON) as f:
        data = json.load(f)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    seen: Dict[str, set] = {k: set() for k in splits}

    for img_info in data["images"]:
        path = get_image_path(img_info)
        split = img_info["split"]
        key = "train" if split in ("train", "restval") else split
        if key in splits and path not in seen[key]:
            splits[key].append(path)
            seen[key].add(path)
    return splits


def parse_part(part: str) -> Tuple[str, Optional[int]]:
    if part.startswith("1."):
        idx = int(part.split(".")[1]) - 1
        assert 0 <= idx <= 5, f"Train part must be 1.1~1.6, got: {part}"
        return "train", idx
    elif part == "2":
        return "test", None
    elif part == "3":
        return "val", None
    raise ValueError(f"Invalid PART: {part}")


def chunk_list(lst: list, n: int) -> List[list]:
    size = math.ceil(len(lst) / n)
    return [lst[i * size: (i + 1) * size] for i in range(n)]


def get_h5_path(out_dir: Path, split: str, chunk_idx: Optional[int]) -> Path:
    if chunk_idx is not None:
        return out_dir / f"train_part{chunk_idx + 1}.h5"
    return out_dir / f"{split}.h5"


split, chunk_idx = parse_part(PART)
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

all_splits  = load_split_paths()
image_paths = all_splits[split]

if chunk_idx is not None:
    image_paths = chunk_list(image_paths, 6)[chunk_idx]
    label = f"train part {chunk_idx + 1}/6"
else:
    label = split

h5_path = get_h5_path(out_dir, split, chunk_idx)

existing_keys: set = set()
if h5_path.exists():
    with h5py.File(h5_path, "r") as f:
        existing_keys = set(f.keys())

pending = [p for p in image_paths if os.path.basename(p) not in existing_keys]

print(f"Split: {label} | Total: {len(image_paths)} | Already done: {len(image_paths)-len(pending)} | Remaining: {len(pending)}")
print(f"Output HDF5: {h5_path}")

if pending:
    extractor = CLIPFeatureExtractor().to(device).eval()
    processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID)

    dataset = ImagePathDataset(pending, processor)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
        prefetch_factor=2,
    )

    errors: List[str] = []

    with h5py.File(h5_path, "a") as h5f:
        for pixel_values, paths in tqdm(loader, desc=label, unit="batch"):
            pixel_values = pixel_values.to(device, non_blocking=True)
            features = extractor(pixel_values).cpu()

            for i, img_path in enumerate(paths):
                if pixel_values[i].sum().item() == 0.0 and features[i].abs().max().item() < 1e-6:
                    tqdm.write(f"[WARN] skipping bad image: {img_path}")
                    errors.append(img_path)
                    continue
                key = os.path.basename(img_path)
                h5f.create_dataset(key, data=features[i].numpy(),
                                   compression="gzip", compression_opts=5, shuffle=True)

    print(f"Done: {len(pending)-len(errors)} images → {h5_path}")
    if errors:
        err_log = out_dir / f"errors_part{PART.replace('.', '_')}.txt"
        err_log.write_text("\n".join(errors))
        print(f"Errors ({len(errors)} images): {err_log}")
else:
    print("All already extracted. Done.")
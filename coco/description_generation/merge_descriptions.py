# %%
import pandas as pd
import os

# %%
INPUT_DIR  = "data/coco/descriptions"
OUTPUT_DIR = "data/coco/descriptions"

SINGLE_GROUPS = {
    "test": {
        "files": [
            "coco_descs_split0_test_1.csv",
            "coco_descs_split0_test_2.csv",
        ],
        "output": os.path.join(OUTPUT_DIR, "coco_descs_test_merged.csv"),
    },
    "val": {
        "files": [
            "coco_descs_split0_val_1.csv",
            "coco_descs_split0_val_2.csv",
        ],
        "output": os.path.join(OUTPUT_DIR, "coco_descs_val_merged.csv"),
    },
}

GROUPS = {
    "part_A": [
        "coco_descs_split1_1_train.csv",
        "coco_descs_split1_2_train.csv",
        "coco_descs_split2_1_train.csv",
        "coco_descs_split2_2_train.csv",
        "coco_descs_split3_1_train.csv",
        "coco_descs_split3_2_train.csv",
        "coco_descs_split4_1_train.csv",
        "coco_descs_split4_2_train.csv",
    ],
    "part_B": [
        "coco_descs_split5_1_train.csv",
        "coco_descs_split5_2_train.csv",
        "coco_descs_split6_1_train.csv",
        "coco_descs_split6_2_train.csv",
        "coco_descs_split7_1_train.csv",
        "coco_descs_split7_2_train.csv",
        "coco_descs_split8_1_train.csv",
        "coco_descs_split8_2_train.csv",
    ],
    "part_C": [
        "coco_descs_split9_1_train.csv",
        "coco_descs_split9_2_train.csv",
        "coco_descs_split10_1_train.csv",
        "coco_descs_split10_2_train.csv",
        "coco_descs_split11_1_train.csv",
        "coco_descs_split11_2_train.csv",
        "coco_descs_split12_1_train.csv",
        "coco_descs_split12_2_train.csv",
    ],
}

OUTPUT_FILES = {
    "part_A": os.path.join(OUTPUT_DIR, "coco_descs_train_merged_A.csv"),
    "part_B": os.path.join(OUTPUT_DIR, "coco_descs_train_merged_B.csv"),
    "part_C": os.path.join(OUTPUT_DIR, "coco_descs_train_merged_C.csv"),
}

# %%
all_files = (
    [f for g in SINGLE_GROUPS.values() for f in g["files"]]
    + [f for files in GROUPS.values() for f in files]
)
missing = [f for f in all_files if not os.path.exists(os.path.join(INPUT_DIR, f))]
if missing:
    raise FileNotFoundError(
        "Missing the following CSV files:\n"
        + "\n".join(f"  - {f}" for f in missing)
    )

# %%
for split_name, cfg in SINGLE_GROUPS.items():
    print(f"\n=== {split_name} ===")
    parts = []
    for fname in cfg["files"]:
        csv_path = os.path.join(INPUT_DIR, fname)
        df = pd.read_csv(csv_path)
        parts.append(df)
        print(f"  {fname}: {len(df):,} images")

    merged = pd.concat(parts, ignore_index=True)

    n_dup = merged["image"].duplicated().sum()
    if n_dup > 0:
        print(f"  [WARNING] {n_dup} duplicate images found!")
    else:
        print(f"  No duplicate images.")

    merged.to_csv(cfg["output"], index=False)
    print(f"  Merged {len(merged):,} images -> {cfg['output']}")
    print(f"  Columns: {list(merged.columns)}")

# %%
for group_name, file_list in GROUPS.items():
    print(f"\n=== Group {group_name} ===")
    parts = []
    for fname in file_list:
        csv_path = os.path.join(INPUT_DIR, fname)
        df = pd.read_csv(csv_path)
        parts.append(df)
        print(f"  {fname}: {len(df):,} images")

    merged = pd.concat(parts, ignore_index=True)

    n_dup = merged["image"].duplicated().sum()
    if n_dup > 0:
        print(f"  [WARNING] {n_dup} duplicate images found!")
    else:
        print(f"  No duplicate images.")

    out_path = OUTPUT_FILES[group_name]
    merged.to_csv(out_path, index=False)
    print(f"  Merged {len(merged):,} images -> {out_path}")
    print(f"  Columns: {list(merged.columns)}")

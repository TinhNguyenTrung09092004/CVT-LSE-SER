# %%
import pandas as pd
import os

# %%
INPUT_DIR  = "data/flickr30k/descriptions"
OUTPUT_DIR = "data/flickr30k/descriptions"

GROUPS = {
    "part_A": [
        "flickr30k_descs_split1_train.csv",
        "flickr30k_descs_split2_train.csv",
        "flickr30k_descs_split3_train.csv",
    ],
    "part_B": [
        "flickr30k_descs_split4_train.csv",
        "flickr30k_descs_split5_train.csv",
        "flickr30k_descs_split6_train.csv",
    ],
    "part_C": [
        "flickr30k_descs_split7_train.csv",
        "flickr30k_descs_split8_train.csv",
        "flickr30k_descs_split9_train.csv",
        "flickr30k_descs_split10_train.csv",
    ],
}

OUTPUT_FILES = {
    "part_A": os.path.join(OUTPUT_DIR, "flickr30k_descs_train_merged_A.csv"),
    "part_B": os.path.join(OUTPUT_DIR, "flickr30k_descs_train_merged_B.csv"),
    "part_C": os.path.join(OUTPUT_DIR, "flickr30k_descs_train_merged_C.csv"),
}

# %%
all_files = [f for files in GROUPS.values() for f in files]
missing = [f for f in all_files if not os.path.exists(os.path.join(INPUT_DIR, f))]
if missing:
    raise FileNotFoundError(
        "Missing the following CSV files:\n"
        + "\n".join(f"  - {f}" for f in missing)
    )

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

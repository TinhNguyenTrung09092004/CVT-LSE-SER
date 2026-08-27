# %%
import pandas as pd

SCORED_A = "data/flickr30k/descriptions/flickr30k_descs_train_A_scored.csv"
SCORED_B = "data/flickr30k/descriptions/flickr30k_descs_train_B_scored.csv"
SCORED_C = "data/flickr30k/descriptions/flickr30k_descs_train_C_scored.csv"
OUTPUT   = "data/flickr30k/descriptions/flickr30k_descs_train_scored.csv"

# %%
dfs = []
for path, label in [(SCORED_A, "A"), (SCORED_B, "B"), (SCORED_C, "C")]:
    df = pd.read_csv(path)
    print(f"Set {label}: {len(df):,} images")
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
print(f"\nTotal after merge: {len(merged):,} images")

# %%
merged.to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}")

print("\n=== STATISTICS ===")
print(f"Images with best_score > 0.5: {(merged['best_score'] > 0.5).sum():,}")
print(f"Images with best_score > 0.3: {(merged['best_score'] > 0.3).sum():,}")
print(f"Average best score:   {merged['best_score'].mean():.4f}")
print(f"\n10 sample images:")
print(merged[["image", "best_desc", "best_score"]].head(10).to_string(index=False))

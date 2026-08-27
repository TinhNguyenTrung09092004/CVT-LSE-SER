# %%
# !pip install -q pandas numpy
# !pip install -q git+https://github.com/salaniz/pycocoevalcap.git

# %%
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# %%
@dataclass
class TestConfig:
    METHOD_A_CSV: str = "outputs/coco/coco_llm_rerank_selections.csv"
    METHOD_A_LABEL: str = "CVT-LSE-SER (LLM rerank)"

    METHOD_B_CSV: str = "outputs/coco/beam_candidates_epoch_<EPOCH>.csv"
    METHOD_B_LABEL: str = "CVT-LSE (beam top-1)"

    N_BOOTSTRAP: int = 2000
    BLOCK: int = 200
    seed: int = 0


test_config = TestConfig()

assert test_config.METHOD_A_CSV != test_config.METHOD_B_CSV, (
    "METHOD_A_CSV and METHOD_B_CSV point at the same file - a method cannot be "
    "compared against itself."
)
assert shutil.which("java") is not None, (
    "java not found - PTBTokenizer and METEOR both require a java runtime"
)

METRIC_NAMES = ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr"]
BLEU_NAMES = ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]
MEAN_NAMES = ["ROUGE_L", "CIDEr"]

# %%
@dataclass
class Record:
    image_id: int
    caption: str
    references: List[str]


def reference_columns(columns: Sequence[str]) -> List[str]:
    matched = []
    for col in columns:
        match = re.fullmatch(r"ground_truth_(\d+)", str(col))
        if match:
            matched.append((int(match.group(1)), col))
    return [col for _, col in sorted(matched)]


def read_references(row: pd.Series, columns: Sequence[str]) -> List[str]:
    refs = []
    for col in columns:
        value = row[col]
        if pd.notna(value) and str(value).strip():
            refs.append(str(value).strip())
    return refs


def detect_format(df: pd.DataFrame) -> str:
    if "selected_caption" in df.columns:
        return "selections"
    if "candidate_rank" in df.columns and "caption" in df.columns:
        return "candidates"
    raise ValueError(
        "Unrecognised CSV layout: expected either 'selected_caption' or "
        f"('candidate_rank', 'caption'). Columns present: {list(df.columns)}"
    )


def load_records(csv_path: str, label: str) -> Dict[int, Record]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found for {label}: {csv_path}")

    df = pd.read_csv(csv_path)
    layout = detect_format(df)
    ref_cols = reference_columns(df.columns)
    if not ref_cols:
        raise ValueError(f"No ground_truth_* columns in {csv_path}")

    caption_col = "selected_caption" if layout == "selections" else "caption"
    valid = df[df[caption_col].notna() & (df[caption_col].astype(str).str.strip() != "")]
    n_no_caption = df["image_id"].nunique() - valid["image_id"].nunique()

    records: Dict[int, Record] = {}
    n_no_reference = 0
    ranks_taken = []

    for image_id, group in valid.groupby("image_id"):
        if layout == "candidates":
            chosen = group.sort_values("candidate_rank").iloc[0]
            ranks_taken.append(int(chosen["candidate_rank"]))
        else:
            chosen = group.iloc[0]
        references = read_references(chosen, ref_cols)
        if not references:
            n_no_reference += 1
            continue
        records[int(image_id)] = Record(
            image_id=int(image_id),
            caption=str(chosen[caption_col]).strip(),
            references=references,
        )

    if not records:
        raise ValueError(f"No scorable rows in {csv_path}")

    print(f"{label}")
    print(f"  file   : {csv_path}")
    print(f"  layout : {layout}")
    if layout == "candidates":
        print(f"  taking the best available candidate_rank per image "
              f"(ranks used: {min(ranks_taken)}-{max(ranks_taken)})")
    print(f"  images : {len(records)}")
    if n_no_caption:
        print(f"  skipped {n_no_caption} images with an empty caption")
    if n_no_reference:
        print(f"  skipped {n_no_reference} images without ground truth")

    return records


records_a = load_records(test_config.METHOD_A_CSV, test_config.METHOD_A_LABEL)
print()
records_b = load_records(test_config.METHOD_B_CSV, test_config.METHOD_B_LABEL)

# %%
def normalize_reference(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).strip()


shared_ids = sorted(set(records_a) & set(records_b))

print("\n" + "=" * 78)
print("PAIRING")
print("=" * 78)
print(f"  {test_config.METHOD_A_LABEL}: {len(records_a)} images")
print(f"  {test_config.METHOD_B_LABEL}: {len(records_b)} images")
print(f"  paired (intersection)  : {len(shared_ids)} images")
print(f"  dropped from A         : {len(records_a) - len(shared_ids)}")
print(f"  dropped from B         : {len(records_b) - len(shared_ids)}")

if not shared_ids:
    raise ValueError("No overlapping image_id between the two CSVs - the methods cannot be paired")

reference_mismatch = []
for image_id in shared_ids:
    set_a = {normalize_reference(r) for r in records_a[image_id].references}
    set_b = {normalize_reference(r) for r in records_b[image_id].references}
    if set_a != set_b:
        reference_mismatch.append(image_id)

if not reference_mismatch:
    print("\n  ground truth identical across both files")

identical_caption = sum(
    records_a[i].caption.strip().lower() == records_b[i].caption.strip().lower()
    for i in shared_ids
)
print(f"  identical captions     : {identical_caption}/{len(shared_ids)} "
      f"({identical_caption / len(shared_ids) * 100:.2f}%)")

assert identical_caption < len(shared_ids), (
    "Every caption is identical across the two files - the two CSVs describe the same "
    "system, so there is nothing to test."
)

references = {image_id: records_a[image_id].references for image_id in shared_ids}
captions_a = {image_id: records_a[image_id].caption for image_id in shared_ids}
captions_b = {image_id: records_b[image_id].caption for image_id in shared_ids}

# %%
@dataclass
class MethodStats:
    label: str
    corpus: Dict[str, float]
    guess: np.ndarray
    correct: np.ndarray
    testlen: np.ndarray
    reflen: np.ndarray
    per_image: Dict[str, np.ndarray]
    meteor_stats: List[str]


def bleu_from_stats(stats: MethodStats, idx: np.ndarray) -> np.ndarray:
    guess = stats.guess[idx].sum(axis=1).astype(float)
    correct = stats.correct[idx].sum(axis=1).astype(float)
    testlen = stats.testlen[idx].sum(axis=1).astype(float)
    reflen = stats.reflen[idx].sum(axis=1).astype(float)

    ratios = np.cumprod((correct + 1e-15) / (guess + 1e-9), axis=1)
    bleus = ratios ** (1.0 / np.arange(1, 5, dtype=float))

    length_ratio = (testlen + 1e-15) / (reflen + 1e-9)
    penalty = np.where(length_ratio < 1.0, np.exp(1.0 - 1.0 / length_ratio), 1.0)
    return bleus * penalty[:, None]


class PairedScorer:
    def __init__(self, image_ids: Sequence[int], references: Dict[int, List[str]]):
        self.image_ids = list(image_ids)
        self.keys = [str(i) for i in self.image_ids]
        self.tokenizer = PTBTokenizer()
        self.rouge = Rouge()
        self.cider = Cider()
        self.meteor = Meteor()
        self.meteor_text_mode = hasattr(self.meteor.meteor_p.stdin, "encoding")

        raw_gts = {str(i): [{"caption": r} for r in references[i]] for i in self.image_ids}
        tokenized = self.tokenizer.tokenize(raw_gts)
        self.gts = {k: tokenized[k] for k in self.keys}

    def tokenize_captions(self, captions: Dict[int, str]) -> Dict[str, List[str]]:
        raw = {str(i): [{"caption": captions[i]}] for i in self.image_ids}
        tokenized = self.tokenizer.tokenize(raw)
        return {k: tokenized[k] for k in self.keys}

    def meteor_aggregate(self, stats: Sequence[str]) -> float:
        stdin = self.meteor.meteor_p.stdin
        stdout = self.meteor.meteor_p.stdout
        line = "EVAL ||| " + " ||| ".join(stats) + "\n"
        stdin.write(line if self.meteor_text_mode else line.encode())
        stdin.flush()
        for _ in range(len(stats)):
            stdout.readline()
        aggregate = stdout.readline()
        return float(aggregate if self.meteor_text_mode else aggregate.decode())

    def analyse(self, captions: Dict[int, str], label: str) -> MethodStats:
        res = self.tokenize_captions(captions)
        n = len(self.keys)

        bleu_scorer = BleuScorer(n=4)
        for key in self.keys:
            bleu_scorer += (res[key][0], self.gts[key])
        corpus_bleu, _ = bleu_scorer.compute_score(option="closest", verbose=0)

        rouge_l, rouge_per = self.rouge.compute_score(self.gts, res)
        cider, cider_per = self.cider.compute_score(self.gts, res)
        meteor, _ = self.meteor.compute_score(self.gts, res)

        corpus = {
            "BLEU_1": float(corpus_bleu[0]),
            "BLEU_2": float(corpus_bleu[1]),
            "BLEU_3": float(corpus_bleu[2]),
            "BLEU_4": float(corpus_bleu[3]),
            "METEOR": float(meteor),
            "ROUGE_L": float(rouge_l),
            "CIDEr": float(cider),
        }

        guess = np.zeros((n, 4), dtype=np.int64)
        correct = np.zeros((n, 4), dtype=np.int64)
        testlen = np.zeros(n, dtype=np.int64)
        reflen = np.zeros(n, dtype=np.int64)
        for position, comps in enumerate(bleu_scorer.ctest):
            length = comps["testlen"]
            testlen[position] = length
            reflen[position] = min((abs(l - length), l) for l in comps["reflen"])[1]
            guess[position] = comps["guess"]
            correct[position] = comps["correct"]

        per_image = {
            "ROUGE_L": np.asarray(rouge_per, dtype=float),
            "CIDEr": np.asarray(cider_per, dtype=float),
        }
        meteor_stats = [self.meteor._stat(res[key][0], self.gts[key]) for key in self.keys]

        stats = MethodStats(
            label=label,
            corpus=corpus,
            guess=guess,
            correct=correct,
            testlen=testlen,
            reflen=reflen,
            per_image=per_image,
            meteor_stats=meteor_stats,
        )

        full = np.arange(n)[None, :]
        rebuilt_bleu = bleu_from_stats(stats, full)[0]
        assert np.allclose(rebuilt_bleu, [corpus[name] for name in BLEU_NAMES], atol=1e-9), (
            f"[{label}] BLEU sufficient statistics do not reproduce the corpus BLEU "
            f"({rebuilt_bleu} vs {[corpus[name] for name in BLEU_NAMES]})"
        )
        for name in MEAN_NAMES:
            assert abs(float(per_image[name].mean()) - corpus[name]) < 1e-9, (
                f"[{label}] per-image {name} mean does not equal the corpus {name}"
            )
        rebuilt_meteor = self.meteor_aggregate(meteor_stats)
        assert abs(rebuilt_meteor - corpus["METEOR"]) < 1e-6, (
            f"[{label}] METEOR sufficient statistics do not reproduce the corpus METEOR "
            f"({rebuilt_meteor} vs {corpus['METEOR']})"
        )
        return stats


scorer = PairedScorer(shared_ids, references)
stats_a = scorer.analyse(captions_a, test_config.METHOD_A_LABEL)
stats_b = scorer.analyse(captions_b, test_config.METHOD_B_LABEL)

# %%
n_images = len(shared_ids)
rng = np.random.default_rng(test_config.seed)
collected: Dict[str, List[np.ndarray]] = {name: [] for name in METRIC_NAMES}

print("\n" + "=" * 78)
print(f"PAIRED BOOTSTRAP  ({test_config.N_BOOTSTRAP} resamples of {n_images} images)")
print("=" * 78)

done = 0
while done < test_config.N_BOOTSTRAP:
    size = min(test_config.BLOCK, test_config.N_BOOTSTRAP - done)
    idx = rng.integers(0, n_images, size=(size, n_images))

    bleu_a = bleu_from_stats(stats_a, idx)
    bleu_b = bleu_from_stats(stats_b, idx)
    for order, name in enumerate(BLEU_NAMES):
        collected[name].append(bleu_a[:, order] - bleu_b[:, order])

    for name in MEAN_NAMES:
        collected[name].append(
            stats_a.per_image[name][idx].mean(axis=1) - stats_b.per_image[name][idx].mean(axis=1)
        )

    meteor_block = np.empty(size, dtype=float)
    for position in range(size):
        draw = idx[position]
        meteor_block[position] = (
            scorer.meteor_aggregate([stats_a.meteor_stats[j] for j in draw])
            - scorer.meteor_aggregate([stats_b.meteor_stats[j] for j in draw])
        )
    collected["METEOR"].append(meteor_block)

    done += size
    print(f"  {done}/{test_config.N_BOOTSTRAP} resamples")

# %%
results: Dict[str, Dict[str, object]] = {}

for name in METRIC_NAMES:
    values = np.concatenate(collected[name])
    observed = stats_a.corpus[name] - stats_b.corpus[name]

    ci_low = float(np.percentile(values, 2.5))
    ci_high = float(np.percentile(values, 97.5))

    n_at_or_below = int(np.sum(values <= 0.0))
    n_at_or_above = int(np.sum(values >= 0.0))
    p_value = min(1.0, 2.0 * (min(n_at_or_below, n_at_or_above) + 1) / (len(values) + 1))

    results[name] = {
        "corpus_a": stats_a.corpus[name],
        "corpus_b": stats_b.corpus[name],
        "observed_diff": float(observed),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_value),
    }

# %%
print("\n" + "=" * 78)
print(f"RESULTS  (n={n_images} paired images, x100, higher is better)")
print("=" * 78)
print(f"  A = {test_config.METHOD_A_LABEL}")
print(f"  B = {test_config.METHOD_B_LABEL}")
print(f"  delta = A - B, {test_config.N_BOOTSTRAP} bootstrap resamples, seed {test_config.seed}")

table_rows = []
for name in METRIC_NAMES:
    entry = results[name]
    table_rows.append({
        "metric": name,
        "A": round(entry["corpus_a"] * 100, 2),
        "B": round(entry["corpus_b"] * 100, 2),
        "delta": round(entry["observed_diff"] * 100, 2),
        "95%_CI": f"[{entry['ci_low'] * 100:+.2f}, {entry['ci_high'] * 100:+.2f}]",
        "p": f"{entry['p_value']:.4g}",
    })

print()
print(pd.DataFrame(table_rows).to_string(index=False))
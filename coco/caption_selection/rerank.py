# %%
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# %%
# !pip install -q transformers accelerate bitsandbytes
# !pip install -q nltk pandas
# !pip install -q git+https://github.com/salaniz/pycocoevalcap.git
# !pip install -q sacremoses

# %%
import torch
import pandas as pd
import numpy as np
import importlib.util
import os
import re
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# %%
@dataclass
class Config:
    BEAM_WIDTH: int = 5
    LLM_BATCH_SIZE: int = 1
    K_PROMPT: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

config = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


set_seed(config.seed)

# %%
class BaseLLMRanker(ABC):
    def __init__(self):
        self.invalid_lines: List[str] = []
        self._sample_prompt_printed = False

    @abstractmethod
    def quick_chat(self, message: str, system: str = "You are a helpful assistant.") -> str:
        pass

    def _build_rerank_prompt(self, images_section: List[str], context_section: str = "") -> str:
        if context_section.strip():
            context_block = f"""DETECTION EVIDENCE (3 vision models on the actual image):
{context_section}

EVIDENCE GUIDE:
- Objects (YOLO, conf >= 0.6 — HIGHLY RELIABLE): Exact COCO object categories with counts.
  If YOLO says "motorcycle" → prefer "motorcycle"/"motorbike"; penalize "bike"/"vehicle".
  If YOLO says "bicycle" → prefer "bicycle"/"bike"; penalize "motorcycle".
  If YOLO says "tennis racket" → prefer that over "racket"/"paddle"/"object".
- Actions (HOI, conf >= 0.40 — SUPPORTING): Detected human-object interaction verbs.
  Use to confirm or reject the human activity described.
- Relations (RelTR, conf >= 0.40 — SUPPORTING): Subject-relation-object scene graph triplets.
  Use to verify spatial arrangements and secondary activities.
- Scene descs (BLIP-scored Qwen synthesis — CONTEXTUAL): Full semantic sentences
  generated from combined detection signals; score reflects visual grounding by BLIP.
  Higher score = more reliable. Use to confirm overall scene and activity when
  YOLO/HOI/RelTR evidence alone is ambiguous (e.g., "A person riding a bicycle on the road",
  "A kitchen with sink and countertop"). Do NOT override high-confidence YOLO object
  detections with desc wording.

SELECTION RULES (apply in strict priority order):
1. NO HALLUCINATIONS: Strongly penalize captions that mention objects clearly absent from the Objects list.
2. CORRECT OBJECT LABEL: Prefer captions using the exact detected category or its direct synonym over vague alternatives.
3. CORRECT ACTION: If Actions evidence shows a clear interaction, prefer captions matching that human activity.
4. CORRECT SPATIAL CONTEXT: Use Relations to confirm described spatial arrangements and activities.
5. SCENE TYPE MATCH: Use Scene descs to confirm overall scene type; prefer captions whose general theme matches the highest-scored desc.
6. NO <unk> TOKENS: Discard any caption containing "<unk>" or broken phrasing."""
        else:
            context_block = """No detection evidence available.
Select the most fluent and specific caption."""

        return f"""IMAGE CAPTION RERANKING TASK

Select the SINGLE BEST caption for each image using the visual detection evidence.

{context_block}

CAPTIONS TO EVALUATE:

{chr(10).join(images_section)}

OUTPUT FORMAT: Respond with ONLY image ID and selected caption number, one per line.
Format: IMAGE_ID: NUMBER

Example:
0: 2
1: 3"""

    def _record_invalid(self, line: str, reason: str):
        self.invalid_lines.append(f"{reason}: {line!r}")

    def _parse_rerank_response(
        self,
        response: str,
        shuffle_mappings: Dict[int, Dict[int, int]],
        id_mapping: Dict[int, int]
    ) -> Dict[int, int]:
        results = {}
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                parts = line.split(":", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) != 2:
                self._record_invalid(line, "expected 'IMAGE_ID: NUMBER'")
                continue

            img_id_str = parts[0].strip().replace("Image", "").replace("image", "").strip()
            if not img_id_str.isdigit():
                self._record_invalid(line, "non-numeric image id")
                continue
            img_id = int(img_id_str)
            if img_id not in shuffle_mappings:
                self._record_invalid(line, f"unknown image id {img_id}")
                continue

            number_match = re.search(r'\d+', parts[1].strip())
            if not number_match:
                self._record_invalid(line, "no caption number")
                continue

            shuffled_idx = int(number_match.group()) - 1
            if not 0 <= shuffled_idx < len(shuffle_mappings[img_id]):
                self._record_invalid(line, f"caption number {shuffled_idx + 1} out of range")
                continue

            results[id_mapping[img_id]] = shuffle_mappings[img_id][shuffled_idx]
        return results

    def _format_reltr(self, pred: str) -> str:
        parts = pred.split(' - ')
        if len(parts) == 3:
            subj, rel, obj = parts
            return f"{subj} {rel} {obj}"
        return pred

    def _format_hoi(self, action: str) -> str:
        return action.replace('_', ' ').replace(' - ', ' ')

    def batch_rerank(
        self,
        all_image_captions: List[Tuple[int, List[Tuple[str, float]]]],
        batch_size: int = 100,
        image_names: Dict[int, str] = None,
        detection_data: Dict = None,
        desc_data: Dict = None
    ) -> Dict[int, int]:
        if len(all_image_captions) == 0:
            return {}

        images_section = []
        context_section = []
        shuffle_mappings = {}
        id_mapping = {}

        for local_id, (img_id, captions) in enumerate(all_image_captions):
            indices = list(range(len(captions)))
            np.random.shuffle(indices)
            shuffle_mappings[local_id] = {i: orig_idx for i, orig_idx in enumerate(indices)}
            id_mapping[local_id] = img_id

            images_section.append(f"Image {local_id}:")
            for i, orig_idx in enumerate(indices, 1):
                cap, score = captions[orig_idx]
                images_section.append(f"  {i}. {cap}")
            images_section.append("")

            if image_names and img_id in image_names:
                img_name = image_names[img_id]
                ctx_parts = []

                # Detection evidence: YOLO objects, HOI actions, RelTR relations
                if detection_data and img_name in detection_data:
                    det = detection_data[img_name]
                    if det['yolo']:
                        ctx_parts.append(f"Objects: {det['yolo']}")
                    if det['hoi']:
                        hoi_str = "; ".join([self._format_hoi(p[0]) for p in det['hoi'][:config.K_PROMPT]])
                        ctx_parts.append(f"Actions: {hoi_str}")
                    if det['reltr']:
                        rel_str = "; ".join([self._format_reltr(p[0]) for p in det['reltr'][:config.K_PROMPT]])
                        ctx_parts.append(f"Relations: {rel_str}")

                if desc_data and img_name in desc_data:
                    top_descs = desc_data[img_name][:config.K_PROMPT]
                    descs_str = "; ".join([f"{d} ({s:.2f})" for d, s in top_descs])
                    ctx_parts.append(f"Scene descs: {descs_str}")

                if ctx_parts:
                    context_section.append(f"Image {local_id}: " + " | ".join(ctx_parts))

        full_prompt = self._build_rerank_prompt(images_section, "\n".join(context_section))
        self.last_full_prompt = full_prompt

        if not self._sample_prompt_printed:
            self._sample_prompt_printed = True
            print("\nSAMPLE PROMPT SENT TO LLM")
            print(full_prompt)
            print()

        response = self.quick_chat(
            full_prompt,
            system="You are a precise image caption evaluator. Use YOLO object detections (highly reliable), HOI action detections, RelTR relation detections, and BLIP-scored Qwen scene descriptions (contextual) to select the caption that best matches the actual image. Penalize hallucinated objects, wrong object labels, incorrect human activities, and <unk> tokens. Output ONLY in IMAGE_ID: NUMBER format, one line per image. No explanations."
        )
        return self._parse_rerank_response(response, shuffle_mappings, id_mapping)


class QwenLLMRanker(BaseLLMRanker):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        device: str = "auto",
        max_new_tokens: int = 1024,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ):
        super().__init__()
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        model_kwargs = {"device_map": device}

        if load_in_4bit or load_in_8bit:
            if importlib.util.find_spec("bitsandbytes") is None:
                raise ImportError("bitsandbytes is required for quantized loading but is not installed")

            from transformers import BitsAndBytesConfig
            if load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.device = self.model.device

    def quick_chat(self, message: str, system: str = "You are a helpful assistant.") -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message}
        ]

        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )

        input_ids = (tokenized.input_ids if hasattr(tokenized, "input_ids") else tokenized).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)


LLMRanker = QwenLLMRanker

# %%

DETECTION_DIR = "data/coco/detections"


# %%
def load_detection_data(split: str = "test"):
    BODY_PARTS = {
        'head', 'leg', 'ear', 'tail', 'mouth', 'nose', 'eye', 'face',
        'hand', 'arm', 'paw', 'hair', 'neck', 'foot', 'finger', 'wing',
        'feet', 'legs', 'arms', 'hands', 'ears', 'eyes', 'fingers', 'wings',
        'toe', 'toes', 'teeth', 'tongue', 'beak', 'claw', 'claws',
        'forehead', 'chin', 'cheek', 'elbow', 'knee', 'shoulder', 'wrist',
        'ankle', 'hip', 'chest', 'stomach', 'back', 'waist', 'thumb',
    }

    TRIVIAL_RELATIONS = {'has', 'of', 'part of', 'belong to', 'belonging to'}

    USEFUL_RELATIONS = {
        'behind', 'in', 'near', 'under', 'above', 'between', 'in front of',
        'next to', 'beside', 'along', 'across', 'around', 'inside',
        'holding', 'riding', 'sitting on', 'standing on', 'walking on',
        'carrying', 'watching', 'looking at', 'laying on', 'lying on',
        'using', 'eating', 'playing', 'covered in', 'mounted on',
        'parked on', 'hanging from', 'attached to', 'sitting in',
        'standing in', 'walking in', 'flying in', 'growing on',
    }

    USEFUL_WEARING = {
        'hat', 'helmet', 'backpack', 'glasses', 'sunglasses', 'mask',
        'jacket', 'coat', 'dress', 'uniform', 'suit', 'tie',
        'shirt', 'short', 'shorts', 'pant', 'pants', 'jean', 'jeans',
        'shoe', 'shoes', 'boot', 'boots', 'glove', 'gloves',
        'skirt', 'sweater', 'hoodie', 'vest', 'scarf', 'apron',
        'jersey', 'cap', 'beanie', 'bandana', 'goggles',
    }

    def filter_reltr(pred, score):
        if pred in ('no_relation', 'no relation'):
            return False
        if score < 0.40:
            return False
        parts = pred.split(' - ')
        if len(parts) != 3:
            return False
        subj, rel, obj = parts
        subj_lower, rel_lower, obj_lower = subj.lower(), rel.lower(), obj.lower()
        if subj_lower in BODY_PARTS and obj_lower in BODY_PARTS:
            return False
        if rel_lower in TRIVIAL_RELATIONS:
            if obj_lower in BODY_PARTS or subj_lower in BODY_PARTS:
                return False
            return score > 0.65
        if rel_lower == 'on':
            if obj_lower in BODY_PARTS or obj_lower in {'man', 'boy', 'woman', 'girl', 'person', 'people'}:
                return False
            if subj_lower in BODY_PARTS:
                return False
        if rel_lower in USEFUL_RELATIONS:
            return True
        if rel_lower == 'wearing':
            return obj_lower in USEFUL_WEARING or score > 0.60
        return score > 0.55

    def filter_hoi(pred, score):
        if pred == 'no_interaction':
            return False
        if 'no_interaction' in pred:
            return False
        if score < 0.40:
            return False
        return True

    hoi_path = os.path.join(DETECTION_DIR, 'hoi', f'coco_{split}_hoi_detection.csv')
    reltr_path = os.path.join(DETECTION_DIR, 'reltr', f'coco_{split}_reltr_detection.csv')
    yolo_path = os.path.join(DETECTION_DIR, 'yolo', f'coco_{split}_detection.csv')

    for path in (hoi_path, reltr_path, yolo_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Detection CSV not found: {path}")

    hoi_df = pd.read_csv(hoi_path)
    reltr_df = pd.read_csv(reltr_path)
    yolo_df = pd.read_csv(yolo_path)

    detection_data = {}

    for _, row in yolo_df.iterrows():
        img = row['image']
        detection_data[img] = {
            'yolo': row['objects'] if pd.notna(row['objects']) else '',
            'hoi': [],
            'reltr': []
        }

    for _, row in hoi_df.iterrows():
        img = row['image']
        if img not in detection_data:
            detection_data[img] = {'yolo': '', 'hoi': [], 'reltr': []}
        if filter_hoi(row['prediction'], row['score']):
            action = row['prediction'].replace('Human - ', '')
            detection_data[img]['hoi'].append((action, row['score']))

    for _, row in reltr_df.iterrows():
        img = row['image']
        if img not in detection_data:
            detection_data[img] = {'yolo': '', 'hoi': [], 'reltr': []}
        if filter_reltr(row['prediction'], row['score']):
            detection_data[img]['reltr'].append((row['prediction'], row['score']))

    for img in detection_data:
        detection_data[img]['hoi'] = sorted(
            detection_data[img]['hoi'], key=lambda x: x[1], reverse=True
        )
        detection_data[img]['reltr'] = sorted(
            detection_data[img]['reltr'], key=lambda x: x[1], reverse=True
        )

    print(f"Detection data loaded: {len(detection_data)} images")

    return detection_data

detection_data = load_detection_data(split="test")

# %%
def load_desc_data(csv_path: str, score_threshold: float = 0.25) -> Dict[str, List[Tuple[str, float]]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Desc CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    desc_data = {}

    for _, row in df.iterrows():
        descs_scores = []
        for i in range(1, 6):
            desc  = str(row.get(f'desc_{i}',  '')).strip()
            score = float(row.get(f'score_{i}', 0.0))
            if desc and desc.lower() != 'nan' and score >= score_threshold:
                descs_scores.append((desc, score))

        descs_scores.sort(key=lambda x: x[1], reverse=True)
        if descs_scores:
            desc_data[str(row['image'])] = descs_scores

    print(f"Desc data loaded: {len(desc_data)} images")
    return desc_data

DESC_CSV_PATH = "data/coco/descriptions/coco_descs_test_scored.csv"
desc_data = load_desc_data(DESC_CSV_PATH)

# %%
def load_candidates_from_csv(csv_path: str) -> Tuple[List, Dict, Dict]:
    df = pd.read_csv(csv_path)

    caption_buffer = []
    image_id_to_name = {}
    gts = {}

    for img_id, group in df.groupby('image_id'):
        group = group.sort_values('candidate_rank')

        img_name = group.iloc[0]['image_name']
        image_id_to_name[img_id] = img_name

        top_k_captions = [
            (row['caption'], row['beam_score'])
            for _, row in group.iterrows()
        ]

        ref_strings = []
        refs = []
        for i in range(1, 6):
            col = f'ground_truth_{i}'
            if col in group.columns:
                gt = group.iloc[0][col]
                if pd.notna(gt) and gt:
                    ref_strings.append(str(gt))
                    refs.append(str(gt).split())

        if refs and top_k_captions:
            caption_buffer.append((img_id, top_k_captions, refs, ref_strings))
            gts[str(img_id)] = ref_strings

    print(f"Loaded {len(caption_buffer)} images from CSV")

    return caption_buffer, image_id_to_name, gts

CSV_PATH = "outputs/coco/beam_candidates_epoch_<EPOCH>.csv"
caption_buffer, image_id_to_name, gts = load_candidates_from_csv(CSV_PATH)

# %%
llm_ranker = LLMRanker(model_id="Qwen/Qwen3-4B")

# %%
all_best_indices = {}

for batch_start in tqdm.tqdm(range(0, len(caption_buffer), config.LLM_BATCH_SIZE), desc="LLM Reranking"):
    batch_end = min(batch_start + config.LLM_BATCH_SIZE, len(caption_buffer))
    batch = caption_buffer[batch_start:batch_end]
    batch_for_llm = [(item[0], item[1]) for item in batch]

    batch_results = llm_ranker.batch_rerank(
        batch_for_llm,
        batch_size=config.LLM_BATCH_SIZE,
        image_names=image_id_to_name,
        detection_data=detection_data,
        desc_data=desc_data
    )
    all_best_indices.update(batch_results)
    torch.cuda.empty_cache()

res = {}
fallback_ids = set()

for img_id, top_k_captions, refs, ref_strings in caption_buffer:
    if img_id in all_best_indices:
        best_idx = all_best_indices[img_id]
    else:
        best_idx = 0
        fallback_ids.add(img_id)

    best_caption = top_k_captions[best_idx][0]
    res[str(img_id)] = [best_caption]

# %%
# Save the LLM's selection results to CSV
RERANK_CSV_PATH = 'outputs/coco/coco_llm_rerank_selections.csv'
os.makedirs(os.path.dirname(RERANK_CSV_PATH), exist_ok=True)

rerank_rows = []
for img_id, top_k_captions, refs, ref_strings in caption_buffer:
    best_idx = all_best_indices.get(img_id, 0)
    best_caption, best_score = top_k_captions[best_idx]

    row = {
        'image_id': img_id,
        'image_name': image_id_to_name.get(img_id, 'unknown'),
        'selected_rank': best_idx + 1,
        'selected_caption': best_caption,
        'beam_score': best_score,
        'selection_source': 'beam_fallback' if img_id in fallback_ids else 'llm',
    }
    for i, gt in enumerate(ref_strings, 1):
        row[f'ground_truth_{i}'] = gt

    rerank_rows.append(row)

rerank_df = pd.DataFrame(rerank_rows)
rerank_df.to_csv(RERANK_CSV_PATH, index=False)
print(f"LLM rerank selections saved to {RERANK_CSV_PATH}  ({len(rerank_df)} rows)")

# %%
bleu_scorer = Bleu(4)
(bleu_1, bleu_2, bleu_3, bleu_4), _ = bleu_scorer.compute_score(gts, res)

cider_scorer = Cider()
cider_score, _ = cider_scorer.compute_score(gts, res)

rouge_scorer = Rouge()
rouge_score, _ = rouge_scorer.compute_score(gts, res)

meteor_scorer = Meteor()
meteor_score, _ = meteor_scorer.compute_score(gts, res)

print("\nMS COCO EVALUATION RESULTS")
print(f"BLEU-1:  {bleu_1:.4f}")
print(f"BLEU-2:  {bleu_2:.4f}")
print(f"BLEU-3:  {bleu_3:.4f}")
print(f"BLEU-4:  {bleu_4:.4f}")
print(f"CIDEr:   {cider_score:.4f}")
print(f"METEOR:  {meteor_score:.4f}")
print(f"ROUGE-L: {rouge_score:.4f}")

invalid_line_count = len(llm_ranker.invalid_lines)

print(f"\nLLM errors (fell back to beam top-1): {len(fallback_ids)}/{len(caption_buffer)} images")
print(f"Invalid response lines: {invalid_line_count}")
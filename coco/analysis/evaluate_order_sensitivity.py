# %%
# !pip install -q transformers accelerate bitsandbytes
# !pip install -q nltk pandas
# !pip install -q git+https://github.com/salaniz/pycocoevalcap.git
# !pip install -q sacremoses

# %%
import torch
import pandas as pd
import numpy as np
import os
import re
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
CSV_PATH       = "outputs/coco/beam_candidates_epoch_<EPOCH>.csv"
DESC_CSV_PATH  = "data/coco/descriptions/coco_descs_test_scored.csv"
DETECTION_DIR  = "data/coco/detections"

LLM_MODEL_ID   = "Qwen/Qwen3-4B"
N_ORDERINGS    = 5
ORDER_SEED     = 42
ORDER_TAGS     = [f"P{k + 1}" for k in range(N_ORDERINGS)]


# %%
@dataclass
class Config:
    LLM_BATCH_SIZE: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

config = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(config.seed)

# %%
class BaseLLMRanker(ABC):
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
            try:
                if ":" in line:
                    parts = line.split(":", 1)
                else:
                    parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                img_id_str = parts[0].strip().replace("Image", "").replace("image", "").strip()
                img_id = int(img_id_str)
                number_match = re.search(r'\d+', parts[1].strip())
                if not number_match:
                    continue
                shuffled_idx = int(number_match.group()) - 1
                if img_id in shuffle_mappings and 0 <= shuffled_idx < len(shuffle_mappings[img_id]):
                    original_idx = shuffle_mappings[img_id][shuffled_idx]
                    global_img_id = id_mapping[img_id]
                    results[global_img_id] = original_idx
            except Exception:
                continue
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
        desc_data: Dict = None,
        orderings: Dict[int, List[int]] = None
    ) -> Dict[int, int]:
        if len(all_image_captions) == 0:
            return {}

        images_section = []
        context_section = []
        shuffle_mappings = {}
        id_mapping = {}

        for local_id, (img_id, captions) in enumerate(all_image_captions):
            if orderings is not None and img_id in orderings:
                indices = list(orderings[img_id])
            else:
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
                        hoi_str = "; ".join([self._format_hoi(p[0]) for p in det['hoi'][:3]])
                        ctx_parts.append(f"Actions: {hoi_str}")
                    if det['reltr']:
                        rel_str = "; ".join([self._format_reltr(p[0]) for p in det['reltr'][:3]])
                        ctx_parts.append(f"Relations: {rel_str}")

                if desc_data and img_name in desc_data:
                    top_descs = desc_data[img_name][:3]
                    descs_str = "; ".join([f"{d} ({s:.2f})" for d, s in top_descs])
                    ctx_parts.append(f"Scene descs: {descs_str}")

                if ctx_parts:
                    context_section.append(f"Image {local_id}: " + " | ".join(ctx_parts))

        full_prompt = self._build_rerank_prompt(images_section, "\n".join(context_section))

        try:
            response = self.quick_chat(
                full_prompt,
                system="You are a precise image caption evaluator. Use YOLO object detections (highly reliable), HOI action detections, RelTR relation detections, and BLIP-scored Qwen scene descriptions (contextual) to select the caption that best matches the actual image. Penalize hallucinated objects, wrong object labels, incorrect human activities, and <unk> tokens. Output ONLY in IMAGE_ID: NUMBER format, one line per image. No explanations."
            )
            return self._parse_rerank_response(response, shuffle_mappings, id_mapping)
        except Exception as e:
            print(f"LLM rerank error: {e}")
            return {}


class QwenLLMRanker(BaseLLMRanker):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        device: str = "auto",
        max_new_tokens: int = 1024,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        model_kwargs = {"device_map": device}

        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except ImportError:
                print("Warning: bitsandbytes not installed, falling back to float16")
                model_kwargs["torch_dtype"] = torch.float16
        elif load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                print("Warning: bitsandbytes not installed, falling back to float16")
                model_kwargs["torch_dtype"] = torch.float16
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

    hoi_df = pd.read_csv(hoi_path) if os.path.exists(hoi_path) else pd.DataFrame()
    reltr_df = pd.read_csv(reltr_path) if os.path.exists(reltr_path) else pd.DataFrame()
    yolo_df = pd.read_csv(yolo_path) if os.path.exists(yolo_path) else pd.DataFrame()

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
        )[:5]
        detection_data[img]['reltr'] = sorted(
            detection_data[img]['reltr'], key=lambda x: x[1], reverse=True
        )[:5]

    return detection_data

detection_data = load_detection_data(split="test")

# %%
def load_desc_data(csv_path: str, score_threshold: float = 0.25) -> Dict[str, List[Tuple[str, float]]]:
    if not os.path.exists(csv_path):
        print(f"Desc CSV not found: {csv_path}")
        return {}

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

    return desc_data

desc_data = load_desc_data(DESC_CSV_PATH)

# %%
def load_candidates_from_csv(csv_path: str) -> Tuple[List, Dict]:
    df = pd.read_csv(csv_path)

    caption_buffer = []
    image_id_to_name = {}

    for img_id, group in df.groupby('image_id'):
        group = group.sort_values('candidate_rank')

        img_name = group.iloc[0]['image_name']
        image_id_to_name[img_id] = img_name

        top_k_captions = [
            (row['caption'], row['beam_score'])
            for _, row in group.iterrows()
        ]

        refs = []
        for i in range(1, 6):
            col = f'ground_truth_{i}'
            if col in group.columns:
                gt = group.iloc[0][col]
                if pd.notna(gt) and gt:
                    refs.append(str(gt).split())

        if refs and top_k_captions:
            caption_buffer.append((img_id, top_k_captions))

    return caption_buffer, image_id_to_name

caption_buffer, image_id_to_name = load_candidates_from_csv(CSV_PATH)

# %%
llm_ranker = LLMRanker(model_id=LLM_MODEL_ID)

# %%
def build_cyclic_orderings(buffer, n_orderings: int = N_ORDERINGS, seed: int = ORDER_SEED):
    rng = random.Random(seed)
    variants = [{} for _ in range(n_orderings)]

    for img_id, top_k_captions in buffer:
        n = len(top_k_captions)
        base = list(range(n))
        rng.shuffle(base)
        for k in range(n_orderings):
            variants[k][img_id] = [base[(i + k) % n] for i in range(n)]

    return variants


ordering_variants = build_cyclic_orderings(caption_buffer)

if caption_buffer:
    demo_id = caption_buffer[0][0]
    print(f"Prompt orderings for image_id={demo_id} (values = original beam rank, 1-based):")
    for tag, variant in zip(ORDER_TAGS, ordering_variants):
        print(f"  {tag}: {[i + 1 for i in variant[demo_id]]}")

# %%
def run_rerank(orderings: Dict[int, List[int]], tag: str):
    selections = {}

    for batch_start in tqdm.tqdm(
        range(0, len(caption_buffer), config.LLM_BATCH_SIZE),
        desc=f"LLM Reranking {tag}"
    ):
        batch_end = min(batch_start + config.LLM_BATCH_SIZE, len(caption_buffer))
        batch = caption_buffer[batch_start:batch_end]

        batch_results = llm_ranker.batch_rerank(
            batch,
            batch_size=config.LLM_BATCH_SIZE,
            image_names=image_id_to_name,
            detection_data=detection_data,
            desc_data=desc_data,
            orderings=orderings
        )
        selections.update(batch_results)
        torch.cuda.empty_cache()

    return selections

def build_ranks(selections: Dict[int, int]) -> Dict[int, int]:
    ranks = {}

    for img_id, top_k_captions in caption_buffer:
        best_idx = selections.get(img_id, 0)
        if best_idx >= len(top_k_captions):
            best_idx = 0
        ranks[img_id] = best_idx + 1

    return ranks


order_ranks = {}

for tag, orderings in zip(ORDER_TAGS, ordering_variants):
    order_ranks[tag] = build_ranks(run_rerank(orderings, tag))

# %%
n_total = len(caption_buffer)
n_consistent = sum(
    1 for img_id, _ in caption_buffer
    if len({order_ranks[tag][img_id] for tag in ORDER_TAGS}) == 1
)

print(f"\nImages with the SAME selection across all {N_ORDERINGS} orderings: "
      f"{n_consistent}/{n_total} ({n_consistent / n_total * 100:.2f}%)")
print(f"Images whose selection CHANGES with candidate order: "
      f"{n_total - n_consistent}/{n_total} ({(n_total - n_consistent) / n_total * 100:.2f}%)")

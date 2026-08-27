# !pip install -q transformers accelerate bitsandbytes
# !pip install -q nltk pandas
# !pip install -q git+https://github.com/salaniz/pycocoevalcap.git

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import h5py
import tqdm
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pandas as pd

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer as _PTBTok


def _ptb_tokenize_batch(captions: list[str]) -> list[str]:
    ptb = _PTBTok()
    inp = {str(i): [{"caption": c}] for i, c in enumerate(captions)}
    out = ptb.tokenize(inp)
    return [out[str(i)][0] for i in range(len(captions))]


@dataclass
class Config:
    max_words: int = 45
    maxlen: int = max_words + 1
    batch_size: int = 64
    epochs: int = 7
    stop_after_epoch: Optional[int] = 7
    learning_rate: float = 1e-4
    d_model: int = 768
    n_heads_encoder: int = 12
    n_heads_decoder: int = 12
    n_layers_encoder: int = 2
    n_layers_decoder: int = 3

    karpathy_json: str = "data/flickr30k/dataset_flickr30k.json"
    image_dir: str = "data/flickr30k/flickr30k_images"

    train_h5_parts: tuple = (
        "data/flickr30k/features/train_part1.h5",
        "data/flickr30k/features/train_part2.h5",
    )
    val_h5: str = "data/flickr30k/features/val.h5"
    test_h5: str = "data/flickr30k/features/test.h5"

    train_desc_csv: str = "data/flickr30k/descriptions/flickr30k_descs_train_scored.csv"
    val_desc_csv: str = "data/flickr30k/descriptions/flickr30k_descs_val_scored.csv"
    test_desc_csv: str = "data/flickr30k/descriptions/flickr30k_descs_test_scored.csv"

    vocab_size: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    BEAM_WIDTH: int = 5

    min_freq: int = 5

    max_train: Optional[int] = None
    max_val: Optional[int] = None
    max_test: Optional[int] = None

    feature_dim: int = 1024
    grid_h: int = 16
    grid_w: int = 16

    max_desc_len: int = 20
    num_descs_per_image: int = 5
    lambda_desc: float = 0.3

    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    warmup_steps: int = 2000

    seed: int = 42
    compile_decoder: bool = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def _worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class WarmupCosineScheduler:

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.opt = optimizer
        self.warmup = warmup_steps
        self.total = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        s = self._step
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if s < self.warmup:
                lr = base * s / max(self.warmup, 1)
            else:
                prog = (s - self.warmup) / max(self.total - self.warmup, 1)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * prog))
            g["lr"] = lr

    @property
    def current_lr(self) -> float:
        return self.opt.param_groups[0]["lr"]


class DataManager:

    def __init__(self, config: Config):
        self.config = config
        self.caption_mapping = {}
        self.reference_mapping = {}
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {}
        self.karpathy_data = None

    def _get_image_path(self, img_info: dict) -> str:
        filename = img_info["filename"]
        return os.path.join(self.config.image_dir, filename)

    def load_captions(self) -> Dict:

        with open(self.config.karpathy_json, "r") as f:
            self.karpathy_data = json.load(f)

        all_sents = [s for img in self.karpathy_data["images"] for s in img["sentences"]]
        tokenized = _ptb_tokenize_batch([s["raw"] for s in all_sents])
        for s, tok in zip(all_sents, tokenized):
            s["tok"] = tok

        for img_info in self.karpathy_data["images"]:
            img_path = self._get_image_path(img_info)

            captions_for_img = []
            references_for_img = []
            for sent in img_info["sentences"]:
                tokens = sent["tok"].split()

                if not tokens:
                    continue

                references_for_img.append(sent["tok"])

                if len(tokens) < 5 or len(tokens) > self.config.max_words:
                    continue

                caption = "<start> " + sent["tok"] + " <end>"
                captions_for_img.append(caption)

            if references_for_img:
                self.reference_mapping[img_path] = references_for_img

            if captions_for_img:
                self.caption_mapping[img_path] = captions_for_img

        return self.caption_mapping

    def build_vocab(self, train_data: Dict) -> Dict:
        counter = Counter()
        for captions in train_data.values():
            for caption in captions:
                tokens = caption.split()
                counter.update(t for t in tokens if t not in self.word2idx)

        len_word2idx = len(self.word2idx)
        for token in sorted(counter.keys()):
            if counter[token] >= self.config.min_freq:
                self.word2idx[token] = len_word2idx
                len_word2idx += 1

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        return self.word2idx

    def split_by_karpathy(self) -> Tuple[Dict, Dict, Dict]:

        train_data = {}
        val_data = {}
        test_data = {}

        for img_info in self.karpathy_data["images"]:
            img_path = self._get_image_path(img_info)
            split = img_info["split"]

            if split == "test":
                if img_path in self.reference_mapping:
                    test_data[img_path] = [
                        "<start> " + ref + " <end>"
                        for ref in self.reference_mapping[img_path]
                    ]
                continue

            if img_path not in self.caption_mapping:
                continue

            if split in ("train", "restval"):
                train_data[img_path] = self.caption_mapping[img_path]
            elif split == "val":
                val_data[img_path] = self.caption_mapping[img_path]

        def _cap(d: Dict, n: Optional[int]) -> Dict:
            if n is None or n >= len(d):
                return d
            keys = list(d.keys())[:n]
            return {k: d[k] for k in keys}

        return (
            _cap(train_data, self.config.max_train),
            _cap(val_data, self.config.max_val),
            _cap(test_data, self.config.max_test),
        )

    def references_by_filename(self, split_data: Dict) -> Dict[str, List[str]]:
        return {
            os.path.basename(img_path): self.reference_mapping[img_path]
            for img_path in split_data
            if img_path in self.reference_mapping
        }


class DescManager:

    def __init__(self, config: Config):
        self.config = config
        self.desc_data: Dict[str, List[Tuple[str, float]]] = {}
        self.encoded_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _load_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            descs = []
            for i in range(1, self.config.num_descs_per_image + 1):
                desc = str(row.get(f"desc_{i}", "")).strip()
                score = float(row.get(f"score_{i}", 0.0))
                if desc and desc.lower() != "nan":
                    descs.append((desc, score))
                else:
                    descs.append(("", 0.0))
            self.desc_data[str(row["image"])] = descs

    def load_all(self):
        """Load all three CSV splits."""
        self._load_csv(self.config.train_desc_csv)
        self._load_csv(self.config.val_desc_csv)
        self._load_csv(self.config.test_desc_csv)

    def build_encoding_cache(self, word2idx: Dict):
        pad_row = [0] * self.config.max_desc_len

        all_filenames = list(self.desc_data.keys())
        flat_descs = [
            desc.lower() if desc else ""
            for filename in all_filenames
            for desc, _ in self.desc_data[filename][: self.config.num_descs_per_image]
        ]

        non_empty_indices = [i for i, t in enumerate(flat_descs) if t]
        non_empty_descs = [flat_descs[i] for i in non_empty_indices]
        tokenized_non_empty = _ptb_tokenize_batch(non_empty_descs) if non_empty_descs else []

        tokenized_flat = [""] * len(flat_descs)
        for i, tok in zip(non_empty_indices, tokenized_non_empty):
            tokenized_flat[i] = tok

        tok_idx = 0
        for filename in all_filenames:
            descs = self.desc_data[filename]
            all_token_ids = []
            all_scores = []

            for _, score in descs[: self.config.num_descs_per_image]:
                tok_str = tokenized_flat[tok_idx]
                tok_idx += 1
                if tok_str:
                    tokens = tok_str.split()[: self.config.max_desc_len]
                    ids = [word2idx.get(t, word2idx.get("<unk>", 3)) for t in tokens]
                    ids += [0] * (self.config.max_desc_len - len(ids))
                else:
                    ids = pad_row.copy()
                all_token_ids.append(ids)
                all_scores.append(score)

            while len(all_token_ids) < self.config.num_descs_per_image:
                all_token_ids.append(pad_row.copy())
                all_scores.append(0.0)

            self.encoded_cache[filename] = (
                torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_scores, dtype=torch.float),
            )

        print(f"DescManager: cached encodings for {len(self.encoded_cache)} images.")

    def get_encoded(self, filepath: str) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = os.path.basename(filepath)
        if filename in self.encoded_cache:
            return self.encoded_cache[filename]
        return (
            torch.zeros(
                self.config.num_descs_per_image,
                self.config.max_desc_len,
                dtype=torch.long,
            ),
            torch.zeros(self.config.num_descs_per_image, dtype=torch.float),
        )


class CaptionDataset(Dataset):

    def __init__(
        self,
        caption_data: Dict,
        config: Config,
        word2idx: Dict,
        h5_paths: List[str],
        desc_manager: Optional[DescManager] = None,
    ):
        self.config = config
        self.word2idx = word2idx
        self.desc_manager = desc_manager
        self.h5_paths = h5_paths
        self._h5_files: Optional[List] = None

        self._key_to_h5_idx: Dict[str, int] = {}
        for i, p in enumerate(h5_paths):
            with h5py.File(p, "r") as f:
                for key in f.keys():
                    self._key_to_h5_idx[key] = i

        self.samples = [
            (img_path, cap)
            for img_path, caps in caption_data.items()
            for cap in caps
            if os.path.basename(img_path) in self._key_to_h5_idx
        ]

        missing = sum(
            1 for img_path in caption_data
            if os.path.basename(img_path) not in self._key_to_h5_idx
        )
        if missing:
            print(f"[WARN] {missing} images not found in H5. Skipping.")

    def _ensure_open(self):
        if self._h5_files is None:
            self._h5_files = [h5py.File(p, "r") for p in self.h5_paths]

    def load_and_prepare_caption(
        self, caption: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = caption.split()

        input_tokens = tokens[:-1][: self.config.maxlen]
        target_tokens = tokens[1:][: self.config.maxlen]
        sample_size = len(input_tokens)
        padding_size = self.config.maxlen - sample_size

        if padding_size > 0:
            padding_vec = ["<pad>"] * padding_size
            input_tokens += padding_vec
            target_tokens += padding_vec

        input_t = torch.tensor(
            [
                self.word2idx.get(token, self.word2idx.get("<unk>", 0))
                for token in input_tokens
            ],
            dtype=torch.long,
        )
        target_t = torch.tensor(
            [
                self.word2idx.get(token, self.word2idx["<unk>"])
                for token in target_tokens
            ],
            dtype=torch.long,
        )

        tgt_padding_mask = torch.ones(self.config.maxlen, dtype=torch.bool)
        tgt_padding_mask[:sample_size] = False

        return (
            input_t[: self.config.maxlen],
            target_t[: self.config.maxlen],
            tgt_padding_mask,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        self._ensure_open()
        img_path, caption = self.samples[index]
        key = os.path.basename(img_path)
        h5_idx = self._key_to_h5_idx[key]
        features = torch.from_numpy(self._h5_files[h5_idx][key][:])

        cap_tuple = self.load_and_prepare_caption(caption)

        if self.desc_manager is not None:
            desc_tokens, desc_scores = self.desc_manager.get_encoded(img_path)
        else:
            desc_tokens = torch.zeros(
                self.config.num_descs_per_image,
                self.config.max_desc_len,
                dtype=torch.long,
            )
            desc_scores = torch.zeros(self.config.num_descs_per_image, dtype=torch.float)

        return features, cap_tuple, desc_tokens, desc_scores


class RotaryPositionalEncoding2D(nn.Module):

    def __init__(self, dim: int, height: int = 8, width: int = 8):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
        h_pos = torch.arange(height).float()
        w_pos = torch.arange(width).float()
        h_freqs = torch.einsum("i, j -> i j", h_pos, inv_freq)
        w_freqs = torch.einsum("i, j -> i j", w_pos, inv_freq)
        h_emb = h_freqs.unsqueeze(1).repeat(1, width, 1).reshape(-1, h_freqs.shape[-1])
        w_emb = w_freqs.unsqueeze(0).repeat(height, 1, 1).reshape(-1, w_freqs.shape[-1])
        cos_emb = torch.cat([h_emb.cos(), w_emb.cos()], dim=-1)
        sin_emb = torch.cat([h_emb.sin(), w_emb.sin()], dim=-1)
        if cos_emb.shape[-1] < dim:
            padding = dim - cos_emb.shape[-1]
            cos_emb = F.pad(cos_emb, (0, padding), value=1.0)
            sin_emb = F.pad(sin_emb, (0, padding), value=0.0)
        self.register_buffer("cos_emb", cos_emb)
        self.register_buffer("sin_emb", sin_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.cos_emb.unsqueeze(0)) + (
            self._rotate_half(x) * self.sin_emb.unsqueeze(0)
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)


class Learned2DPE(nn.Module):

    def __init__(self, dim: int, height: int = 8, width: int = 8):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, height * width, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class Sinusoidal2DPE(nn.Module):

    def __init__(self, dim: int, height: int = 8, width: int = 8):
        super().__init__()
        pe = torch.zeros(height * width, dim)
        position = torch.arange(0, height * width).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class RotaryPositionalEncoding(nn.Module):

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i , j -> i j", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_emb", emb.cos())
        self.register_buffer("sin_emb", emb.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return (x * self.cos_emb[:seq_len].unsqueeze(0)) + (
            self._rotate_half(x) * self.sin_emb[:seq_len].unsqueeze(0)
        )

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)


class DescTokenEncoder(nn.Module):

    def __init__(self, shared_embedding: nn.Embedding, d_model: int):
        super().__init__()
        self.embedding = shared_embedding
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(
        self, desc_tokens: torch.Tensor, desc_scores: torch.Tensor
    ) -> torch.Tensor:
        B, num_descs, max_desc_len = desc_tokens.shape

        flat_tokens = desc_tokens.reshape(B * num_descs, max_desc_len)
        embedded = self.embedding(flat_tokens)

        mask = (flat_tokens != 0).float().unsqueeze(-1)
        desc_emb = (embedded * mask).sum(1) / mask.sum(1).clamp(min=1)
        desc_emb = desc_emb.reshape(B, num_descs, -1)
        desc_emb = desc_emb * desc_scores.unsqueeze(-1)

        return self.proj(desc_emb)


class DescAuxHead(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        pooled = encoder_output.mean(dim=1)
        return F.normalize(self.proj(pooled), dim=-1)


class DescVisualCrossAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, desc_emb: torch.Tensor, visual_features: torch.Tensor
    ) -> torch.Tensor:
        enhanced, _ = self.cross_attn(
            query=desc_emb, key=visual_features, value=visual_features
        )
        return self.norm(desc_emb + enhanced)


class Encoder(nn.Module):

    def __init__(self, config: Config, feature_dim: int = 1024):
        super().__init__()
        self.rope_dim = 256
        self.learned_dim = 256
        self.sin_dim = config.d_model - self.rope_dim - self.learned_dim

        self.rope_2d = RotaryPositionalEncoding2D(dim=self.rope_dim, height=config.grid_h, width=config.grid_w)
        self.learned_2d = Learned2DPE(dim=self.learned_dim, height=config.grid_h, width=config.grid_w)
        self.sinusoidal_2d = Sinusoidal2DPE(dim=self.sin_dim, height=config.grid_h, width=config.grid_w)

        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model, nhead=config.n_heads_encoder, batch_first=True
            ),
            num_layers=config.n_layers_encoder,
        )

        self.attention_pool = nn.MultiheadAttention(
            embed_dim=config.d_model, num_heads=config.n_heads_encoder, batch_first=True
        )

        self.desc_visual_cross_attn = DescVisualCrossAttention(
            d_model=config.d_model, n_heads=config.n_heads_encoder
        )

    def forward(
        self, combined_features: torch.Tensor, desc_memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        proj_features = self.feature_proj(combined_features)
        x1 = self.rope_2d(proj_features[..., : self.rope_dim])
        x2 = self.learned_2d(
            proj_features[..., self.rope_dim : self.rope_dim + self.learned_dim]
        )
        x3 = self.sinusoidal_2d(proj_features[..., self.rope_dim + self.learned_dim :])
        features = torch.cat([x1, x2, x3], dim=-1)

        if desc_memory is not None:
            desc_memory = self.desc_visual_cross_attn(desc_memory, features)
            features = torch.cat([features, desc_memory], dim=1)

        attn_features, _ = self.attention_pool(features, features, features)
        return self.encoder(attn_features)


class Decoder(nn.Module):

    def __init__(self, config: Config, shared_embedding: Optional[nn.Embedding] = None):
        super().__init__()
        self.embedding = (
            shared_embedding
            if shared_embedding is not None
            else nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        )
        self.rope = RotaryPositionalEncoding(config.d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.d_model, nhead=config.n_heads_decoder, batch_first=True
            ),
            num_layers=config.n_layers_decoder,
        )
        self.out = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        encoder_outputs: torch.Tensor,
        tgt_attention_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.rope(self.embedding(x))
        decoder_out = self.decoder(
            tgt=x,
            memory=encoder_outputs,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_attention_mask,
        )
        return self.out(decoder_out)


class CaptionModel(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config, feature_dim=config.feature_dim)

        self.shared_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=0
        )

        self.desc_token_encoder = DescTokenEncoder(self.shared_embedding, config.d_model)
        self.desc_aux_head = DescAuxHead(config.d_model)
        self.decoder = Decoder(config, shared_embedding=self.shared_embedding)

    def forward(
        self,
        image_features: torch.Tensor,
        input_tokens: torch.Tensor,
        desc_tokens: Optional[torch.Tensor] = None,
        desc_scores: Optional[torch.Tensor] = None,
        tgt_attention_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        desc_memory = None
        if desc_tokens is not None and desc_scores is not None:
            desc_memory = self.desc_token_encoder(desc_tokens, desc_scores)

        encoder_outputs = self.encoder(image_features, desc_memory)
        caption_logits = self.decoder(
            input_tokens, encoder_outputs, tgt_attention_mask, tgt_key_padding_mask
        )
        pred_desc_emb = self.desc_aux_head(encoder_outputs)

        return caption_logits, pred_desc_emb

    def save(self, path: str):
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "shared_embedding": self.shared_embedding.state_dict(),
                "desc_token_encoder": self.desc_token_encoder.state_dict(),
                "desc_aux_head": self.desc_aux_head.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.shared_embedding.load_state_dict(ckpt["shared_embedding"])
        self.desc_token_encoder.load_state_dict(ckpt["desc_token_encoder"])
        self.desc_aux_head.load_state_dict(ckpt["desc_aux_head"])


class Trainer:

    def __init__(
        self,
        model: CaptionModel,
        config: Config,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        trainable_params = list(model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        total_steps = len(train_dataloader) * config.epochs
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps=config.warmup_steps, total_steps=total_steps
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.mask = self._create_causal_mask(config.maxlen, config.device)

        self.train_loss = []
        self.valid_loss = []
        self.training_accuracy = []
        self.validation_accuracy = []

    @staticmethod
    def _create_causal_mask(maxlen: int, device: str) -> torch.Tensor:
        mask = torch.triu(torch.ones((maxlen, maxlen)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, 0.0)
            .to(device)
        )
        mask.requires_grad = False
        return mask

    @staticmethod
    def _calculate_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        return ((y_pred == y_true).sum() / y_pred.numel()).item()

    def _compute_target_desc_emb(
        self, desc_tokens: torch.Tensor, desc_scores: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            desc_word_embs = self.model.shared_embedding(desc_tokens)
            tmask = (desc_tokens != 0).float().unsqueeze(-1)
            desc_phrase_embs = (desc_word_embs * tmask).sum(2) / tmask.sum(2).clamp(min=1)
            weights = F.softmax(desc_scores, dim=-1).unsqueeze(-1)
            target = (desc_phrase_embs * weights).sum(1)
            return F.normalize(target, dim=-1)

    def train_epoch(self, epoch: int):
        self.model.encoder.train()
        self.model.decoder.train()
        self.model.desc_token_encoder.train()
        self.model.desc_aux_head.train()

        epoch_loss = []
        epoch_acc = []

        for batch_idx, (features, ys, desc_tokens, desc_scores) in enumerate(
            tqdm.tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
        ):
            features = features.to(self.config.device, non_blocking=True)
            desc_tokens = desc_tokens.to(self.config.device, non_blocking=True)
            desc_scores = desc_scores.to(self.config.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                desc_memory = self.model.desc_token_encoder(desc_tokens, desc_scores)
                encoder_outputs = self.model.encoder(features, desc_memory)

                target_desc_emb = self._compute_target_desc_emb(desc_tokens, desc_scores)
                pred_desc_emb = self.model.desc_aux_head(encoder_outputs)
                desc_loss = (1.0 - (pred_desc_emb * target_desc_emb).sum(dim=-1)).mean()

                input_tokens = ys[0].to(self.config.device)
                tgt_tokens = ys[1].to(self.config.device)
                pad = ys[2].to(self.config.device)

                preds = self.model.decoder(
                    input_tokens,
                    encoder_outputs,
                    tgt_key_padding_mask=pad,
                    tgt_attention_mask=self.mask,
                )

                pad_mask = ~pad
                caption_loss = self.loss_fn(preds[pad_mask], tgt_tokens[pad_mask])
                total_acc = self._calculate_accuracy(
                    preds[pad_mask].argmax(dim=-1), tgt_tokens[pad_mask]
                )
                total_loss = caption_loss + self.config.lambda_desc * desc_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            epoch_loss.append(total_loss.item())
            epoch_acc.append(total_acc)

        self.train_loss.extend(epoch_loss)
        self.training_accuracy.extend(epoch_acc)

        return np.mean(epoch_loss), np.mean(epoch_acc)

    def validate(self) -> Tuple[float, float]:
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.model.desc_token_encoder.eval()
        self.model.desc_aux_head.eval()

        epoch_loss = []
        epoch_acc = []

        with torch.no_grad():
            for features, ys, desc_tokens, desc_scores in self.valid_dataloader:
                features = features.to(self.config.device, non_blocking=True)
                desc_tokens = desc_tokens.to(self.config.device, non_blocking=True)
                desc_scores = desc_scores.to(self.config.device, non_blocking=True)

                desc_memory = self.model.desc_token_encoder(desc_tokens, desc_scores)
                encoder_outputs = self.model.encoder(features, desc_memory)

                target_desc_emb = self._compute_target_desc_emb(desc_tokens, desc_scores)
                pred_desc_emb = self.model.desc_aux_head(encoder_outputs)
                desc_loss = (1.0 - (pred_desc_emb * target_desc_emb).sum(dim=-1)).mean()

                input_tokens = ys[0].to(self.config.device)
                tgt_tokens = ys[1].to(self.config.device)
                pad = ys[2].to(self.config.device)

                preds = self.model.decoder(
                    input_tokens,
                    encoder_outputs,
                    tgt_key_padding_mask=pad,
                    tgt_attention_mask=self.mask,
                )

                pad_mask = ~pad
                caption_loss = self.loss_fn(preds[pad_mask], tgt_tokens[pad_mask])
                total_acc = self._calculate_accuracy(
                    preds[pad_mask].argmax(dim=-1), tgt_tokens[pad_mask]
                )
                total_loss = caption_loss.item() + self.config.lambda_desc * desc_loss.item()
                epoch_loss.append(total_loss)
                epoch_acc.append(total_acc)

        self.valid_loss.extend(epoch_loss)
        self.validation_accuracy.extend(epoch_acc)

        return np.mean(epoch_loss), np.mean(epoch_acc)

    def save_checkpoint(self, epoch: int, path: str):
        loader_gen = getattr(self.train_dataloader, "generator", None)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "scheduler_step": self.scheduler._step,
                "scheduler_total": self.scheduler.total,
                "scheduler_warmup": self.scheduler.warmup,
                "scheduler_base_lrs": self.scheduler.base_lrs,
                "scheduler_min_lr": self.scheduler.min_lr,
                "train_loss": self.train_loss,
                "valid_loss": self.valid_loss,
                "training_accuracy": self.training_accuracy,
                "validation_accuracy": self.validation_accuracy,
                "rng_python": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
                "rng_dataloader": (
                    loader_gen.get_state() if loader_gen is not None else None
                ),
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.scheduler._step = ckpt["scheduler_step"]
        self.scheduler.total = ckpt["scheduler_total"]
        self.scheduler.warmup = ckpt["scheduler_warmup"]
        self.scheduler.base_lrs = ckpt["scheduler_base_lrs"]
        self.scheduler.min_lr = ckpt["scheduler_min_lr"]
        self.train_loss = ckpt["train_loss"]
        self.valid_loss = ckpt["valid_loss"]
        self.training_accuracy = ckpt["training_accuracy"]
        self.validation_accuracy = ckpt["validation_accuracy"]

        if "rng_python" in ckpt:
            random.setstate(ckpt["rng_python"])
            np.random.set_state(ckpt["rng_numpy"])
            torch.set_rng_state(ckpt["rng_torch"].cpu())
            if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_cuda"]])
            loader_gen = getattr(self.train_dataloader, "generator", None)
            if ckpt["rng_dataloader"] is not None and loader_gen is not None:
                loader_gen.set_state(ckpt["rng_dataloader"].cpu())
        else:
            print("[WARN] Checkpoint has no RNG state; resumed run will not "
                  "reproduce an uninterrupted one.")

        start_epoch = ckpt["epoch"] + 1
        print(f"Checkpoint loaded from {path}, resuming from epoch {start_epoch + 1}")
        return start_epoch

    def train(self, num_epochs: Optional[int] = None, csv_callback=None, checkpoint_path: Optional[str] = None, model_path: Optional[str] = None, resume_from: Optional[str] = None, stop_after_epoch: Optional[int] = None):
        if num_epochs is None:
            num_epochs = self.config.epochs

        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from)

        for epoch in range(start_epoch, num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if checkpoint_path is not None:
                self.save_checkpoint(epoch, checkpoint_path)

            if model_path is not None:
                epoch_model_path = model_path.format(epoch=epoch + 1)
                self.model.save(epoch_model_path)
                print(f"Model weights saved to {epoch_model_path}")

            if csv_callback is not None:
                csv_callback(epoch)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if stop_after_epoch is not None and (epoch + 1) >= stop_after_epoch:
                print(
                    f"Stopping after epoch {epoch+1}/{num_epochs} as requested. "
                    f"Resume with resume_from='{checkpoint_path}'."
                )
                break

    def plot_metrics(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, label="train_loss")
        plt.plot(self.valid_loss, label="valid_loss")
        plt.title("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.training_accuracy, label="train_acc")
        plt.plot(self.validation_accuracy, label="valid_acc")
        plt.title("Accuracy")
        plt.legend()

        plt.show()


class InferenceEngine:

    def __init__(
        self, model: CaptionModel, config: Config, word2idx: Dict, idx2word: Dict
    ):
        self.model = model
        self.config = config
        self.word2idx = word2idx
        self.idx2word = idx2word

    def beam_search(
        self,
        image_features: Optional[torch.Tensor] = None,
        desc_tokens: Optional[torch.Tensor] = None,
        desc_scores: Optional[torch.Tensor] = None,
        beam_width: int = 3,
        max_len: Optional[int] = None,
        length_penalty: float = 0.6,
        encoder_outputs: Optional[torch.Tensor] = None,
    ) -> List[Tuple[str, float]]:
        if max_len is None:
            max_len = self.config.maxlen

        self.model.eval()

        with torch.inference_mode():
            if encoder_outputs is None:
                desc_memory = None
                if desc_tokens is not None and desc_scores is not None:
                    desc_memory = self.model.desc_token_encoder(
                        desc_tokens.to(self.config.device), desc_scores.to(self.config.device)
                    )
                encoder_outputs = self.model.encoder(
                    image_features.to(self.config.device), desc_memory
                )

            end_idx = self.word2idx["<end>"]
            pad_idx = self.word2idx["<pad>"]

            beams = [([self.word2idx["<start>"]], 0.0)]
            completed_beams = []

            for step in range(max_len):
                active = [(seq, sc) for seq, sc in beams if seq[-1] != end_idx]
                done   = [(seq, sc) for seq, sc in beams if seq[-1] == end_idx]
                completed_beams.extend(done)

                if not active:
                    break

                seq_len = len(active[0][0])
                token_batch = torch.full(
                    (len(active), seq_len), pad_idx, dtype=torch.long, device=self.config.device
                )
                for i, (seq, _) in enumerate(active):
                    token_batch[i] = torch.tensor(seq, device=self.config.device)

                enc = encoder_outputs.expand(len(active), -1, -1)
                x = self.model.decoder.rope(self.model.decoder.embedding(token_batch))
                causal_mask = torch.triu(
                    torch.full(
                        (seq_len, seq_len), float("-inf"), device=self.config.device
                    ),
                    diagonal=1,
                )
                logits = self.model.decoder.out(
                    self.model.decoder.decoder(x, enc, tgt_mask=causal_mask)
                )
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

                top_log_probs, top_indices = log_probs.topk(beam_width, dim=-1)

                all_candidates = []
                for i, (seq, score) in enumerate(active):
                    for j in range(beam_width):
                        new_seq   = seq + [top_indices[i, j].item()]
                        new_score = score + top_log_probs[i, j].item()
                        all_candidates.append((new_seq, new_score))

                all_candidates.sort(
                    key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True
                )
                beams = all_candidates[:beam_width]

                if all(seq[-1] == end_idx for seq, _ in beams):
                    completed_beams.extend(beams)
                    break

            completed_beams.extend(beams)

            completed_beams.sort(
                key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True
            )
            top_k_beams = completed_beams[:beam_width]

            top_k_captions = []
            for seq, raw_score in top_k_beams:
                caption = " ".join(
                    [
                        self.idx2word.get(token, "<unk>")
                        for token in seq[1:]
                        if token != self.word2idx["<end>"]
                    ]
                )
                normalized_score = raw_score / (len(seq) ** length_penalty)
                top_k_captions.append((caption, normalized_score))

            return top_k_captions


config = Config()
set_seed(config.seed)

dataloader_generator = torch.Generator()
dataloader_generator.manual_seed(config.seed)

valid_generator = torch.Generator()
valid_generator.manual_seed(config.seed + 1)

test_generator = torch.Generator()
test_generator.manual_seed(config.seed + 2)

data_manager = DataManager(config)
captions_mapping = data_manager.load_captions()

train_data, valid_data, test_data = data_manager.split_by_karpathy()
test_references = data_manager.references_by_filename(test_data)

word2idx = data_manager.build_vocab(train_data)
config.vocab_size = len(word2idx)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(valid_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Vocabulary size: {config.vocab_size}")

desc_manager = DescManager(config)
desc_manager.load_all()
desc_manager.build_encoding_cache(word2idx)

train_dataset = CaptionDataset(train_data, config, word2idx, h5_paths=list(config.train_h5_parts), desc_manager=desc_manager)
valid_dataset = CaptionDataset(valid_data, config, word2idx, h5_paths=[config.val_h5], desc_manager=desc_manager)
test_dataset = CaptionDataset(test_data, config, word2idx, h5_paths=[config.test_h5], desc_manager=desc_manager)

import gc
del train_data, valid_data, test_data
del data_manager.karpathy_data, data_manager.caption_mapping, data_manager.reference_mapping
del desc_manager.desc_data
gc.collect()

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False, worker_init_fn=_worker_init_fn, generator=dataloader_generator)
valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False, worker_init_fn=_worker_init_fn, generator=valid_generator)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=False, worker_init_fn=_worker_init_fn, generator=test_generator)

idx2word = {v: k for k, v in word2idx.items()}

VOCAB_PATH = "outputs/flickr30k/word2idx.json"
os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
with open(VOCAB_PATH, "w") as f:
    json.dump(word2idx, f)
print(f"Vocabulary ({len(word2idx)} tokens) saved to {VOCAB_PATH}")

print("Data loading complete!")

set_seed(config.seed)

model = CaptionModel(config)
model.to(config.device)
print(f"Model initialized on {config.device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

if config.device == "cuda" and config.compile_decoder:
    model.decoder = torch.compile(model.decoder, mode="default")


inference_engine = InferenceEngine(model, config, word2idx, idx2word)


def generate_csv(epoch):
    model.eval()
    img_results = {}

    print(
        f"Generating beam search candidates for epoch {epoch+1} (beam_width={config.BEAM_WIDTH})..."
    )
    for batch_idx, (features, _, batch_desc_tokens, batch_desc_scores) in enumerate(
        tqdm.tqdm(test_dataloader, desc=f"Generating epoch {epoch+1}")
    ):
        features = features.to(config.device, non_blocking=True)
        batch_desc_tokens = batch_desc_tokens.to(config.device, non_blocking=True)
        batch_desc_scores = batch_desc_scores.to(config.device, non_blocking=True)

        with torch.inference_mode():
            desc_memory_batch = model.desc_token_encoder(batch_desc_tokens, batch_desc_scores)
            enc_out_batch = model.encoder(features, desc_memory_batch)
        del features, batch_desc_tokens, batch_desc_scores, desc_memory_batch

        for i in range(enc_out_batch.shape[0]):
            dataset_idx = batch_idx * config.batch_size + i
            img_path = test_dataset.samples[dataset_idx][0]
            img_name = os.path.basename(img_path)

            if img_name not in img_results:
                top_k_captions = inference_engine.beam_search(
                    encoder_outputs=enc_out_batch[i : i + 1],
                    beam_width=config.BEAM_WIDTH,
                )
                img_results[img_name] = {
                    "preds": top_k_captions,
                    "refs": test_references.get(img_name, []),
                }

    print(f"Total images: {len(img_results)}")

    rows = []
    for image_id, (img_name, data) in enumerate(img_results.items()):
        for rank, (cap, score) in enumerate(data["preds"]):
            row = {
                "image_id": image_id,
                "image_name": img_name,
                "candidate_rank": rank,
                "caption": cap,
                "beam_score": score,
            }
            for j, gt in enumerate(data["refs"], 1):
                row[f"ground_truth_{j}"] = gt
            rows.append(row)

    csv_path = f"outputs/flickr30k/beam_candidates_epoch_{epoch+1}.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved {len(img_results)} images ({len(rows)} candidates) to {csv_path}")

    del img_results, rows
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


trainer = Trainer(model, config, train_dataloader, valid_dataloader)

CHECKPOINT_PATH = "outputs/flickr30k/checkpoint_latest.pt"
MODEL_PATH = "outputs/flickr30k/model_checkpoint_epoch_{epoch}.pth"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
RESUME_FROM = None

trainer.train(
    num_epochs=config.epochs,
    csv_callback=generate_csv,
    checkpoint_path=CHECKPOINT_PATH,
    model_path=MODEL_PATH,
    resume_from=RESUME_FROM,
    stop_after_epoch=config.stop_after_epoch,
)

trainer.plot_metrics()
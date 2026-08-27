# !pip install -q thop torchvision transformers
# !pip install ultralytics -q
# !git clone https://github.com/Artanic30/HOICLIP.git
# !pip install -q ftfy regex tqdm opencv-python scipy cython pycocotools
# !pip install -q git+https://github.com/openai/CLIP.git

# %%
import math
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from thop import profile, clever_format

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

N_WARMUP = 20
N_RUNS   = 200

CLIP_MODEL_ID     = "openai/clip-vit-large-patch14"
CLIP_IMAGE_SIZE   = 224
CLIP_FLOPS_GFLOPS = 162.2

VOCAB_SIZE   = 10101
D_MODEL      = 768
N_HEADS      = 12
NUM_DESCS    = 5
MAX_DESC_LEN = 20
MAXLEN       = 46
FEATURE_DIM  = 1024
GRID_SIZE    = 16
NUM_PATCHES  = GRID_SIZE * GRID_SIZE
ROPE_DIM     = 256
LEARNED_DIM  = 256
ENC_LAYERS   = 2
DEC_LAYERS   = 3

BEAM_WIDTH     = 5
LENGTH_PENALTY = 0.6
PAD_IDX        = 0
START_IDX      = 1
END_IDX        = 2
CAP_N_WARMUP   = 3
CAP_N_RUNS     = 20

YOLO_WEIGHTS    = "yolo12x.pt"
YOLO_IMAGE_SIZE = 640

RELTR_IMAGE_SIZE = 800

HOICLIP_CLIP_MODEL_ID = "ViT-B/32"
HOICLIP_IMAGE_H       = 800
HOICLIP_IMAGE_W       = 1333
HOICLIP_CLIP_SIZE     = 224

BLIP_MODEL_ID   = "Salesforce/blip-itm-base-coco"
BLIP_IMAGE_SIZE = 384
BLIP_TEXT_LEN   = 20
BLIP_VOCAB_SIZE = 30000
BLIP_NUM_DESCS  = 5

QWEN_MODEL_ID        = "Qwen/Qwen3-4B"
QWEN_BATCH           = 1
QWEN_TOKENS_PER_DESC = 17
QWEN_DESC_MAX_TOK    = QWEN_BATCH * NUM_DESCS * QWEN_TOKENS_PER_DESC + 1

RERANK_BATCH        = 1
RERANK_MAX_TOK      = 1024
RERANK_NUM_CAPTIONS = 5
K_PROMPT            = 3

rows              = []
caption_breakdown = []


def count_params(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_time(model: nn.Module, dummy_inputs, n_warmup=N_WARMUP, n_runs=N_RUNS, batch_size=BATCH_SIZE):
    model.eval()
    with torch.no_grad():
        if isinstance(dummy_inputs, (list, tuple)):
            for _ in range(n_warmup):
                model(*dummy_inputs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(*dummy_inputs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
        else:
            for _ in range(n_warmup):
                model(dummy_inputs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                model(dummy_inputs)
            if DEVICE == "cuda":
                torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    return elapsed / n_runs / batch_size * 1000


def measure_flops(model: nn.Module, dummy_inputs, batch_size: int = 1):
    """Measure total MACs for batch_size samples; divide outside to get per-image."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(dummy_inputs, (list, tuple)):
            macs, params = profile(model, inputs=dummy_inputs, verbose=False)
        else:
            macs, params = profile(model, inputs=(dummy_inputs,), verbose=False)
    return macs, params


from transformers import CLIPVisionModel

class CLIPExtractorWrapper(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
    def forward(self, x):
        out = self.clip(pixel_values=x)
        return out.last_hidden_state[:, 1:, :]

_clip_raw      = CLIPVisionModel.from_pretrained(CLIP_MODEL_ID)
clip_extractor = CLIPExtractorWrapper(_clip_raw).to(DEVICE).eval()
for p in clip_extractor.parameters():
    p.requires_grad = False

clip_total, _ = count_params(clip_extractor)

dummy_clip = torch.randn(BATCH_SIZE, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE).to(DEVICE)
clip_ms    = measure_time(clip_extractor, dummy_clip, batch_size=BATCH_SIZE)

rows.append(("CLIP-ViT-L/14", clip_total, CLIP_FLOPS_GFLOPS, clip_ms,
             "visual feature extraction"))

del clip_extractor, _clip_raw, dummy_clip
torch.cuda.empty_cache() if DEVICE == "cuda" else None


class RotaryPositionalEncoding2D(nn.Module):
    def __init__(self, dim, height=8, width=8):
        super().__init__()
        assert dim % 4 == 0
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim).float() / half_dim))
        h_pos = torch.arange(height).float()
        w_pos = torch.arange(width).float()
        h_freqs = torch.einsum("i,j->ij", h_pos, inv_freq)
        w_freqs = torch.einsum("i,j->ij", w_pos, inv_freq)
        h_emb = h_freqs.unsqueeze(1).repeat(1, width, 1).reshape(-1, h_freqs.shape[-1])
        w_emb = w_freqs.unsqueeze(0).repeat(height, 1, 1).reshape(-1, w_freqs.shape[-1])
        cos_emb = torch.cat([h_emb.cos(), w_emb.cos()], dim=-1)
        sin_emb = torch.cat([h_emb.sin(), w_emb.sin()], dim=-1)
        if cos_emb.shape[-1] < dim:
            pad = dim - cos_emb.shape[-1]
            cos_emb = F.pad(cos_emb, (0, pad), value=1.0)
            sin_emb = F.pad(sin_emb, (0, pad), value=0.0)
        self.register_buffer("cos_emb", cos_emb)
        self.register_buffer("sin_emb", sin_emb)
    def forward(self, x):
        return (x * self.cos_emb.unsqueeze(0)) + (self._rotate_half(x) * self.sin_emb.unsqueeze(0))
    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)


class Learned2DPE(nn.Module):
    def __init__(self, dim, height=8, width=8):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, height * width, dim))
    def forward(self, x):
        return x + self.pe


class Sinusoidal2DPE(nn.Module):
    def __init__(self, dim, height=8, width=8):
        super().__init__()
        pe = torch.zeros(height * width, dim)
        pos = torch.arange(0, height * width).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_emb", emb.cos())
        self.register_buffer("sin_emb", emb.sin())
    def forward(self, x):
        seq_len = x.size(1)
        return (x * self.cos_emb[:seq_len].unsqueeze(0)) + (
            self._rotate_half(x) * self.sin_emb[:seq_len].unsqueeze(0))
    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)


class DescTokenEncoder(nn.Module):
    def __init__(self, shared_embedding, d_model):
        super().__init__()
        self.embedding = shared_embedding
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
    def forward(self, desc_tokens, desc_scores):
        B, num_descs, max_desc_len = desc_tokens.shape
        flat = desc_tokens.reshape(B * num_descs, max_desc_len)
        emb  = self.embedding(flat)
        mask = (flat != 0).float().unsqueeze(-1)
        desc_emb = (emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        desc_emb = desc_emb.reshape(B, num_descs, -1) * desc_scores.unsqueeze(-1)
        return self.proj(desc_emb)


class DescAuxHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
    def forward(self, enc_out):
        return F.normalize(self.proj(enc_out.mean(dim=1)), dim=-1)


class DescVisualCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, desc_emb, visual_features):
        enhanced, _ = self.cross_attn(query=desc_emb, key=visual_features, value=visual_features)
        return self.norm(desc_emb + enhanced)


class Encoder(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, feature_dim=FEATURE_DIM):
        super().__init__()
        rope_dim    = ROPE_DIM
        learned_dim = LEARNED_DIM
        sin_dim     = d_model - rope_dim - learned_dim
        self.rope_dim    = rope_dim
        self.learned_dim = learned_dim

        self.rope_2d       = RotaryPositionalEncoding2D(rope_dim, height=GRID_SIZE, width=GRID_SIZE)
        self.learned_2d    = Learned2DPE(learned_dim, height=GRID_SIZE, width=GRID_SIZE)
        self.sinusoidal_2d = Sinusoidal2DPE(sin_dim, height=GRID_SIZE, width=GRID_SIZE)

        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True),
            num_layers=ENC_LAYERS,
        )
        self.attention_pool        = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.desc_visual_cross_attn = DescVisualCrossAttention(d_model, n_heads)

    def forward(self, combined_features, desc_memory=None):
        proj = self.feature_proj(combined_features)
        x1 = self.rope_2d(proj[..., :self.rope_dim])
        x2 = self.learned_2d(proj[..., self.rope_dim: self.rope_dim + self.learned_dim])
        x3 = self.sinusoidal_2d(proj[..., self.rope_dim + self.learned_dim:])
        features = torch.cat([x1, x2, x3], dim=-1)
        if desc_memory is not None:
            desc_memory = self.desc_visual_cross_attn(desc_memory, features)
            features = torch.cat([features, desc_memory], dim=1)
        attn_features, _ = self.attention_pool(features, features, features)
        return self.encoder(attn_features)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS, shared_embedding=None):
        super().__init__()
        self.embedding = shared_embedding if shared_embedding is not None \
            else nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.rope    = RotaryPositionalEncoding(d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True),
            num_layers=DEC_LAYERS,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_outputs, tgt_mask=None, tgt_key_padding_mask=None):
        x = self.rope(self.embedding(x))
        dec_out = self.decoder(tgt=x, memory=encoder_outputs,
                               tgt_mask=tgt_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out(dec_out)


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads_enc=N_HEADS, n_heads_dec=N_HEADS,
                 num_descs=NUM_DESCS, max_desc_len=MAX_DESC_LEN):
        super().__init__()
        self.shared_embedding  = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.encoder           = Encoder(d_model, n_heads_enc, feature_dim=FEATURE_DIM)
        self.desc_token_encoder = DescTokenEncoder(self.shared_embedding, d_model)
        self.desc_aux_head      = DescAuxHead(d_model)
        self.decoder           = Decoder(vocab_size, d_model, n_heads_dec,
                                         shared_embedding=self.shared_embedding)

    def forward(self, image_features, input_tokens, desc_tokens, desc_scores,
                tgt_mask=None, tgt_key_padding_mask=None):
        desc_memory      = self.desc_token_encoder(desc_tokens, desc_scores)
        encoder_outputs = self.encoder(image_features, desc_memory)
        logits          = self.decoder(input_tokens, encoder_outputs, tgt_mask, tgt_key_padding_mask)
        pred_desc_emb    = self.desc_aux_head(encoder_outputs)
        return logits, pred_desc_emb


class BeamSearchRunner(nn.Module):
    def __init__(self, model, beam_width=BEAM_WIDTH, max_len=MAXLEN):
        super().__init__()
        self.model      = model
        self.beam_width = beam_width
        self.max_len    = max_len

    def forward(self, image_features, desc_tokens, desc_scores):
        device = image_features.device

        desc_memory      = self.model.desc_token_encoder(desc_tokens, desc_scores)
        encoder_outputs = self.model.encoder(image_features, desc_memory)

        beams     = [([START_IDX], 0.0)]
        completed = []

        for _ in range(self.max_len):
            active = [(s, sc) for s, sc in beams if s[-1] != END_IDX]
            completed.extend((s, sc) for s, sc in beams if s[-1] == END_IDX)
            if not active:
                break

            seq_len = len(active[0][0])
            token_batch = torch.full((len(active), seq_len), PAD_IDX,
                                     dtype=torch.long, device=device)
            for i, (seq, _) in enumerate(active):
                token_batch[i] = torch.tensor(seq, device=device)

            enc = encoder_outputs.expand(len(active), -1, -1)
            x   = self.model.decoder.rope(self.model.decoder.embedding(token_batch))
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
            )
            logits = self.model.decoder.out(
                self.model.decoder.decoder(x, enc, tgt_mask=causal_mask)
            )
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            top_log_probs, top_indices = log_probs.topk(self.beam_width, dim=-1)

            candidates = []
            for i, (seq, score) in enumerate(active):
                for j in range(self.beam_width):
                    candidates.append((seq + [top_indices[i, j].item()],
                                       score + top_log_probs[i, j].item()))

            candidates.sort(key=lambda c: c[1] / (len(c[0]) ** LENGTH_PENALTY),
                            reverse=True)
            beams = candidates[:self.beam_width]

            if all(s[-1] == END_IDX for s, _ in beams):
                break

        return beams

caption_model = CaptionModel(
    vocab_size=VOCAB_SIZE, d_model=D_MODEL,
    n_heads_enc=N_HEADS, n_heads_dec=N_HEADS,
    num_descs=NUM_DESCS, max_desc_len=MAX_DESC_LEN,
).to(DEVICE).eval()

dummy_feat   = torch.randn(BATCH_SIZE, NUM_PATCHES, FEATURE_DIM).to(DEVICE)
dummy_descs   = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, NUM_DESCS, MAX_DESC_LEN)).to(DEVICE)
dummy_scores = torch.rand(BATCH_SIZE, NUM_DESCS).to(DEVICE)

cap_total, cap_train = count_params(caption_model)

beam_runner = BeamSearchRunner(caption_model).to(DEVICE).eval()

cap_ms = measure_time(
    beam_runner,
    (dummy_feat, dummy_descs, dummy_scores),
    n_warmup=CAP_N_WARMUP, n_runs=CAP_N_RUNS,
    batch_size=BATCH_SIZE,
)

cap_macs, _ = profile(
    beam_runner,
    inputs=(dummy_feat, dummy_descs, dummy_scores),
    verbose=False,
)
cap_gflops = cap_macs * 2 / 1e9 / BATCH_SIZE

rows.append(("CaptionModel (beam search)", cap_total, cap_gflops, cap_ms,
             f"caption generation (beam={BEAM_WIDTH}, max_len={MAXLEN})"))

_seen_ptrs = set()
for name, sub in caption_model.named_children():
    unique_p = sum(x.numel() for x in sub.parameters()
                   if id(x) not in _seen_ptrs)
    _seen_ptrs.update(id(x) for x in sub.parameters())
    caption_breakdown.append((name, unique_p))


from ultralytics import YOLO as _YOLO
_yolo       = _YOLO(YOLO_WEIGHTS)
_yolo_model = _yolo.model.to(DEVICE).eval()

yolo_total, _ = count_params(_yolo_model)

_dummy_yolo = torch.randn(BATCH_SIZE, 3, YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE).to(DEVICE)
yolo_macs, _ = measure_flops(_yolo_model, _dummy_yolo)
yolo_ms      = measure_time(_yolo_model, _dummy_yolo, batch_size=BATCH_SIZE)

rows.append(("YOLOv12-X", yolo_total, yolo_macs * 2 / 1e9 / BATCH_SIZE, yolo_ms,
                     "object detection"))
del _yolo_model, _yolo, _dummy_yolo
torch.cuda.empty_cache() if DEVICE == "cuda" else None


class _RelTRApprox(nn.Module):
    def __init__(self):
        super().__init__()
        D = 256
        res50 = torchvision.models.resnet50(weights=None)
        self.backbone    = nn.Sequential(*list(res50.children())[:-2])
        self.input_proj  = nn.Conv2d(2048, D, kernel_size=1)
        enc_layer = nn.TransformerEncoderLayer(d_model=D, nhead=8, dim_feedforward=2048,
                                               dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=6)
        dec_layer = nn.TransformerDecoderLayer(d_model=D, nhead=8, dim_feedforward=2048,
                                               dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=6)
        self.entity_queries  = nn.Embedding(100, D)
        self.triplet_queries = nn.Embedding(200, D)
        self.sub_cls  = nn.Linear(D, 151)
        self.obj_cls  = nn.Linear(D, 151)
        self.rel_cls  = nn.Linear(D, 51)
        self.sub_bbox = nn.Linear(D, 4)
        self.obj_bbox = nn.Linear(D, 4)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.input_proj(feat)
        B    = feat.shape[0]
        src  = feat.flatten(2).permute(0, 2, 1)
        mem  = self.encoder(src)
        q_e  = self.entity_queries.weight.unsqueeze(0).expand(B, -1, -1)
        q_t  = self.triplet_queries.weight.unsqueeze(0).expand(B, -1, -1)
        out  = self.decoder(torch.cat([q_e, q_t], dim=1), mem)
        return self.sub_cls(out[:, :100]), self.obj_cls(out[:, :100]), self.rel_cls(out[:, 100:])

_reltr = _RelTRApprox().to(DEVICE).eval()
reltr_total, _ = count_params(_reltr)

_dummy_reltr = torch.randn(BATCH_SIZE, 3, RELTR_IMAGE_SIZE, RELTR_IMAGE_SIZE).to(DEVICE)
reltr_macs, _ = measure_flops(_reltr, _dummy_reltr)
reltr_ms      = measure_time(_reltr, _dummy_reltr, batch_size=BATCH_SIZE)

rows.append(("RelTR", reltr_total, reltr_macs * 2 / 1e9 / BATCH_SIZE, reltr_ms,
                     "visual relation detection"))
del _reltr, _dummy_reltr
torch.cuda.empty_cache() if DEVICE == "cuda" else None


import clip as _openai_clip

class _HOICLIPApprox(nn.Module):
    def __init__(self):
        super().__init__()
        D = 256
        res50 = torchvision.models.resnet50(weights=None)
        self.backbone   = nn.Sequential(*list(res50.children())[:-2])
        self.input_proj = nn.Conv2d(2048, D, 1)
        enc_layer = nn.TransformerEncoderLayer(d_model=D, nhead=8, dim_feedforward=2048,
                                               dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=6)
        dec_layer = nn.TransformerDecoderLayer(d_model=D, nhead=8, dim_feedforward=2048,
                                               dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=3)
        self.query_embed = nn.Embedding(64, D)
        clip_model, _    = _openai_clip.load(HOICLIP_CLIP_MODEL_ID, device="cpu")
        self.clip_visual = clip_model.visual
        self.clip_text   = clip_model.transformer
        self.hoi_cls   = nn.Linear(D, 600)
        self.obj_cls   = nn.Linear(D, 80)
        self.sub_bbox  = nn.Linear(D, 4)
        self.obj_bbox  = nn.Linear(D, 4)
        self.verb_proj = nn.Linear(512, D)
        self.obj_proj  = nn.Linear(512, D)

    def forward(self, x, clip_img):
        feat = self.backbone(x)
        feat = self.input_proj(feat)
        B    = feat.shape[0]
        src  = feat.flatten(2).permute(0, 2, 1)
        mem  = self.encoder(src)
        q    = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        out  = self.decoder(q, mem)
        clip_feat = self.clip_visual(clip_img)
        return self.hoi_cls(out), self.obj_cls(out), clip_feat

_hoiclip = _HOICLIPApprox().to(DEVICE).eval()
hoiclip_total, _ = count_params(_hoiclip)

_dummy_hoi_img  = torch.randn(BATCH_SIZE, 3, HOICLIP_IMAGE_H, HOICLIP_IMAGE_W).to(DEVICE)
_dummy_hoi_clip = torch.randn(BATCH_SIZE, 3, HOICLIP_CLIP_SIZE, HOICLIP_CLIP_SIZE).to(DEVICE)

hoiclip_macs, _ = measure_flops(_hoiclip, (_dummy_hoi_img, _dummy_hoi_clip))
hoiclip_ms      = measure_time(_hoiclip, (_dummy_hoi_img, _dummy_hoi_clip),
                                batch_size=BATCH_SIZE)

rows.append(("HOICLIP", hoiclip_total, hoiclip_macs * 2 / 1e9 / BATCH_SIZE, hoiclip_ms,
                     "human-object interaction"))
del _hoiclip, _dummy_hoi_img, _dummy_hoi_clip
torch.cuda.empty_cache() if DEVICE == "cuda" else None


from transformers import BlipForImageTextRetrieval as _BLIP

_blip = _BLIP.from_pretrained(BLIP_MODEL_ID).to(DEVICE).eval()
blip_total, _ = count_params(_blip)

_dummy_blip_img  = torch.randn(BATCH_SIZE, 3, BLIP_IMAGE_SIZE, BLIP_IMAGE_SIZE).to(DEVICE)
_dummy_blip_ids  = torch.randint(0, BLIP_VOCAB_SIZE, (BATCH_SIZE, BLIP_TEXT_LEN)).to(DEVICE)
_dummy_blip_mask = torch.ones(BATCH_SIZE, BLIP_TEXT_LEN, dtype=torch.long).to(DEVICE)

class _BlipWrapper(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, pv, ids, mask):
        return self.m(pixel_values=pv, input_ids=ids,
                      attention_mask=mask, use_itm_head=True)

_blip_wrap      = _BlipWrapper(_blip)
blip_ms_per_desc = measure_time(_blip_wrap, (_dummy_blip_img, _dummy_blip_ids, _dummy_blip_mask),
                                batch_size=BATCH_SIZE)
blip_ms_per_img = blip_ms_per_desc * BLIP_NUM_DESCS

rows.append(("BLIP-ITM-base-coco", blip_total, None, blip_ms_per_img,
                     f"description quality scoring (x{BLIP_NUM_DESCS} descs/img)"))
del _blip, _blip_wrap
torch.cuda.empty_cache() if DEVICE == "cuda" else None


from transformers import AutoModelForCausalLM, AutoTokenizer

_qwen_tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
_qwen     = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
)
_qwen.eval()
qwen_total, _ = count_params(_qwen)

_DUMMY_PROMPT = (
    f"For image below, generate exactly {NUM_DESCS} semantic descriptions. "
    "Each description must contain more than 5 words, be specific, diverse, and faithful to the visual context. \n\n"
    + "\n".join([
        f"Image {i}:\n  Objects  : person 2, car 1\n  Actions  : riding bicycle\n  Relations: person - riding - bicycle"
        for i in range(QWEN_BATCH)
    ])
    + "\n\nOUTPUT FORMAT (strictly follow, one line per image):\n"
    + "0: description1 | description2 | description3 | description4 | description5\n...\n"
    + "Output ONLY the numbered lines, nothing else."
)
_msgs = [
    {"role": "system", "content": "You are a precise image scene describer. Follow the output format exactly."},
    {"role": "user",   "content": _DUMMY_PROMPT},
]
_inputs    = _qwen_tok.apply_chat_template(
    _msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=False
)
_input_ids = (_inputs.input_ids if hasattr(_inputs, "input_ids") else _inputs).to(_qwen.device)
_attn_mask = torch.ones_like(_input_ids)

torch.cuda.synchronize() if DEVICE == "cuda" else None
_t0 = time.perf_counter()
with torch.no_grad():
    _out = _qwen.generate(
        _input_ids, attention_mask=_attn_mask,
        max_new_tokens=QWEN_DESC_MAX_TOK, do_sample=False,
        pad_token_id=_qwen_tok.eos_token_id
    )
torch.cuda.synchronize() if DEVICE == "cuda" else None
qwen_batch_ms   = (time.perf_counter() - _t0) * 1000
qwen_ms_per_img = qwen_batch_ms / QWEN_BATCH

rows.append(("Qwen3-4B (desc gen)", qwen_total, None, qwen_ms_per_img,
                     f"semantic description gen (batch={QWEN_BATCH} imgs)"))



_DUMMY_OBJECTS   = "person 2, bicycle 1, car 1"
_DUMMY_ACTIONS   = ["ride bicycle", "hold bicycle", "sit on bicycle"]
_DUMMY_RELATIONS = ["person riding bicycle", "person on street", "car behind person"]
_DUMMY_DESCS     = [
    ("a person is riding a bicycle down a city street", 0.91),
    ("a cyclist rides past parked cars on the road", 0.87),
    ("a man wearing a helmet rides a bike outdoors", 0.83),
]

_rerank_context = "\n".join([
    f"Image {i}: "
    + " | ".join([
        f"Objects: {_DUMMY_OBJECTS}",
        "Actions: "   + "; ".join(_DUMMY_ACTIONS[:K_PROMPT]),
        "Relations: " + "; ".join(_DUMMY_RELATIONS[:K_PROMPT]),
        "Scene descs: " + "; ".join(f"{d} ({s:.2f})" for d, s in _DUMMY_DESCS[:K_PROMPT]),
    ])
    for i in range(RERANK_BATCH)
])

_rerank_images_section = []
for _i in range(RERANK_BATCH):
    _rerank_images_section.append(f"Image {_i}:")
    for _j in range(RERANK_NUM_CAPTIONS):
        _rerank_images_section.append(f"  {_j + 1}. a person is riding a bicycle on a street")
    _rerank_images_section.append("")

_RERANK_CONTEXT_BLOCK = f"""DETECTION EVIDENCE (3 vision models on the actual image):
{_rerank_context}

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

_RERANK_PROMPT = f"""IMAGE CAPTION RERANKING TASK

Select the SINGLE BEST caption for each image using the visual detection evidence.

{_RERANK_CONTEXT_BLOCK}

CAPTIONS TO EVALUATE:

{chr(10).join(_rerank_images_section)}

OUTPUT FORMAT: Respond with ONLY image ID and selected caption number, one per line.
Format: IMAGE_ID: NUMBER

Example:
0: 2
1: 3"""

_RERANK_SYSTEM = (
    "You are a precise image caption evaluator. Use YOLO object detections (highly reliable), "
    "HOI action detections, RelTR relation detections, and BLIP-scored Qwen scene descriptions "
    "(contextual) to select the caption that best matches the actual image. Penalize hallucinated "
    "objects, wrong object labels, incorrect human activities, and <unk> tokens. Output ONLY in "
    "IMAGE_ID: NUMBER format, one line per image. No explanations."
)

_rerank_msgs = [
    {"role": "system", "content": _RERANK_SYSTEM},
    {"role": "user",   "content": _RERANK_PROMPT},
]
_rerank_ids = _qwen_tok.apply_chat_template(
    _rerank_msgs, return_tensors="pt", add_generation_prompt=True, enable_thinking=False
)
_rerank_input_ids = (_rerank_ids.input_ids if hasattr(_rerank_ids, "input_ids") else _rerank_ids).to(_qwen.device)
_rerank_attn_mask = torch.ones_like(_rerank_input_ids)

torch.cuda.synchronize() if DEVICE == "cuda" else None
_t0 = time.perf_counter()
with torch.no_grad():
    _rerank_out = _qwen.generate(
        _rerank_input_ids, attention_mask=_rerank_attn_mask,
        max_new_tokens=RERANK_MAX_TOK, do_sample=False,
        pad_token_id=_qwen_tok.eos_token_id
    )
torch.cuda.synchronize() if DEVICE == "cuda" else None
qwen_rerank_batch_ms   = (time.perf_counter() - _t0) * 1000
qwen_rerank_ms_per_img = qwen_rerank_batch_ms / RERANK_BATCH

rows.append(("Qwen3-4B (LLM Reranker)", qwen_total, None, qwen_rerank_ms_per_img,
                     f"caption reranking (batch={RERANK_BATCH} imgs)"))
del _qwen, _qwen_tok
torch.cuda.empty_cache() if DEVICE == "cuda" else None


def fmt_flops(gflops):
    return f"{gflops:.2f} G" if gflops is not None else "N/A*"


RULE = "=" * 96

print("\n\n" + RULE)
print("  COMPUTATIONAL COST REPORT")
print(RULE)
print(f"  Device: {DEVICE}   |   Batch size: {BATCH_SIZE}\n")

print(f"  {'Model':<30} {'Params (M)':>12} {'FLOPs/img':>14} {'Time (ms/img)':>15}  Role")
print("  " + "-" * 92)
for _name, _params, _gflops, _ms, _role in rows:
    print(f"  {_name:<30} {_params/1e6:>12.2f} {fmt_flops(_gflops):>14} {_ms:>15.2f}  {_role}")

print("\n  CaptionModel breakdown (shared_embedding counted once)\n")
for _name, _params in caption_breakdown:
    print(f"  {_name:<30} {_params/1e6:>12.2f} M")
print("  " + "-" * 45)
print(f"  {'TOTAL (unique params)':<30} {cap_total/1e6:>12.2f} M")


import os, io, re, gzip, html, random, warnings, urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

try:
    import regex as regex_lib
except Exception:
    regex_lib = None
try:
    import ftfy
except Exception:
    ftfy = None

@dataclass
class Config:
    train_dir: str = "/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/StyleClassificationIndoors/StyleClassificationIndoors/train"
    test_dir: str = "/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/StyleClassificationIndoors/StyleClassificationIndoors/test"
    class_map_path: str = "/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/StyleClassificationIndoors/StyleClassificationIndoors/class_mapping.txt"
    sample_submission: str = "/kaggle/input/competitions/cse-281-spring-26-scene-style-classification/sample_submission.csv"
    seed: int = 42
    num_classes: int = 17
    asset_dir: str = "/kaggle/working/clip_assets"
    clip_checkpoint_name: str = "ViT-B-16.pt"
    bpe_name: str = "bpe_simple_vocab_16e6.txt.gz"
    clip_checkpoint_url: str = "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
    bpe_url: str = "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
    image_size: int = 224
    image_batch_size: int = 64
    n_folds: int = 3
    mlp_epochs: int = 90
    mlp_batch_size: int = 256
    mlp_lr: float = 1e-3
    mlp_weight_decay: float = 1e-3
    label_smoothing: float = 0.05
    early_stop_patience: int = 12
    text_logit_scale: float = 12.0
    zero_shot_weight: float = 0.0
    tta_views: Tuple[str, ...] = ("center", "flip", "crop92", "crop92_flip", "crop86")
    submission_no_tta: str = "/kaggle/working/submission_no_tta.csv"
    submission_tta: str = "/kaggle/working/submission.csv"

CLASS_NAMES = [
    "asian", "boho", "coastal", "contemporary", "craftsman", "eclectic",
    "farmhouse", "french-country", "industrial", "mediterranean", "minimalist",
    "modern", "scandinavian", "shabby-chic-style", "southwestern", "tropical", "victorian",
]

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    if not torch.cuda.is_available():
        print("Using CPU: CUDA not available.")
        return torch.device("cpu")
    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print("CUDA detected:", name, "capability:", cap)
        emb = torch.nn.Embedding(10, 4).cuda()
        idx = torch.tensor([1, 2, 3], dtype=torch.long, device="cuda")
        _ = emb(idx); torch.cuda.synchronize()
        print("Using GPU:", name)
        return torch.device("cuda")
    except Exception as e:
        print("[WARN] CUDA visible but unusable:", repr(e))
        print("[WARN] Falling back to CPU.")
        try: torch.cuda.empty_cache()
        except Exception: pass
        return torch.device("cpu")


def search_file(filename):
    candidates = [Path("/kaggle/working") / filename, Path("/kaggle/working/clip_assets") / filename,
                  Path("/root/.cache/clip") / filename, Path.home() / ".cache" / "clip" / filename]
    for root in ["/kaggle/input", "/kaggle/working"]:
        rp = Path(root)
        if rp.exists():
            candidates.extend(list(rp.rglob(filename)))
    for p in candidates:
        if p.exists() and p.is_file():
            print("Found local asset:", p)
            return str(p)
    return None


def download_file(url, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("Downloading:", url, "->", dst)
    try:
        urllib.request.urlretrieve(url, dst)
        print("Download complete:", dst)
        return dst
    except Exception as e:
        raise RuntimeError(
            "Could not download required CLIP asset. If Kaggle internet is off, run once with internet enabled or cache files in /kaggle/working/clip_assets. Reason: " + repr(e)
        )


def get_clip_assets(cfg):
    os.makedirs(cfg.asset_dir, exist_ok=True)
    ckpt = search_file(cfg.clip_checkpoint_name)
    if ckpt is None:
        ckpt = download_file(cfg.clip_checkpoint_url, os.path.join(cfg.asset_dir, cfg.clip_checkpoint_name))
    bpe = search_file(cfg.bpe_name)
    if bpe is None:
        bpe = download_file(cfg.bpe_url, os.path.join(cfg.asset_dir, cfg.bpe_name))
    print("Using CLIP checkpoint:", ckpt)
    print("Using BPE vocab:", bpe)
    return ckpt, bpe


def load_class_mapping(path):
    c2i, i2c = {}, {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line: continue
                name, label = line.split(":")
                idx = int(label.strip()); name = name.strip()
                c2i[name] = idx; i2c[idx] = name
        print("Loaded class mapping from file.")
        return c2i, i2c
    for i, name in enumerate(CLASS_NAMES): c2i[name] = i; i2c[i] = name
    return c2i, i2c


def collect_train_paths(train_dir, class_to_idx):
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths, labels = [], []
    for cname, cid in sorted(class_to_idx.items(), key=lambda x: x[1]):
        d = Path(train_dir) / cname
        if not d.exists():
            print("[WARN] Missing class folder:", d); continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in valid:
                paths.append(str(p)); labels.append(int(cid))
    return paths, labels


def get_test_paths_from_submission(cfg):
    sample = pd.read_csv(cfg.sample_submission)
    image_col = "ImageName" if "ImageName" in sample.columns else sample.columns[0]
    label_col = "ClassLabel" if "ClassLabel" in sample.columns else sample.columns[1]
    exts = ["", ".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    paths = []
    for name in sample[image_col].astype(str).tolist():
        found = None
        for ext in exts:
            p = Path(cfg.test_dir) / f"{name}{ext}"
            if p.exists(): found = str(p); break
        paths.append(found if found is not None else str(Path(cfg.test_dir) / name))
    return sample, paths, image_col, label_col


def safe_open_rgb(path):
    try:
        with open(path, "rb") as f: raw = f.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        print("[WARN] Bad image:", path, repr(e))
        return Image.new("RGB", (224, 224), (128, 128, 128))


def center_crop_fraction(img, frac):
    w, h = img.size
    nw, nh = max(1, int(w * frac)), max(1, int(h * frac))
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


class ClipTransform:
    def __init__(self, image_size=224, view="center"):
        self.image_size = image_size; self.view = view
    def __call__(self, img):
        img = ImageOps.fit(img, (self.image_size, self.image_size), method=Image.BICUBIC, centering=(0.5, 0.5))
        if self.view == "flip": img = ImageOps.mirror(img)
        elif self.view == "crop92":
            img = center_crop_fraction(img, 0.92).resize((self.image_size, self.image_size), Image.BICUBIC)
        elif self.view == "crop92_flip":
            img = center_crop_fraction(img, 0.92).resize((self.image_size, self.image_size), Image.BICUBIC)
            img = ImageOps.mirror(img)
        elif self.view == "crop86":
            img = center_crop_fraction(img, 0.86).resize((self.image_size, self.image_size), Image.BICUBIC)
        arr = np.asarray(img).astype("float32") / 255.0
        arr = (arr - np.array(CLIP_MEAN, dtype="float32")) / np.array(CLIP_STD, dtype="float32")
        arr = np.transpose(arr, (2, 0, 1))
        return torch.tensor(arr, dtype=torch.float32)


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform): self.paths = paths; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx): return self.transform(safe_open_rgb(self.paths[idx]))

# ---------------- manual tokenizer ----------------

def basic_clean(text):
    if ftfy is not None: text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text): return re.sub(r"\s+", " ", text).strip()

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(n) for n in cs]))

def get_pairs(word):
    pairs = set(); prev = word[0]
    for char in word[1:]: pairs.add((prev, char)); prev = char
    return pairs

class SimpleTokenizer:
    def __init__(self, bpe_path):
        self.byte_encoder = bytes_to_unicode(); self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}
        with gzip.open(bpe_path, "rt", encoding="utf-8") as f: merges = f.read().split("\n")
        merges = merges[1:49152-256-2+1]
        merges = [tuple(m.split()) for m in merges if m]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges: vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = {v:i for i,v in enumerate(vocab)}
        self.decoder = {i:v for v,i in self.encoder.items()}
        self.bpe_ranks = {merge:i for i,merge in enumerate(merges)}
        self.cache = {"<|startoftext|>":"<|startoftext|>", "<|endoftext|>":"<|endoftext|>"}
        if regex_lib is not None:
            self.pat = regex_lib.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", regex_lib.IGNORECASE)
        else:
            self.pat = None
    def bpe(self, token):
        if token in self.cache: return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs: return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks: break
            first, second = bigram; new_word = []; i = 0
            while i < len(word):
                try:
                    j = word.index(first, i); new_word.extend(word[i:j]); i = j
                except Exception:
                    new_word.extend(word[i:]); break
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second); i += 2
                else:
                    new_word.append(word[i]); i += 1
            word = tuple(new_word)
            if len(word) == 1: break
            pairs = get_pairs(word)
        word = " ".join(word); self.cache[token] = word; return word
    def encode(self, text):
        out = []; text = whitespace_clean(basic_clean(text)).lower()
        tokens = regex_lib.findall(self.pat, text) if self.pat is not None else re.findall(r"\w+|[^\w\s]+", text, flags=re.UNICODE)
        for token in tokens:
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            out.extend(self.encoder[bpe] for bpe in self.bpe(token).split(" "))
        return out

def tokenize_texts(tokenizer, texts, context_length=77):
    sot = tokenizer.encoder["<|startoftext|>"]; eot = tokenizer.encoder["<|endoftext|>"]
    result = torch.zeros(len(texts), context_length, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = [sot] + tokenizer.encode(text) + [eot]
        tokens = tokens[:context_length]
        result[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return result

# ---------------- manual CLIP modules ----------------

class QuickGELU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(1.702 * x)

class LayerNormFp32(nn.LayerNorm):
    def forward(self, x):
        dtype = x.dtype
        return super().forward(x.float()).to(dtype)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNormFp32(d_model)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(d_model, d_model * 4))
        self.mlp.add_module("gelu", QuickGELU())
        self.mlp.add_module("c_proj", nn.Linear(d_model * 4, d_model))
        self.ln_2 = LayerNormFp32(d_model)
        self.attn_mask = attn_mask
    def attention(self, x):
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]
    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__(); self.width = width; self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
    def forward(self, x):
        for block in self.resblocks: x = block(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5; grid = input_resolution // patch_size
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(grid*grid + 1, width))
        self.ln_pre = LayerNormFp32(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNormFp32(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
    def forward(self, x):
        x = self.conv1(x); x = x.reshape(x.shape[0], x.shape[1], -1); x = x.permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2); x = self.transformer(x); x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        return x @ self.proj

class ManualCLIP(nn.Module):
    def __init__(self, embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers):
        super().__init__(); self.context_length = context_length
        self.visual = VisionTransformer(image_resolution, vision_patch_size, vision_width, vision_layers, vision_width//64, embed_dim)
        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads, self.build_attention_mask())
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNormFp32(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length); mask.fill_(float("-inf")); mask.triu_(1); return mask
    def encode_image(self, image): return self.visual(image)
    def encode_text(self, text):
        x = self.token_embedding(text) + self.positional_embedding
        x = x.permute(1, 0, 2); x = self.transformer(x); x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        eot = text.argmax(dim=-1)
        return x[torch.arange(x.shape[0], device=x.device), eot] @ self.text_projection


def strip_prefixes(sd):
    out = {}
    for k, v in sd.items():
        kk = k
        for p in ["module.", "model."]:
            if kk.startswith(p): kk = kk[len(p):]
        out[kk] = v
    return out


def load_clip_checkpoint(path):
    print("Loading raw checkpoint:", path)
    try:
        obj = torch.load(path, map_location="cpu")
        sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj if isinstance(obj, dict) else obj.state_dict()
        return strip_prefixes(sd)
    except Exception:
        print("torch.load failed; trying torch.jit.load...")
        return strip_prefixes(torch.jit.load(path, map_location="cpu").state_dict())


def build_manual_clip_from_state_dict(sd):
    # Official CLIP JIT checkpoints include non-weight metadata keys.
    # They are not actual nn.Module weights, so remove them before load_state_dict.
    sd = dict(sd)
    for meta_key in ["input_resolution", "context_length", "vocab_size"]:
        if meta_key in sd:
            sd.pop(meta_key)

    embed_dim = sd["text_projection"].shape[1]
    context_length = sd["positional_embedding"].shape[0]
    vocab_size = sd["token_embedding.weight"].shape[0]
    transformer_width = sd["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in sd if k.startswith("transformer.resblocks")))
    vision_width = sd["visual.conv1.weight"].shape[0]
    vision_layers = len(set(k.split(".")[3] for k in sd if k.startswith("visual.transformer.resblocks")))
    vision_patch_size = sd["visual.conv1.weight"].shape[-1]
    grid = int((sd["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = grid * vision_patch_size
    print("Inferred CLIP:", dict(embed_dim=embed_dim, image_resolution=image_resolution, vision_layers=vision_layers, vision_width=vision_width, patch=vision_patch_size, text_layers=transformer_layers))
    model = ManualCLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # After the MLP naming fix, these should normally be zero.
    print("Manual CLIP loaded. Missing:", len(missing), "Unexpected:", len(unexpected))
    if missing:
        print("First missing:", missing[:20])
    if unexpected:
        print("First unexpected:", unexpected[:20])

    if len(missing) > 0:
        raise RuntimeError(
            "Manual CLIP did not load correctly: there are missing weight tensors. "
            "This usually means the manual architecture names do not match the checkpoint."
        )

    # Unexpected should normally be zero after metadata removal. If only harmless
    # metadata remains from a different checkpoint format, strict=False would allow it,
    # but we print it so you can verify.
    return model


def clean_class_name(name): return name.replace("-", " ").replace("_", " ")


def build_text_features(model, tokenizer, names, device):
    prompts = [
        "a photo of a {} interior design room", "a photo of a {} style room", "a photo of a {} living room",
        "an indoor room in {} style", "a home interior with {} design style", "a professionally photographed {} interior",
        "a decorated room with {} interior design", "a stylish {} room interior", "a {} interior decoration style",
    ]
    feats = []
    model.eval()
    with torch.no_grad():
        for name in names:
            c = clean_class_name(name)
            texts = [p.format(c) for p in prompts]
            tokens = tokenize_texts(tokenizer, texts, context_length=model.context_length).to(device)
            t = model.encode_text(tokens).float(); t = F.normalize(t, dim=-1)
            t = F.normalize(t.mean(dim=0), dim=0)
            feats.append(t.cpu())
    feats = torch.stack(feats)
    print("Text features:", tuple(feats.shape))
    return feats


def extract_image_features(model, paths, cfg, device, view="center"):
    ds = ImagePathDataset(paths, ClipTransform(cfg.image_size, view))
    loader = DataLoader(ds, batch_size=cfg.image_batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    feats = []; model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            f = model.encode_image(batch).float(); f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
    feats = torch.cat(feats)
    print(f"Image features [{view}]:", tuple(feats.shape))
    return feats


def make_zero_shot_probs(img, text, cfg):
    img = F.normalize(img.float(), dim=-1); text = F.normalize(text.float(), dim=-1)
    return torch.softmax((img @ text.T) * cfg.text_logit_scale, dim=-1).float()


def make_classifier_features(img, text, cfg):
    img = F.normalize(img.float(), dim=-1); text = F.normalize(text.float(), dim=-1)
    logits = (img @ text.T) * cfg.text_logit_scale
    probs = torch.softmax(logits, dim=-1)
    return torch.cat([img, logits, probs], dim=1).float()

class FeatureMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 384), nn.BatchNorm1d(384), nn.ReLU(inplace=True), nn.Dropout(0.40),
            nn.Linear(384, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.30), nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)


def train_mlp_fold(fold, x_all, y_all, train_idx, val_idx, cfg, device):
    train_ds = TensorDataset(x_all[train_idx], torch.tensor(y_all[train_idx], dtype=torch.long))
    val_ds = TensorDataset(x_all[val_idx], torch.tensor(y_all[val_idx], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=cfg.mlp_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.mlp_batch_size, shuffle=False)
    model = FeatureMLP(x_all.shape[1], cfg.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.mlp_lr, weight_decay=cfg.mlp_weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.45, patience=3, min_lr=2e-5)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    best_acc, best_loss, best_state, bad = -1.0, 1e18, None, 0
    for epoch in range(1, cfg.mlp_epochs + 1):
        model.train(); tloss = tcorr = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True); logits = model(xb); loss = crit(logits, yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tloss += float(loss.item()) * xb.size(0); tcorr += int((logits.argmax(1) == yb).sum()); total += xb.size(0)
        train_loss = tloss / max(total, 1); train_acc = tcorr / max(total, 1)
        model.eval(); vloss = vcorr = vtotal = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device); logits = model(xb); loss = crit(logits, yb)
                vloss += float(loss.item()) * xb.size(0); vcorr += int((logits.argmax(1) == yb).sum()); vtotal += xb.size(0)
        val_loss = vloss / max(vtotal, 1); val_acc = vcorr / max(vtotal, 1); sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        print(f"Fold {fold} | Epoch {epoch:03d} | loss={train_loss:.4f} | acc={train_acc:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={lr:.2e}")
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = val_acc, val_loss; bad = 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f"  Best fold {fold} saved: val_acc={best_acc:.4f}, val_loss={best_loss:.4f}")
        else:
            bad += 1
        if bad >= cfg.early_stop_patience:
            print(f"Early stopping fold {fold}."); break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval(); return model, best_acc, best_loss


def predict_mlp(model, x, cfg, device):
    loader = DataLoader(TensorDataset(x), batch_size=cfg.mlp_batch_size, shuffle=False)
    outs = []; model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device); outs.append(torch.softmax(model(xb), dim=1).cpu())
    return torch.cat(outs).numpy()


def blend(mlp, zs, cfg): return (1.0 - cfg.zero_shot_weight) * mlp + cfg.zero_shot_weight * zs


def save_submission(probs, sample, image_col, label_col, path):
    preds = np.argmax(probs, axis=1).astype(int)
    sub = sample.copy(); sub[label_col] = preds; sub[[image_col, label_col]].to_csv(path, index=False)
    print("Saved:", path, "Rows:", len(sub))
    counts = pd.Series(preds).value_counts().sort_index()
    for idx, count in counts.items(): print(f"{int(idx):2d} {CLASS_NAMES[int(idx)] if int(idx)<len(CLASS_NAMES) else '?':22s}: {count}")


def main():
    cfg = Config(); seed_everything(cfg.seed); device = get_device()
    ckpt, bpe = get_clip_assets(cfg)
    c2i, i2c = load_class_mapping(cfg.class_map_path)
    train_paths, y = collect_train_paths(cfg.train_dir, c2i); y = np.array(y, dtype=np.int64)
    print("Train:", len(train_paths))
    sample, test_paths, image_col, label_col = get_test_paths_from_submission(cfg); print("Test:", len(test_paths))
    tokenizer = SimpleTokenizer(bpe)
    sd = load_clip_checkpoint(ckpt)
    clip = build_manual_clip_from_state_dict(sd).to(device); clip.eval()
    names = [i2c[i] if i in i2c else CLASS_NAMES[i] for i in range(cfg.num_classes)]
    text_features = build_text_features(clip, tokenizer, names, device)
    print("Extracting train CLIP features...")
    train_img = extract_image_features(clip, train_paths, cfg, device, view="center")
    x_all = make_classifier_features(train_img, text_features, cfg); print("Final train features:", tuple(x_all.shape))
    print("Extracting test CLIP features for TTA...")
    test_x, test_zs = {}, {}
    for view in cfg.tta_views:
        img = extract_image_features(clip, test_paths, cfg, device, view=view)
        test_x[view] = make_classifier_features(img, text_features, cfg)
        test_zs[view] = make_zero_shot_probs(img, text_features, cfg).numpy()
    del clip
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_scores, center_probs_list, tta_probs_list = [], [], []
    center_zs = test_zs["center"]; tta_zs = np.mean([test_zs[v] for v in cfg.tta_views], axis=0)
    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        print("\n" + "="*80); print(f"FOLD {fold}/{cfg.n_folds}"); print("="*80)
        model, acc, loss = train_mlp_fold(fold, x_all, y, tr, va, cfg, device)
        fold_scores.append(acc); print(f"Fold {fold} best val_acc={acc:.4f}, val_loss={loss:.4f}")
        center_probs_list.append(predict_mlp(model, test_x["center"], cfg, device))
        view_probs = [predict_mlp(model, test_x[v], cfg, device) for v in cfg.tta_views]
        tta_probs_list.append(np.mean(view_probs, axis=0))
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Fold validation accuracies:", fold_scores); print("Mean validation accuracy:", float(np.mean(fold_scores)))
    mlp_no_tta = np.mean(center_probs_list, axis=0); mlp_tta = np.mean(tta_probs_list, axis=0)
    final_no_tta = blend(mlp_no_tta, center_zs, cfg); final_tta = blend(mlp_tta, tta_zs, cfg)
    print(f"Zero-shot blend weight = {cfg.zero_shot_weight:.2f}")
    save_submission(final_no_tta, sample, image_col, label_col, cfg.submission_no_tta)
    save_submission(final_tta, sample, image_col, label_col, cfg.submission_tta)
    print("Done. Main submission:", cfg.submission_tta)

main()
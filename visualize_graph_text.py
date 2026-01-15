# visualize_graph_text_clip_embeddings.py
# ------------------------------------------------------------
# Usage:
#   CUDA_VISIBLE_DEVICES=1 python visualize_graph_text_clip_embeddings.py
#
# It will:
#  1) load trained .pt
#  2) compute graph/text embeddings for N samples
#  3) t-SNE plot (graph vs text pairs)
#  4) similarity matrix heatmap for a small batch
# ------------------------------------------------------------

import os, json, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# =========================
# Paths (EDIT HERE)
# =========================
DATA_DIR = "data_preparation/clip_dataset"
JSONL_PATH = os.path.join(DATA_DIR, "dataset.jsonl")
GRAPH_EMB_DIR = os.path.join(DATA_DIR, "graph_emb")
CKPT_PATH = os.path.join(DATA_DIR, "graph_text_clip_vanilla.pt")  # <- your trained pt

OUT_DIR = os.path.join(DATA_DIR, "viz")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Same hyperparams as training
# =========================
MAX_LEN = 256
CLIP_DIM = 512
TEXT_WIDTH = 512
TEXT_LAYERS = 6
TEXT_HEADS = 8
DROPOUT = 0.0

# Visualization controls
N_SAMPLES = 800          # t-SNE sample size (try 200~2000)
TSNE_PERPLEXITY = 30
TSNE_SEED = 42
SIM_BATCH = 64           # similarity heatmap batch size


# =========================================================
# Dataset (robust path resolver for your jsonl format)
# =========================================================
class GraphTextDataset(Dataset):
    """
    jsonl example:
    {"id":"6", "formula":"Ni4S5Se3", "graph_emb":"clip_dataset/graph_emb/6.npy", "text":"...", ...}
    """
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.jsonl_dir = os.path.dirname(os.path.abspath(jsonl_path))
        self.items = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                it = json.loads(line)
                if "text" not in it or "id" not in it:
                    continue

                rid = str(it["id"]).strip()
                formula = str(it.get("formula", "")).strip()

                gpath = it.get("graph_emb", None)
                resolved = None

                # 1) resolve relative to jsonl dir
                if gpath:
                    cand = gpath if os.path.isabs(gpath) else os.path.join(self.jsonl_dir, gpath)
                    if os.path.exists(cand):
                        resolved = cand

                # 2) fallback: DATA_DIR/graph_emb/{id}.npy
                if resolved is None:
                    cand = os.path.join(GRAPH_EMB_DIR, f"{rid}.npy")
                    if os.path.exists(cand):
                        resolved = cand

                # 3) fallback: DATA_DIR/graph_emb/{formula}_{id}.npy
                if resolved is None and formula:
                    cand = os.path.join(GRAPH_EMB_DIR, f"{formula}_{rid}.npy")
                    if os.path.exists(cand):
                        resolved = cand

                if resolved is None:
                    continue

                it["_graph_path"] = resolved
                self.items.append(it)

        if len(self.items) == 0:
            raise RuntimeError(f"No valid items loaded from {jsonl_path}")

        print(f"[Dataset] loaded {len(self.items)} items")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        graph = np.load(it["_graph_path"]).astype(np.float32).reshape(-1)  # [G]
        text = it["text"]
        return {
            "graph": graph,
            "text": text,
            "id": str(it.get("id", idx)),
            "formula": str(it.get("formula", "")),
        }


def infer_graph_dim(ds: GraphTextDataset) -> int:
    return int(ds[0]["graph"].shape[0])


# =========================================================
# Byte-level tokenizer (same as training)
# =========================================================
PAD = 0
SOT = 2
EOT = 3
VOCAB_SIZE = 256

def encode_text_to_bytes(text: str, max_len: int):
    b = text.encode("utf-8", errors="ignore")
    ids = [SOT] + list(b)[: max_len - 2] + [EOT]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    ids = torch.tensor(ids, dtype=torch.long)
    attn = (ids != PAD).long()
    return ids, attn


def make_collate_fn(max_len: int):
    def collate(batch):
        graphs = torch.tensor(np.stack([b["graph"] for b in batch]), dtype=torch.float32)  # [B,G]

        input_ids = []
        attention_mask = []
        for b in batch:
            ids, attn = encode_text_to_bytes(b["text"], max_len)
            input_ids.append(ids)
            attention_mask.append(attn)

        return {
            "graph": graphs,
            "input_ids": torch.stack(input_ids, 0),
            "attention_mask": torch.stack(attention_mask, 0),
            "id": [b["id"] for b in batch],
            "formula": [b["formula"] for b in batch],
        }
    return collate


# =========================================================
# Model definition (must match training)
# =========================================================
class PositionalEmbedding(nn.Module):
    def __init__(self, width: int, max_seq_length: int):
        super().__init__()
        pe = torch.zeros(max_seq_length, width)
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class AttentionHead(nn.Module):
    def __init__(self, width: int, head_size: int):
        super().__init__()
        self.query = nn.Linear(width, head_size)
        self.key = nn.Linear(width, head_size)
        self.value = nn.Linear(width, head_size)
        self.scale = head_size ** -0.5

    def forward(self, x, attn_mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = (Q @ K.transpose(-2, -1)) * self.scale  # [B,L,L]
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(1)  # [B,1,L]
            scores = scores.masked_fill(mask == 0, float("-inf"))
        A = torch.softmax(scores, dim=-1)
        return A @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, width: int, n_heads: int):
        super().__init__()
        assert width % n_heads == 0
        head_size = width // n_heads
        self.heads = nn.ModuleList([AttentionHead(width, head_size) for _ in range(n_heads)])
        self.W_o = nn.Linear(width, width)

    def forward(self, x, attn_mask=None):
        out = torch.cat([h(x, attn_mask=attn_mask) for h in self.heads], dim=-1)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, width: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.mha = MultiHeadAttention(width, n_heads)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * mlp_ratio, width),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.mha(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class VanillaTextEncoder(nn.Module):
    def __init__(self, vocab_size=256, width=512, max_len=256, n_layers=6, n_heads=8, emb_dim=512, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, width)
        self.pos_emb = PositionalEmbedding(width, max_len)
        self.blocks = nn.ModuleList([TransformerEncoderLayer(width, n_heads, 4, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(width)
        self.proj = nn.Linear(width, emb_dim)

    def forward(self, input_ids, attention_mask):
        x = self.tok_emb(input_ids)
        x = self.pos_emb(x)
        for blk in self.blocks:
            x = blk(x, attn_mask=attention_mask)
        x = self.ln_f(x)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        z = self.proj(pooled)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, g):
        z = self.proj(g)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z


class GraphTextCLIP(nn.Module):
    def __init__(self, graph_in_dim: int, clip_dim: int, text_width: int, max_len: int,
                 text_layers: int, text_heads: int, dropout: float):
        super().__init__()
        self.graph_encoder = GraphEncoder(graph_in_dim, clip_dim)
        self.text_encoder = VanillaTextEncoder(
            vocab_size=VOCAB_SIZE,
            width=text_width,
            max_len=max_len,
            n_layers=text_layers,
            n_heads=text_heads,
            emb_dim=clip_dim,
            dropout=dropout,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.no_grad()
    def encode_graph(self, graph):
        return self.graph_encoder(graph)

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)


# =========================================================
# Helpers
# =========================================================
@torch.no_grad()
def compute_embeddings(model, loader, device, max_items=None):
    g_list, t_list = [], []
    ids, formulas = [], []

    seen = 0
    for batch in loader:
        graph = batch["graph"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)

        g = model.encode_graph(graph).cpu().numpy()
        t = model.encode_text(input_ids, attn).cpu().numpy()

        g_list.append(g)
        t_list.append(t)
        ids.extend(batch["id"])
        formulas.extend(batch["formula"])

        seen += g.shape[0]
        if max_items is not None and seen >= max_items:
            break

    G = np.concatenate(g_list, axis=0)
    T = np.concatenate(t_list, axis=0)
    if max_items is not None:
        G = G[:max_items]
        T = T[:max_items]
        ids = ids[:max_items]
        formulas = formulas[:max_items]
    return G, T, ids, formulas


def plot_tsne_pairs(G, T, out_path, seed=42, perplexity=30):
    """
    Plot: for each sample i, we have two points:
      - graph emb (circle)
      - text emb  (star)
    We color by sample index (cycling colormap). For many points, legend omitted.
    """
    X = np.vstack([G, T])  # [2N, D]

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="random",
        learning_rate="auto"
    )
    X2 = tsne.fit_transform(X)

    N = G.shape[0]
    Xg = X2[:N]
    Xt = X2[N:]

    plt.figure(figsize=(10, 8))
    # graph points
    plt.scatter(Xg[:, 0], Xg[:, 1], s=18, alpha=0.65, marker="o")
    # text points
    plt.scatter(Xt[:, 0], Xt[:, 1], s=60, alpha=0.65, marker="*")

    # connect a few pairs (optional; too dense if N big)
    for i in range(min(N, 200)):
        plt.plot([Xg[i, 0], Xt[i, 0]], [Xg[i, 1], Xt[i, 1]], linewidth=0.5, alpha=0.2)

    plt.title("t-SNE: Graph (o) vs Text (*) embeddings")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_heatmap(G, T, out_path, title="Cosine similarity (Graph @ Text^T)"):
    """
    Since embeddings are L2-normalized, dot product = cosine similarity.
    """
    S = G @ T.T  # [B,B]
    plt.figure(figsize=(8, 6))
    im = plt.imshow(S, aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("Text index in batch")
    plt.ylabel("Graph index in batch")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    diag = np.diag(S)
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    print(f"[Sim] diag mean={np.nanmean(diag):.4f}, diag std={np.nanstd(diag):.4f}")
    print(f"[Sim] off  mean={np.nanmean(off):.4f}, off  std={np.nanstd(off):.4f}")


def main():
    assert os.path.exists(JSONL_PATH), f"Missing jsonl: {JSONL_PATH}"
    assert os.path.exists(CKPT_PATH), f"Missing ckpt: {CKPT_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # dataset + loader
    ds = GraphTextDataset(JSONL_PATH)
    graph_dim = infer_graph_dim(ds)
    print("GRAPH_DIM:", graph_dim)

    # sample subset for t-SNE (optional)
    idxs = list(range(len(ds)))
    random.Random(123).shuffle(idxs)
    idxs = idxs[: min(N_SAMPLES, len(idxs))]
    subset = torch.utils.data.Subset(ds, idxs)

    loader = DataLoader(
        subset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=make_collate_fn(MAX_LEN),
        drop_last=False,
    )

    # build model & load weights
    model = GraphTextCLIP(
        graph_in_dim=graph_dim,
        clip_dim=CLIP_DIM,
        text_width=TEXT_WIDTH,
        max_len=MAX_LEN,
        text_layers=TEXT_LAYERS,
        text_heads=TEXT_HEADS,
        dropout=DROPOUT
    ).to(device)

    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    print("Loaded:", CKPT_PATH)

    # compute embeddings
    G, T, ids, formulas = compute_embeddings(model, loader, device, max_items=N_SAMPLES)
    print("Embeddings:", G.shape, T.shape)

    # 1) t-SNE plot
    tsne_path = os.path.join(OUT_DIR, f"tsne_pairs_N{G.shape[0]}.png")
    plot_tsne_pairs(G, T, tsne_path, seed=TSNE_SEED, perplexity=TSNE_PERPLEXITY)
    print("Saved:", tsne_path)

    # 2) similarity heatmap for a small batch
    B = min(SIM_BATCH, G.shape[0])
    sim_path = os.path.join(OUT_DIR, f"similarity_heatmap_B{B}.png")
    plot_similarity_heatmap(G[:B], T[:B], sim_path)
    print("Saved:", sim_path)

    # 3) simple histogram of diagonal similarities
    diag = np.sum(G * T, axis=1)  # cosine similarity per matched pair
    plt.figure(figsize=(7, 5))
    plt.hist(diag, bins=50, alpha=0.8)
    plt.title("Histogram: cosine similarity of matched (graph,text) pairs")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.tight_layout()
    hist_path = os.path.join(OUT_DIR, f"diag_similarity_hist_N{G.shape[0]}.png")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", hist_path)

    print("Done.")


if __name__ == "__main__":
    main()

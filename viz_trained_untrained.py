# viz_trained_vs_untrained.py
# ------------------------------------------------------------
# Compare:
#   (A) Trained CLIP checkpoint (.pt)
#   (B) Untrained baseline (same architecture, random weights)
#
# Outputs (saved to data_preparation/clip_dataset/viz_baseline):
#   - tsne_trained.png, tsne_untrained.png
#   - sim_heatmap_trained.png, sim_heatmap_untrained.png
#   - diag_hist_trained.png, diag_hist_untrained.png
#   - metrics.json  (top1/top5 + diag stats)
#
# Run:
#   CUDA_VISIBLE_DEVICES=1 python viz_trained_vs_untrained.py
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
CKPT_PATH = os.path.join(DATA_DIR, "graph_text_clip_vanilla.pt")  # trained pt

OUT_DIR = os.path.join(DATA_DIR, "viz_baseline")
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

# Visualization / eval sizes
N_SAMPLES_TSNE = 1200     # samples used for t-SNE (200~2000 recommended)
TSNE_PERPLEXITY = 30
TSNE_SEED = 42

N_SAMPLES_METRICS = 4000  # samples used for retrieval/diag stats (bigger is better)
BATCH_EMB = 256           # batch size for embedding extraction

HEATMAP_B = 64            # similarity heatmap batch size (keep small)


# =========================================================
# Dataset (robust path resolver for your jsonl format)
# =========================================================
class GraphTextDataset(Dataset):
    """
    jsonl example:
    {"id":"6","formula":"Ni4S5Se3","graph_emb":"clip_dataset/graph_emb/6.npy","text":"...",...}
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

                # (1) resolve relative to jsonl dir
                if gpath:
                    cand = gpath if os.path.isabs(gpath) else os.path.join(self.jsonl_dir, gpath)
                    if os.path.exists(cand):
                        resolved = cand

                # (2) fallback: DATA_DIR/graph_emb/{id}.npy
                if resolved is None:
                    cand = os.path.join(GRAPH_EMB_DIR, f"{rid}.npy")
                    if os.path.exists(cand):
                        resolved = cand

                # (3) fallback: DATA_DIR/graph_emb/{formula}_{id}.npy
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
        return {"graph": graph, "text": text, "id": str(it.get("id", idx))}


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
        input_ids, attention_mask = [], []
        for b in batch:
            ids, attn = encode_text_to_bytes(b["text"], max_len)
            input_ids.append(ids)
            attention_mask.append(attn)
        return {
            "graph": graphs,
            "input_ids": torch.stack(input_ids, 0),
            "attention_mask": torch.stack(attention_mask, 0),
            "id": [b["id"] for b in batch],
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
# Embeddings + Metrics
# =========================================================
@torch.no_grad()
def compute_embeddings(model, loader, device, max_items):
    g_list, t_list = [], []
    ids = []

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

        seen += g.shape[0]
        if seen >= max_items:
            break

    G = np.concatenate(g_list, axis=0)[:max_items]
    T = np.concatenate(t_list, axis=0)[:max_items]
    ids = ids[:max_items]
    return G, T, ids


def retrieval_topk(G, T, ks=(1, 5)):
    """
    Compute retrieval accuracy within the sampled set.
    Since embeddings are normalized, cosine similarity = dot product.
    """
    S = G @ T.T  # [N,N]
    N = S.shape[0]
    gt = np.arange(N)

    # g->t
    order_g2t = np.argsort(-S, axis=1)  # descending
    # t->g
    order_t2g = np.argsort(-S, axis=0)  # descending (by column)
    order_t2g = order_t2g  # [N,N], each column sorted

    out = {}
    for k in ks:
        hit_g2t = np.mean([gt[i] in order_g2t[i, :k] for i in range(N)])
        hit_t2g = np.mean([gt[j] in order_t2g[:k, j] for j in range(N)])
        out[f"g2t_top{k}"] = float(hit_g2t)
        out[f"t2g_top{k}"] = float(hit_t2g)

    return out


def diag_stats(G, T):
    diag = np.sum(G * T, axis=1)  # cosine similarity of matched pair
    return {
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "diag_min": float(diag.min()),
        "diag_max": float(diag.max()),
    }


# =========================================================
# Plots
# =========================================================
def plot_tsne_pairs(G, T, out_path, title, seed=42, perplexity=30):
    X = np.vstack([G, T])  # [2N, D]
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
    )
    X2 = tsne.fit_transform(X)
    N = G.shape[0]
    Xg = X2[:N]
    Xt = X2[N:]

    plt.figure(figsize=(10, 8))
    plt.scatter(Xg[:, 0], Xg[:, 1], s=18, alpha=0.65, marker="o", label="graph")
    plt.scatter(Xt[:, 0], Xt[:, 1], s=60, alpha=0.65, marker="*", label="text")

    # connect only first few pairs (avoid clutter)
    for i in range(min(N, 200)):
        plt.plot([Xg[i, 0], Xt[i, 0]], [Xg[i, 1], Xt[i, 1]], linewidth=0.5, alpha=0.15)

    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_heatmap(G, T, out_path, title):
    S = G @ T.T
    plt.figure(figsize=(8, 6))
    im = plt.imshow(S, aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel("text index")
    plt.ylabel("graph index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_diag_hist(G, T, out_path, title):
    diag = np.sum(G * T, axis=1)
    plt.figure(figsize=(7, 5))
    plt.hist(diag, bins=60, alpha=0.85)
    plt.title(title)
    plt.xlabel("cosine similarity (matched graph-text)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# Main
# =========================================================
def build_loader(ds, n_samples, seed=123):
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[: min(n_samples, len(idxs))]
    subset = torch.utils.data.Subset(ds, idxs)
    loader = DataLoader(
        subset,
        batch_size=BATCH_EMB,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=make_collate_fn(MAX_LEN),
        drop_last=False,
    )
    return loader, len(subset)


def main():
    assert os.path.exists(JSONL_PATH), f"Missing jsonl: {JSONL_PATH}"
    assert os.path.exists(CKPT_PATH), f"Missing ckpt: {CKPT_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds = GraphTextDataset(JSONL_PATH)
    graph_dim = infer_graph_dim(ds)
    print("GRAPH_DIM:", graph_dim)

    # ---------- loaders ----------
    loader_tsne, n_tsne = build_loader(ds, N_SAMPLES_TSNE, seed=111)
    loader_metrics, n_met = build_loader(ds, N_SAMPLES_METRICS, seed=222)
    print(f"t-SNE samples: {n_tsne}, metrics samples: {n_met}")

    # ---------- build models ----------
    trained = GraphTextCLIP(
        graph_in_dim=graph_dim,
        clip_dim=CLIP_DIM,
        text_width=TEXT_WIDTH,
        max_len=MAX_LEN,
        text_layers=TEXT_LAYERS,
        text_heads=TEXT_HEADS,
        dropout=DROPOUT,
    ).to(device)

    untrained = GraphTextCLIP(
        graph_in_dim=graph_dim,
        clip_dim=CLIP_DIM,
        text_width=TEXT_WIDTH,
        max_len=MAX_LEN,
        text_layers=TEXT_LAYERS,
        text_heads=TEXT_HEADS,
        dropout=DROPOUT,
    ).to(device)

    # load trained weights
    state = torch.load(CKPT_PATH, map_location="cpu")
    trained.load_state_dict(state, strict=True)
    trained.eval()
    untrained.eval()
    print("Loaded trained checkpoint:", CKPT_PATH)
    print("Untrained baseline: random init (same arch)")

    # ---------- compute embeddings (metrics) ----------
    G_tr, T_tr, _ = compute_embeddings(trained, loader_metrics, device, max_items=n_met)
    G_un, T_un, _ = compute_embeddings(untrained, loader_metrics, device, max_items=n_met)

    # metrics
    metrics = {
        "trained": {},
        "untrained": {},
    }

    metrics["trained"].update(diag_stats(G_tr, T_tr))
    metrics["untrained"].update(diag_stats(G_un, T_un))

    metrics["trained"].update(retrieval_topk(G_tr, T_tr, ks=(1, 5)))
    metrics["untrained"].update(retrieval_topk(G_un, T_un, ks=(1, 5)))

    # save metrics json
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics:", os.path.join(OUT_DIR, "metrics.json"))
    print(json.dumps(metrics, indent=2))

    # ---------- plots: hist + heatmap (metrics subset) ----------
    plot_diag_hist(G_tr, T_tr, os.path.join(OUT_DIR, "diag_hist_trained.png"),
                   "Trained: matched pair cosine similarity histogram")
    plot_diag_hist(G_un, T_un, os.path.join(OUT_DIR, "diag_hist_untrained.png"),
                   "Untrained: matched pair cosine similarity histogram")

    B = min(HEATMAP_B, n_met)
    plot_similarity_heatmap(G_tr[:B], T_tr[:B], os.path.join(OUT_DIR, "sim_heatmap_trained.png"),
                            f"Trained: similarity heatmap (B={B})")
    plot_similarity_heatmap(G_un[:B], T_un[:B], os.path.join(OUT_DIR, "sim_heatmap_untrained.png"),
                            f"Untrained: similarity heatmap (B={B})")

    # ---------- plots: t-SNE (separate extraction for tsne subset) ----------
    G_tr_ts, T_tr_ts, _ = compute_embeddings(trained, loader_tsne, device, max_items=n_tsne)
    G_un_ts, T_un_ts, _ = compute_embeddings(untrained, loader_tsne, device, max_items=n_tsne)

    plot_tsne_pairs(G_tr_ts, T_tr_ts, os.path.join(OUT_DIR, "tsne_trained.png"),
                    title=f"t-SNE (Trained)  N={n_tsne}", seed=TSNE_SEED, perplexity=TSNE_PERPLEXITY)
    plot_tsne_pairs(G_un_ts, T_un_ts, os.path.join(OUT_DIR, "tsne_untrained.png"),
                    title=f"t-SNE (Untrained) N={n_tsne}", seed=TSNE_SEED, perplexity=TSNE_PERPLEXITY)

    print("Saved plots to:", OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

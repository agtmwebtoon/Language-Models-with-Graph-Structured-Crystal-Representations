import os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Paths (YOUR SETUP)
# =========================================================
DATA_DIR = "data_preparation/clip_dataset"
ALL_JSONL = os.path.join(DATA_DIR, "dataset.jsonl")
GRAPH_EMB_DIR = os.path.join(DATA_DIR, "graph_emb")


# =========================================================
# Dataset
# =========================================================
class GraphTextDataset(Dataset):
    """
    jsonl example:
    {
      "id": "6",
      "formula": "Ni4S5Se3",
      "graph_emb": "clip_dataset/graph_emb/6.npy",
      "text": "...",
      "tensor": "[...]"
    }

    We will robustly resolve graph_emb path:
      1) if graph_emb is relative: first resolve relative to jsonl directory
      2) fallback: DATA_DIR/graph_emb/{id}.npy
      3) fallback: DATA_DIR/graph_emb/{formula}_{id}.npy
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

                # ---- resolve graph_emb path ----
                gpath = it.get("graph_emb", None)
                resolved = None

                # (1) resolve relative to jsonl dir
                if gpath:
                    if os.path.isabs(gpath):
                        cand = gpath
                    else:
                        cand = os.path.join(self.jsonl_dir, gpath)
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
                    # skip this sample
                    continue

                it["_graph_path"] = resolved
                self.items.append(it)

        if len(self.items) == 0:
            raise RuntimeError(
                f"No valid items loaded from {jsonl_path}\n"
                f"Check:\n"
                f"- GRAPH_EMB_DIR={GRAPH_EMB_DIR}\n"
                f"- jsonl graph_emb path like 'clip_dataset/graph_emb/6.npy' is not resolvable.\n"
            )

        print(f"[Dataset] loaded {len(self.items)} items (from {jsonl_path})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        graph = np.load(it["_graph_path"]).astype(np.float32).reshape(-1)  # [G]
        text = it["text"]
        return {"graph": graph, "text": text, "id": it["id"]}


def infer_graph_dim(jsonl_path: str) -> int:
    ds = GraphTextDataset(jsonl_path)
    return int(ds[0]["graph"].shape[0])


# =========================================================
# Byte tokenizer
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
        }
    return collate


# =========================================================
# Vanilla Transformer (same as before)
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
        scores = (Q @ K.transpose(-2, -1)) * self.scale
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
    def __init__(self, graph_in_dim: int, clip_dim: int, text_width: int, max_len: int, text_layers: int, text_heads: int, dropout: float):
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

    def forward(self, graph, input_ids, attention_mask):
        g = self.graph_encoder(graph)
        t = self.text_encoder(input_ids, attention_mask)
        logits = self.logit_scale.exp() * (g @ t.t())
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.t(), labels)) / 2.0
        return loss, logits


@torch.no_grad()
def retrieval_acc(logits):
    gt = torch.arange(logits.size(0), device=logits.device)
    return (logits.argmax(1) == gt).float().mean().item(), (logits.argmax(0) == gt).float().mean().item()


# =========================================================
# Train
# =========================================================
def main():
    MAX_LEN = 256
    CLIP_DIM = 512

    TEXT_WIDTH  = 512
    TEXT_LAYERS = 6
    TEXT_HEADS  = 8
    DROPOUT = 0.0

    BATCH_SIZE = 64
    LR = 2e-4
    WEIGHT_DECAY = 0.01
    EPOCHS = 10
    NUM_WORKERS = 4

    SAVE_PATH = os.path.join(DATA_DIR, "graph_text_clip_vanilla.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("JSONL:", ALL_JSONL)

    GRAPH_DIM = infer_graph_dim(ALL_JSONL)
    print("Inferred GRAPH_DIM:", GRAPH_DIM)

    ds = GraphTextDataset(ALL_JSONL)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=make_collate_fn(MAX_LEN),
        drop_last=True,
    )

    model = GraphTextCLIP(
        graph_in_dim=GRAPH_DIM,
        clip_dim=CLIP_DIM,
        text_width=TEXT_WIDTH,
        max_len=MAX_LEN,
        text_layers=TEXT_LAYERS,
        text_heads=TEXT_HEADS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0

        for step, batch in enumerate(loader, 1):
            graph = batch["graph"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)

            loss, logits = model(graph, input_ids, attn)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += loss.item()

            if step % 50 == 0:
                a1, a2 = retrieval_acc(logits.detach())
                print(f"[Epoch {epoch} | Step {step}] loss={loss.item():.4f} acc(g→t)={a1:.3f} acc(t→g)={a2:.3f}")

        print(f"Epoch {epoch} avg loss = {total / max(1, len(loader)):.4f}")
        torch.save(model.state_dict(), SAVE_PATH)
        print("✓ Saved:", SAVE_PATH)

    print("Done.")


if __name__ == "__main__":
    main()

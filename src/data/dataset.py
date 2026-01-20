"""Dataset and DataLoader utilities for graph-text pairs."""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional

from .tokenizer import ByteLevelTokenizer


class GraphTextDataset(Dataset):
    """Dataset for graph-text pairs with robust path resolution."""

    def __init__(self, jsonl_path: str, graph_emb_dir: Optional[str] = None):
        """
        Args:
            jsonl_path: path to JSONL file containing dataset
            graph_emb_dir: directory containing graph embeddings (optional fallback)
        """
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_dir = self.jsonl_path.parent
        self.graph_emb_dir = Path(graph_emb_dir) if graph_emb_dir else None
        self.items: List[Dict] = []

        self._load_data()

        if len(self.items) == 0:
            raise RuntimeError(f"No valid items loaded from {jsonl_path}")

        print(f"[Dataset] Loaded {len(self.items)} items from {jsonl_path}")

    def _load_data(self):
        """Load and validate data from JSONL file."""
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "text" not in item or "id" not in item:
                    continue

                rid = str(item["id"]).strip()
                formula = str(item.get("formula", "")).strip()

                # Resolve graph embedding path
                graph_path = self._resolve_graph_path(item, rid, formula)
                if graph_path is None:
                    continue

                item["_graph_path"] = graph_path
                self.items.append(item)

    def _resolve_graph_path(self, item: Dict, rid: str, formula: str) -> Optional[str]:
        """
        Resolve graph embedding path with multiple fallback strategies.

        Priority:
        1. Relative path from JSONL (if exists)
        2. {graph_emb_dir}/{id}.npy
        3. {graph_emb_dir}/{formula}_{id}.npy
        """
        gpath = item.get("graph_emb", None)
        resolved = None

        # Strategy 1: resolve relative to JSONL directory
        if gpath:
            cand = Path(gpath) if Path(gpath).is_absolute() else self.jsonl_dir / gpath
            if cand.exists():
                resolved = str(cand)

        # Strategy 2: {graph_emb_dir}/{id}.npy
        if resolved is None and self.graph_emb_dir:
            cand = self.graph_emb_dir / f"{rid}.npy"
            if cand.exists():
                resolved = str(cand)

        # Strategy 3: {graph_emb_dir}/{formula}_{id}.npy
        if resolved is None and self.graph_emb_dir and formula:
            cand = self.graph_emb_dir / f"{formula}_{rid}.npy"
            if cand.exists():
                resolved = str(cand)

        return resolved

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        graph = np.load(item["_graph_path"]).astype(np.float32).reshape(-1)
        text = item["text"]

        return {
            "graph": graph,
            "text": text,
            "id": str(item.get("id", idx)),
            "formula": str(item.get("formula", "")),
        }

    @property
    def graph_dim(self) -> int:
        """Infer graph embedding dimension from first sample."""
        return int(self[0]["graph"].shape[0])


class GraphTextCollator:
    """Collate function for batching graph-text pairs."""

    def __init__(self, tokenizer: ByteLevelTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        # Stack graph embeddings
        graphs = torch.tensor(
            np.stack([b["graph"] for b in batch]), dtype=torch.float32
        )
        texts = [b["text"] for b in batch]

        if hasattr(self.tokenizer, "encode_batch"):
            input_ids, attention_mask = self.tokenizer.encode_batch(texts)

        else:
            ids_list, attn_list = [], []
            for t in texts:
                ids, attn = self.tokenizer.encode(t, self.max_len)
                ids_list.append(ids)
                attn_list.append(attn)
            input_ids = torch.stack(ids_list, 0)
            attention_mask = torch.stack(attn_list, 0)

        return {
            "graph": graphs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": [b["id"] for b in batch],
            "formula": [b["formula"] for b in batch],
        }


def create_dataloader(
    dataset: GraphTextDataset,
    tokenizer: ByteLevelTokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = False,
) -> DataLoader:
    """Create DataLoader with proper collation."""
    collator = GraphTextCollator(tokenizer, max_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=drop_last,
    )

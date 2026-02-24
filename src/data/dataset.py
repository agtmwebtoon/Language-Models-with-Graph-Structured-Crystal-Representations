"""Dataset and DataLoader utilities for graph-text pairs and tensor regression."""

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

    def __init__(
        self,
        jsonl_path: str,
        graph_emb_dir: Optional[str] = None,
        normalize_tensor: bool = True,
        tensor_mode: str = "full",
    ):
        """
        Args:
            jsonl_path: path to JSONL file containing dataset
            graph_emb_dir: directory containing graph embeddings (optional fallback)
            normalize_tensor: whether to normalize tensor values (recommended for regression)
            tensor_mode: "2d" for 2D materials (c11,c12,c22,c33) or "full" for full 3x3 tensor
        """
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_dir = self.jsonl_path.parent
        self.graph_emb_dir = Path(graph_emb_dir) if graph_emb_dir else None
        self.items: List[Dict] = []
        self.normalize_tensor = normalize_tensor
        self.tensor_mode = tensor_mode
        self.tensor_mean: Optional[np.ndarray] = None
        self.tensor_std: Optional[np.ndarray] = None

        # Define component indices for different tensor modes
        # Original order: [c11, c12, c13, c21, c22, c23, c31, c32, c33]
        if tensor_mode == "2d":
            self.tensor_indices = [0, 1, 4, 8]  # [c11, c12, c22, c33]
        elif tensor_mode == "voigt2d":
            # 2D elastic Voigt: [C11, C22, C12, C66, C16, C26]
            # Map: C11=c11, C22=c22, C12=c12, C66=c33, C16=c13, C26=c23
            self.tensor_indices = [0, 4, 1, 8, 2, 5]
        else:
            self.tensor_indices = list(range(9))  # All components

        self._load_data()

        if len(self.items) == 0:
            raise RuntimeError(f"No valid items loaded from {jsonl_path}")

        # Compute tensor statistics for normalization
        if self.normalize_tensor:
            self._compute_tensor_stats()

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

    def _compute_tensor_stats(self):
        """Compute mean and std for tensor normalization (for selected components only)."""
        tensors = []
        for item in self.items:
            if "tensor" in item:
                tensor_str = item["tensor"]
                if isinstance(tensor_str, str):
                    tensor_list = json.loads(tensor_str)
                else:
                    tensor_list = tensor_str

                # Select only the components we need based on tensor_mode
                tensor_array = np.array(tensor_list, dtype=np.float32)
                selected_tensor = tensor_array[self.tensor_indices]
                tensors.append(selected_tensor)

        if len(tensors) > 0:
            tensors_array = np.array(tensors, dtype=np.float32)
            self.tensor_mean = tensors_array.mean(axis=0)
            self.tensor_std = tensors_array.std(axis=0) + 1e-8  # Avoid division by zero

            if self.tensor_mode == "2d":
                mode_str = "2D (c11,c12,c22,c33)"
            elif self.tensor_mode == "voigt2d":
                mode_str = "Voigt 2D (C11,C22,C12,C66,C16,C26)"
            else:
                mode_str = "Full 3x3"
            print(f"[Dataset] Tensor mode: {mode_str}")
            print(f"[Dataset] Tensor normalization: mean={self.tensor_mean.mean():.2f}, std={self.tensor_std.mean():.2f}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        graph = np.load(item["_graph_path"]).astype(np.float32).reshape(-1)
        text = item["text"]

        result = {
            "graph": graph,
            "text": text,
            "id": str(item.get("id", idx)),
            "formula": str(item.get("formula", "")),
            "group": item.get("robo_crystal_system", item.get("group", "triclinic")),  # Crystal system for symmetry
        }

        # Add tensor if available
        if "tensor" in item:
            tensor_str = item["tensor"]
            if isinstance(tensor_str, str):
                tensor_list = json.loads(tensor_str)
            else:
                tensor_list = tensor_str

            tensor_array = np.array(tensor_list, dtype=np.float32)

            # Select only the components we need based on tensor_mode
            tensor_array = tensor_array[self.tensor_indices]

            # Normalize if enabled
            if self.normalize_tensor and self.tensor_mean is not None:
                tensor_array = (tensor_array - self.tensor_mean) / self.tensor_std

            result["tensor"] = tensor_array

        return result

    def denormalize_tensor(self, normalized_tensor: np.ndarray) -> np.ndarray:
        """Convert normalized tensor back to original scale."""
        if self.normalize_tensor and self.tensor_mean is not None:
            return normalized_tensor * self.tensor_std + self.tensor_mean
        return normalized_tensor

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

        result = {
            "graph": graphs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": [b["id"] for b in batch],
            "formula": [b["formula"] for b in batch],
            "group": [b["group"] for b in batch],  # Crystal systems for symmetry-aware models
        }

        # Add tensors if available (already normalized in __getitem__)
        if "tensor" in batch[0]:
            tensors = [b["tensor"] for b in batch]
            result["tensor"] = torch.tensor(np.stack(tensors), dtype=torch.float32)

        return result


def create_dataloader(
    dataset: GraphTextDataset,
    tokenizer: ByteLevelTokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
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

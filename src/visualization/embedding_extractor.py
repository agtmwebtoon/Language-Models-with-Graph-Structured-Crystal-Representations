# src/viz/embedding_extractor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class EmbeddingBatch:
    G: np.ndarray                 # [N, D]
    T: np.ndarray                 # [N, D]
    ids: List[str]
    formulas: List[str]


class EmbeddingExtractor:
    """
    Extracts aligned graph/text embeddings from a GraphTextCLIP model and a DataLoader.
    Assumes model.encode_graph / model.encode_text exist and return L2-normalized embeddings.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def extract(
        self,
        loader: DataLoader,
        max_items: Optional[int] = None,
    ) -> EmbeddingBatch:
        self.model.eval()

        g_list, t_list = [], []
        ids, formulas = [], []
        seen = 0

        for batch in loader:
            graph = batch["graph"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn = batch["attention_mask"].to(self.device, non_blocking=True)

            g = self.model.encode_graph(graph).detach().cpu().numpy()
            t = self.model.encode_text(input_ids, attn).detach().cpu().numpy()

            g_list.append(g)
            t_list.append(t)
            ids.extend(batch.get("id", []))
            formulas.extend(batch.get("formula", []))

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

        return EmbeddingBatch(G=G, T=T, ids=ids, formulas=formulas)

# src/viz/visualizer.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os
import random

import torch
from torch.utils.data import Subset

from src.utils.config import Config
from src.data.dataset import GraphTextDataset, create_dataloader
from src.data.tokenizer import ByteLevelTokenizer, HFTokenizerWrapper
from src.models.clip_model import GraphTextCLIP

from .embedding_extractor import EmbeddingExtractor, EmbeddingBatch
from .plots import TSNEConfig, plot_tsne_pairs, plot_similarity_heatmap, plot_diag_hist
from ..utils.run_naming import build_run_name, select_latest_run_dir


@dataclass
class VizRunConfig:
    n_samples: int = 800
    sim_batch: int = 64
    tsne: TSNEConfig = TSNEConfig()
    seed_subset: int = 123
    loader_batch_size: int = 128


class GraphTextCLIPVisualizer:
    """
    End-to-end visualizer for a trained GraphTextCLIP model.
    - Loads config
    - Builds dataset/loader matching backend (custom/t5)
    - Loads checkpoint
    - Extract embeddings and saves plots
    """

    def __init__(self, config: Config, run_cfg: Optional[VizRunConfig] = None):
        self.config = config
        self.run_cfg = run_cfg or VizRunConfig()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config.training.device = str(self.device)

    def _use_latest_run_paths(self):
        runs_dir = Path(self.config.paths.data_dir) / "runs"
        base_name = build_run_name(self.config)

        latest = select_latest_run_dir(runs_dir, base_name)
        if latest is None:
            raise FileNotFoundError(f"No run found for base name: {base_name} under {runs_dir}")

        self.config.paths.run_dir = latest
        self.config.paths.run_name = latest.name
        self.config.paths.checkpoint_dir = latest / "checkpoints"
        self.config.paths.viz_dir = latest / "viz"

        ckpt_name = Path(
            self.config.paths.checkpoint_path).name if self.config.paths.checkpoint_path else "graph_text_clip.pt"
        self.config.paths.checkpoint_path = self.config.paths.checkpoint_dir / ckpt_name

        # viz output도 run/viz로 고정
        self.config.visualization.output_dir = self.config.paths.viz_dir

        os.makedirs(self.config.paths.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.paths.viz_dir, exist_ok=True)

        print(f"[viz] Using latest run: {latest.name}")

    def _build_tokenizer(self):
        if self.config.model.text_backend == "huggingface":
            return HFTokenizerWrapper(
                model_name=self.config.model.text_model_name,
                max_len=self.config.model.max_seq_length,
                padding=self.config.tokenizer.tokenizer_padding,
            )
        return ByteLevelTokenizer(
            pad_token=self.config.tokenizer.pad_token,
            sot_token=self.config.tokenizer.sot_token,
            eot_token=self.config.tokenizer.eot_token,
        )

    def _build_dataset_and_loader(self) -> Tuple[GraphTextDataset, torch.utils.data.DataLoader]:
        dataset = GraphTextDataset(
            jsonl_path=str(self.config.paths.jsonl_path),
            graph_emb_dir=str(self.config.paths.graph_emb_dir),
        )

        # subset selection
        idxs = list(range(len(dataset)))
        random.Random(self.run_cfg.seed_subset).shuffle(idxs)
        idxs = idxs[: min(self.run_cfg.n_samples, len(idxs))]
        subset = Subset(dataset, idxs)

        tokenizer = self._build_tokenizer()

        loader = create_dataloader(
            dataset=subset,
            tokenizer=tokenizer,
            max_len=self.config.model.max_seq_length,
            batch_size=min(self.run_cfg.loader_batch_size, self.config.training.batch_size),
            shuffle=False,
            num_workers=self.config.training.num_workers,
            drop_last=False,
        )
        return dataset, loader

    def _build_model(self, graph_dim: int) -> GraphTextCLIP:
        if self.config.model.text_backend == "huggingface":
            model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=self.config.model.clip_dim,
                text_backend="huggingface",
                text_model_name=self.config.model.text_model_name,
                text_pooling=self.config.model.text_pooling,
                freeze_text_backbone=True,  # inference only
                train_text_layernorm_only=False,
                text_dropout=0.0,
            )
        else:
            model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=self.config.model.clip_dim,
                text_backend="custom",
                text_width=self.config.model.text_width,
                max_len=self.config.model.max_seq_length,
                text_layers=self.config.model.text_layers,
                text_heads=self.config.model.text_heads,
                vocab_size=self.config.model.vocab_size,
                dropout=self.config.model.dropout,
            )
        return model.to(self.device)

    def _load_checkpoint(self, model: torch.nn.Module):
        ckpt = self.config.paths.checkpoint_path
        if ckpt is None:
            raise ValueError("config.paths.checkpoint_path is None")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        return ckpt

    def run(self) -> dict:
        self._use_latest_run_paths()
        dataset, loader = self._build_dataset_and_loader()
        graph_dim = dataset.graph_dim

        model = self._build_model(graph_dim)
        ckpt_path = self._load_checkpoint(model)

        extractor = EmbeddingExtractor(model=model, device=self.device)
        batch: EmbeddingBatch = extractor.extract(loader, max_items=self.run_cfg.n_samples)

        out_dir = self.config.visualization.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # 1) t-SNE
        tsne_path = os.path.join(out_dir, f"tsne_pairs_N{batch.G.shape[0]}_{self.text_backend}.png")
        plot_tsne_pairs(batch.G, batch.T, tsne_path, self.run_cfg.tsne)

        # 2) similarity heatmap
        B = min(self.run_cfg.sim_batch, batch.G.shape[0])
        sim_path = os.path.join(out_dir, f"similarity_heatmap_B{B}_{self.text_backend}.png")
        sim_stats = plot_similarity_heatmap(batch.G[:B], batch.T[:B], sim_path)

        # 3) diag hist
        hist_path = os.path.join(out_dir, f"diag_similarity_hist_N{batch.G.shape[0]}_{self.text_backend}.png")
        plot_diag_hist(batch.G, batch.T, hist_path)

        return {
            "checkpoint": str(ckpt_path),
            "backend": self.text_backend,
            "n_samples": int(batch.G.shape[0]),
            "tsne_path": tsne_path,
            "sim_path": sim_path,
            "hist_path": hist_path,
            "sim_stats": sim_stats,
        }

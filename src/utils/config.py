"""Configuration management for Graph-Text CLIP model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class PathConfig:
    """File and directory paths."""
    data_dir: Path
    jsonl_path: Path
    graph_emb_dir: Path
    checkpoint_path: Path = None
    output_dir: Path = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.jsonl_path = Path(self.jsonl_path)
        self.graph_emb_dir = Path(self.graph_emb_dir)
        if self.checkpoint_path:
            self.checkpoint_path = Path(self.checkpoint_path)
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    clip_dim: int = 512
    text_backend: Literal["t5", "custom"] = "t5"
    text_width: int = 512
    text_layers: int = 6
    text_heads: int = 8
    max_seq_length: int = 256
    vocab_size: int = 256
    dropout: float = 0.0

    t5_model_name: str = "t5-base"
    t5_pooling: Literal["mean", "first", "last"] = "mean"
    t5_padding: Literal["max_length", "longest"] = "max_length"
    t5_dropout: float = 0.0
    freeze_t5: bool = False
    train_layernorm_only: bool = False


    def __post_init__(self):
        assert self.text_width % self.text_heads == 0, "text_width must be divisible by text_heads"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 10
    num_workers: int = 4
    device: str = "cuda"
    log_interval: int = 50


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    pad_token: int = 0
    sot_token: int = 2
    eot_token: int = 3
    vocab_size: int = 256


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    n_samples_tsne: int = 800
    n_samples_metrics: int = 4000
    tsne_perplexity: int = 30
    tsne_seed: int = 42
    similarity_batch_size: int = 64
    output_dir: Path = None

    def __post_init__(self):
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration container."""

    def __init__(
        self,
        data_dir: str,
        jsonl_filename: str = "dataset.jsonl",
        graph_emb_dirname: str = "graph_emb",
        checkpoint_filename: str = None,
        output_dirname: str = None,
    ):
        data_dir = Path(data_dir)

        self.paths = PathConfig(
            data_dir=data_dir,
            jsonl_path=data_dir / jsonl_filename,
            graph_emb_dir=data_dir / graph_emb_dirname,
            checkpoint_path=data_dir / checkpoint_filename if checkpoint_filename else None,
            output_dir=data_dir / output_dirname if output_dirname else None,
        )

        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.tokenizer = TokenizerConfig()
        self.visualization = VisualizationConfig(
            output_dir=data_dir / "viz" if output_dirname is None else Path(output_dirname)
        )

    @classmethod
    def from_defaults(cls, data_dir: str = "data_preparation/clip_dataset"):
        """Create config with default values."""
        return cls(
            data_dir=data_dir,
            checkpoint_filename="graph_text_clip_vanilla.pt",
            output_dirname="viz"
        )

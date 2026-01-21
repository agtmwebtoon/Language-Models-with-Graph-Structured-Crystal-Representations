"""Configuration management for Graph-Text CLIP model."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Any, Dict, Mapping, Union
import traceback

from .run_naming import build_run_name, resolve_run_dir_with_suffix
from .config_dump import dump_config_txt, dump_config_json

CfgLike = Union["Config", Mapping[str, Any], str, None]


@dataclass
class PathConfig:
    """File and directory paths."""
    data_dir: Path
    jsonl_path: Path
    graph_emb_dir: Path

    runs_dir: Path
    run_name: str
    run_dir: Path
    checkpoint_dir: Path
    viz_dir: Path
    checkpoint_path: Path

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.jsonl_path = Path(self.jsonl_path)
        self.graph_emb_dir = Path(self.graph_emb_dir)
        self.runs_dir = Path(self.runs_dir)
        self.run_dir = Path(self.run_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.viz_dir = Path(self.viz_dir)
        self.checkpoint_path = Path(self.checkpoint_path)

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    clip_dim: int = 512

    # ----- TEXT ENCODER -----
    text_backend: Literal["custom", "huggingface"] = "huggingface"

    # For huggingface backend (T5, BERT, RoBERTa, etc.)
    text_model_name: str = "t5-base"  # HuggingFace model name
    text_pooling: Literal["mean", "first", "last"] = "mean"
    text_dropout: float = 0.0
    freeze_text_backbone: bool = True
    train_text_layernorm_only: bool = True

    # For custom transformer backend
    text_width: int = 512
    text_layers: int = 6
    text_heads: int = 8
    vocab_size: int = 256

    # Shared parameters
    max_seq_length: int = 256
    dropout: float = 0.0

    # ----- GRAPH (graph embedding model + projector) -----
    # graph_backbone: graph embedding을 생성한 모델/파이프라인 이름
    graph_backbone: Literal["equiformer", "chgnet", "cgcnn", "mpnn", "other"] = "equiformer"
    graph_backbone_name: Optional[str] = None  # ex) "equiformer_v2_oc20_83M_2M"
    graph_backbone_id: Optional[str] = None  # 짧은 run naming용 (없으면 자동 생성)

    # graph_projector: CLIP 공간으로 매핑하는 head
    graph_projector: Literal["linear", "mlp"] = "linear"
    graph_projector_id: Optional[str] = None  # 보통 linear/mlp

    # ----- IDs for run naming -----
    text_model_id: Optional[str] = None  # 짧은 run naming용 (없으면 자동 생성)

    def __post_init__(self):
        # ---- text_model_id 자동 생성 ----
        if self.text_model_id is None:
            if self.text_backend == "huggingface":
                self.text_model_id = self.text_model_name.replace("/", "-")
            else:
                self.text_model_id = "customTx"

        # ---- graph_backbone_id 자동 생성 ----
        if self.graph_backbone_id is None:
            base = self.graph_backbone_name or self.graph_backbone
            self.graph_backbone_id = str(base).replace("/", "-")

        # ---- graph_projector_id 자동 생성 ----
        if self.graph_projector_id is None:
            self.graph_projector_id = self.graph_projector

        # custom text일 때만 체크
        if self.text_backend == "custom":
            assert self.text_width % self.text_heads == 0, \
                f"text_width ({self.text_width}) must be divisible by text_heads ({self.text_heads})"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 2
    num_workers: int = 4
    device: str = "cuda"
    log_interval: int = 50

    # Train/Val split
    use_validation: bool = True
    val_split: float = 0.1  # 10% for validation
    val_seed: int = 42

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_metric: Literal["val_loss", "val_acc_g2t", "val_acc_t2g"] = "val_loss"
    early_stopping_mode: Literal["min", "max"] = "min"  # min for loss, max for accuracy

    # WandB logging
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[list] = None

    # Visualization during training
    visualize_during_training: bool = False
    visualize_interval: int = 10  # visualize every N epochs
    visualize_n_samples: int = 800  # number of samples to visualize


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    # For byte-level tokenizer
    pad_token: int = 0
    sot_token: int = 2
    eot_token: int = 3
    vocab_size: int = 256

    # For HuggingFace tokenizers
    tokenizer_padding: Literal["max_length", "longest"] = "max_length"


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


class Config:
    """Main configuration container."""

    def __init__(
        self,
        data_dir: str,
        jsonl_filename: str = "dataset.jsonl",
        graph_emb_dirname: str = "graph_emb",
        run_name: Optional[str] = None,
        checkpoint_filename: str = "graph_text_clip.pt",
    ):
        data_dir = Path(data_dir)

        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.tokenizer = TokenizerConfig()

        if run_name is None:
            run_name = build_run_name(self)

        runs_dir = data_dir / "runs"
        run_dir = runs_dir / run_name
        ckpt_dir = run_dir / "checkpoints"
        viz_dir = run_dir / "viz"


        self.paths = PathConfig(
            data_dir=data_dir,
            jsonl_path=data_dir / jsonl_filename,
            graph_emb_dir=data_dir / graph_emb_dirname,
            runs_dir=runs_dir,
            run_name=run_name,
            run_dir=run_dir,
            checkpoint_dir=ckpt_dir,
            viz_dir=viz_dir,
            checkpoint_path=ckpt_dir / checkpoint_filename,

        )
        self.visualization = VisualizationConfig(output_dir=self.paths.viz_dir)

    def apply_overrides(self, payload: Mapping[str, Any]) -> "Config":
        """
        Apply cfg overrides to this Config in-place.
        Only updates existing dataclass fields.
        """
        payload = dict(payload)

        # Update model config
        if "model" in payload and payload["model"] is not None:
            for k, v in payload["model"].items():
                if hasattr(self.model, k):
                    setattr(self.model, k, v)

        # Update training config
        if "training" in payload and payload["training"] is not None:
            for k, v in payload["training"].items():
                if hasattr(self.training, k):
                    setattr(self.training, k, v)

        # Update tokenizer config
        if "tokenizer" in payload and payload["tokenizer"] is not None:
            for k, v in payload["tokenizer"].items():
                if hasattr(self.tokenizer, k):
                    setattr(self.tokenizer, k, v)

        # Update visualization config
        if "visualization" in payload and payload["visualization"] is not None:
            for k, v in payload["visualization"].items():
                if hasattr(self.visualization, k):
                    setattr(self.visualization, k, v)

        # Update data_dir
        if "data_dir" in payload and payload["data_dir"] is not None:
            self.paths.data_dir = Path(payload["data_dir"])

        return self

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], default_data_dir: str = "data_preparation/clip_dataset") -> "Config":
        payload = dict(payload)

        data_dir = payload.get("data_dir") or (payload.get("paths", {}) or {}).get("data_dir") or default_data_dir
        cfg = cls(data_dir=str(data_dir))

        cfg.apply_overrides(payload)

        # Re-run post_init to ensure derived fields are populated
        cfg.model.__post_init__()

        return cfg

    @classmethod
    def from_json(cls, path: Union[str, Path], default_data_dir: str = "data_preparation/clip_dataset") -> "Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload, default_data_dir=default_data_dir)

    @classmethod
    def from_defaults(cls, data_dir: str = "data_preparation/clip_dataset"):
        """Create config with default values."""
        return cls(
            data_dir=data_dir,
            run_name=None
        )

    @classmethod
    def build(cls, cfg_like: CfgLike = None, default_data_dir: str = "data_preparation/clip_dataset") -> "Config":
        """
        Unified entry:
          - None -> from_defaults
          - Config -> returned as-is
          - str -> JSON path
          - dict-like -> from_dict
        """
        if cfg_like is None:
            return cls.from_defaults(data_dir=default_data_dir)
        if isinstance(cfg_like, cls):
            return cfg_like
        if isinstance(cfg_like, (str, Path)):
            return cls.from_json(cfg_like, default_data_dir=default_data_dir)
        return cls.from_dict(cfg_like, default_data_dir=default_data_dir)

    def finalize(self):
        """
        Finalize run directory:
        - build short run_name: {text_model_id}_{graph_backbone_id}
        - resolve name collision with _001, _002, ...
        - create checkpoints/ and viz/ dirs
        - dump full config snapshot (txt + json)
        Call exactly once before training or visualization.
        """

        print("[DEBUG] Config.finalize() called")
        traceback.print_stack(limit=5)

        data_dir = Path(self.paths.data_dir)
        runs_dir = data_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        # 1) build base run name (short)
        base_name = build_run_name(self)  # e.g. t5-base_equiformer

        # 2) resolve collision with numeric suffix
        run_dir = resolve_run_dir_with_suffix(runs_dir, base_name)

        # 3) subdirectories
        ckpt_dir = run_dir / "checkpoints"
        viz_dir = run_dir / "viz"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)

        # 4) update paths
        self.paths.runs_dir = runs_dir
        self.paths.run_name = run_dir.name
        self.paths.run_dir = run_dir
        self.paths.checkpoint_dir = ckpt_dir
        self.paths.viz_dir = viz_dir

        # keep checkpoint filename, move only directory
        ckpt_name = (
            Path(self.paths.checkpoint_path).name
            if self.paths.checkpoint_path
            else "graph_text_clip.pt"
        )
        self.paths.checkpoint_path = ckpt_dir / ckpt_name

        # viz output dir
        self.visualization.output_dir = viz_dir

        # 5) dump config snapshot
        dump_config_txt(self, run_dir / "config.txt")
        dump_config_json(self, run_dir / "config.json")

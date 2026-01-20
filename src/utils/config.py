"""Configuration management for Graph-Text CLIP model."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Any, Dict, Mapping, Union
import traceback

from .run_naming import build_run_name, resolve_run_dir_with_suffix
from .config_dump import dump_config_txt, dump_config_json

CfgLike = Union["Config", Mapping[str, Any], str, None]

def _safe_set(obj, k: str, v: Any):
    """Set attribute only if it exists."""
    if hasattr(obj, k):
        setattr(obj, k, v)


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

    # ----- TEXT -----
    text_backend: Literal["custom", "t5", "bert", "llama", "other"] = "t5"
    text_model_name: Optional[str] = None  # ex) "t5-base", "bert-base-uncased"
    text_model_id: Optional[str] = None  # 짧은 run naming용 (없으면 자동 생성)

    # ----- GRAPH (graph embedding model+ projector) -----
    # graph_backbone: graph embedding을 생성한 모델/파이프라인 이름
    graph_backbone: Literal["equiformer", "chgnet", "cgcnn", "mpnn", "other"] = "equiformer"
    graph_backbone_name: Optional[str] = None  # ex) "equiformer_v2_oc20_83M_2M"
    graph_backbone_id: Optional[str] = None  # 짧은 run naming용 (없으면 자동 생성)

    # graph_projector: CLIP 공간으로 매핑하는 head
    graph_projector: Literal["linear", "mlp"] = "linear"
    graph_projector_id: Optional[str] = None  # 보통 linear/mlp

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
    freeze_t5: bool = True
    train_layernorm_only: bool = True


    def __post_init__(self):
        # ---- text_model_id 자동 생성 ----
        if self.text_model_id is None:
            if self.text_backend == "t5":
                self.text_model_id = self.t5_model_name.replace("/", "-")
                self.text_model_name = self.t5_model_name
            elif self.text_model_name:
                self.text_model_id = self.text_model_name.replace("/", "-")
            else:
                self.text_model_id = "customTx"

        # ---- graph_backbone_id 자동 생성 ----
        if self.graph_backbone_id is None:
            # equiformer_v2_oc20_83M 처럼 짧게
            base = self.graph_backbone_name or self.graph_backbone
            self.graph_backbone_id = str(base).replace("/", "-")

        # ---- graph_projector_id 자동 생성 ----
        if self.graph_projector_id is None:
            self.graph_projector_id = self.graph_projector

        # custom text일 때만 체크
        if self.text_backend == "custom":
            assert self.text_width % self.text_heads == 0


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
        Unknown keys are ignored (safe).
        """
        payload = dict(payload)

        for sec in ["model", "training", "tokenizer", "visualization"]:
            if sec in payload and payload[sec] is not None:
                section_dict = dict(payload[sec])
                obj = getattr(self, sec)
                for k, v in section_dict.items():
                    _safe_set(obj, k, v)

        viz = payload.get("visualization", {})
        tsne = viz.get("tsne")
        if tsne is not None and hasattr(self.visualization, "tsne"):
            for k, v in dict(tsne).items():
                _safe_set(self.visualization.tsne, k, v)

        # model id 자동 채움
        if hasattr(self.model, "__post_init__"):
            self.model.__post_init__()

        if "data_dir" in payload and payload["data_dir"] is not None:
            self.paths.data_dir = Path(payload["data_dir"])

        return self

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], default_data_dir: str = "data_preparation/clip_dataset") -> "Config":
        payload = dict(payload)

        data_dir = payload.get("data_dir") or (payload.get("paths", {}) or {}).get("data_dir") or default_data_dir
        cfg = cls(data_dir=str(data_dir))

        cfg.apply_overrides(payload)


        return cfg

    @classmethod
    def from_json(cls, path: Union[str, Path], default_data_dir: str = "data_preparation/clip_dataset") -> "Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        data_dir = payload.get("data_dir", default_data_dir)

        cfg = cls.from_defaults(data_dir=str(data_dir))
        cfg.apply_overrides(payload)
        return cfg

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

        payload = dict(cfg_like)
        data_dir = payload.get("data_dir", default_data_dir)
        cfg = cls.from_defaults(data_dir=str(data_dir))
        cfg.apply_overrides(payload)
        return cfg

    @classmethod
    def from_defaults(cls, data_dir: str = "data_preparation/clip_dataset"):
        """Create config with default values."""
        return cls(
            data_dir=data_dir,
            run_name=None
        )

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

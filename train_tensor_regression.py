"""
Train tensor regression model using pre-trained CLIP encoders.

Phase 1 (Sanity Check):
    python train_tensor_regression.py --cfg expr_setting/tensor_phase1.json

Phase 2 (Joint Training):
    python train_tensor_regression.py --cfg expr_setting/tensor_phase2.json
"""

import argparse
import random
import os

# Set environment variables before importing torch/transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import torch
from pathlib import Path
from torch.utils.data import Subset

from src.utils.config import Config
from src.data.tokenizer import ByteLevelTokenizer, HFTokenizerWrapper
from src.data.dataset import GraphTextDataset, create_dataloader
from src.models.clip_model import GraphTextCLIP
from src.models.regression_model import GraphTensorRegressor
from src.training.tensor_trainer import TensorRegressionTrainer


def split_dataset(dataset, val_split: float, seed: int = 42):
    """Split dataset into train and validation sets."""
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    indices = list(range(total_size))
    random.Random(seed).shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Config JSON path for tensor regression",
    )
    return p.parse_args()


def main(cfg_like=None):
    # Configuration
    config = Config.build(cfg_like, default_data_dir="data_preparation/clip_dataset")
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", config.training.device)

    # DEBUG: Show which config file is being used
    print(f"\n[DEBUG] Config file: {cfg_like}")
    print(f"[DEBUG] lambda_sym in config: {config.tensor_regression.get('lambda_sym', 'NOT FOUND')}")
    print(f"[DEBUG] lambda_sym_schedule in config: {config.tensor_regression.get('lambda_sym_schedule', 'NOT FOUND')}")
    print(f"[DEBUG] task in config: {config.tensor_regression.get('task', 'NOT FOUND')}\n")

    config.finalize()

    # Tokenizer
    if config.model.text_backend == "huggingface":
        tokenizer = HFTokenizerWrapper(
            model_name=config.model.text_model_name,
            max_len=config.model.max_seq_length,
            padding=config.tokenizer.tokenizer_padding,
        )
        print(f"Tokenizer: HuggingFace ({config.model.text_model_name})")
        print(f"  Max sequence length: {tokenizer.max_len}")
    else:
        tokenizer = ByteLevelTokenizer(
            pad_token=config.tokenizer.pad_token,
            sot_token=config.tokenizer.sot_token,
            eot_token=config.tokenizer.eot_token,
        )
        print("Tokenizer: ByteLevel")

    # Dataset (must have tensor field)
    normalize_tensor = config.tensor_regression.get("normalize_tensor", True)

    # For elastic_2d_symmetry task, force voigt2d mode
    task = config.tensor_regression.get("task", "regression")
    if task == "elastic_2d_symmetry":
        tensor_mode = "voigt2d"
        print("[INFO] Task is elastic_2d_symmetry, forcing tensor_mode='voigt2d'")
    else:
        tensor_mode = config.tensor_regression.get("tensor_mode", "2d")

    print(f"[DEBUG] tensor_mode: {tensor_mode}")

    dataset = GraphTextDataset(
        jsonl_path=str(config.paths.jsonl_path),
        graph_emb_dir=str(config.paths.graph_emb_dir),
        normalize_tensor=normalize_tensor,
        tensor_mode=tensor_mode,
    )
    graph_dim = dataset.graph_dim
    print(f"Graph embedding dimension: {graph_dim}")
    print(f"Total samples with tensor: {len(dataset)}")
    print(f"Tensor normalization: {normalize_tensor}")
    print(f"Tensor mode: {tensor_mode}")

    # Train/Val split
    train_dataset = dataset
    val_dataset = None
    val_loader = None

    if config.training.use_validation:
        train_dataset, val_dataset = split_dataset(
            dataset,
            val_split=config.training.val_split,
            seed=config.training.val_seed,
        )
        print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    else:
        print(f"Using full dataset for training: {len(dataset)} samples")

    # DataLoaders
    train_loader = create_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_len=config.model.max_seq_length,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )

    if val_dataset is not None:
        val_loader = create_dataloader(
            dataset=val_dataset,
            tokenizer=tokenizer,
            max_len=config.model.max_seq_length,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            drop_last=False,
        )

    # Get CLIP checkpoint path from config
    clip_checkpoint_path = config.tensor_regression.clip_checkpoint_path
    if clip_checkpoint_path is None:
        raise ValueError("clip_checkpoint_path must be specified in tensor_regression config")

    # Load pre-trained CLIP model
    print(f"\nLoading pre-trained CLIP model from: {clip_checkpoint_path}")

    # Load checkpoint to infer architecture
    clip_checkpoint = torch.load(clip_checkpoint_path, map_location="cpu")

    # Try to load config from checkpoint directory
    clip_ckpt_path = Path(clip_checkpoint_path)
    clip_config_path = clip_ckpt_path.parent.parent / "config.json"

    if clip_config_path.exists():
        print(f"Loading CLIP config from: {clip_config_path}")
        clip_config = Config.from_json(clip_config_path)

        # Use CLIP's original architecture config
        # Override freeze settings with current config for tensor regression
        if clip_config.model.text_backend == "huggingface":
            clip_model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=clip_config.model.clip_dim,
                text_backend="huggingface",
                text_model_name=clip_config.model.text_model_name,
                text_pooling=clip_config.model.text_pooling,
                freeze_text_backbone=config.model.freeze_text_backbone,  # Use current config
                train_text_layernorm_only=config.model.train_text_layernorm_only,
                train_top_n_layers=config.model.train_top_n_layers,  # Pass top-N setting
                text_dropout=clip_config.model.text_dropout,
            )
        else:
            clip_model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=clip_config.model.clip_dim,
                text_backend="custom",
                text_width=clip_config.model.text_width,
                max_len=clip_config.model.max_seq_length,
                text_layers=clip_config.model.text_layers,
                text_heads=clip_config.model.text_heads,
                vocab_size=clip_config.model.vocab_size,
                dropout=clip_config.model.dropout,
            )

        print(f"✓ Using CLIP architecture: {clip_config.model.text_backend}, clip_dim={clip_config.model.clip_dim}")
    else:
        # Fallback: use current config (may cause dimension mismatch)
        print("⚠ Warning: CLIP config.json not found, using current config (may cause dimension mismatch)")

        if config.model.text_backend == "huggingface":
            clip_model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=config.model.clip_dim,
                text_backend="huggingface",
                text_model_name=config.model.text_model_name,
                text_pooling=config.model.text_pooling,
                freeze_text_backbone=config.model.freeze_text_backbone,
                train_text_layernorm_only=config.model.train_text_layernorm_only,
                train_top_n_layers=config.model.train_top_n_layers,
                text_dropout=config.model.text_dropout,
            )
        else:
            clip_model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=config.model.clip_dim,
                text_backend="custom",
                text_width=config.model.text_width,
                max_len=config.model.max_seq_length,
                text_layers=config.model.text_layers,
                text_heads=config.model.text_heads,
                vocab_size=config.model.vocab_size,
                dropout=config.model.dropout,
            )

    # Load CLIP weights
    clip_model.load_state_dict(clip_checkpoint, strict=True)
    print("✓ CLIP model loaded successfully")

    # Create tensor regression model based on task type
    task = config.tensor_regression.get("task", "regression")

    if task == "elastic_2d_symmetry":
        # Symmetry-aware 2D elastic tensor regression
        from src.models.elastic_regressor_2d import TextElasticRegressor2D

        regressor = TextElasticRegressor2D(
            clip_model=clip_model,
            head_hidden_dim=config.tensor_regression.head_hidden_dims[0] if config.tensor_regression.head_hidden_dims else None,
            head_dropout=config.tensor_regression.head_dropout,
            beta_prior=config.tensor_regression.get("beta_prior", 1.0),
            freeze_clip=config.tensor_regression.freeze_clip,
            freeze_alpha_head=config.tensor_regression.get("freeze_alpha_head", False),
        )
        print(f"  Model type: Symmetry-aware 2D elastic (TextElasticRegressor2D)")
        print(f"  Beta prior: {config.tensor_regression.get('beta_prior', 1.0)}")
        print(f"  Alpha head frozen: {config.tensor_regression.get('freeze_alpha_head', False)}")

    elif config.tensor_regression.modality == "text":
        from src.models.regression_model import TextTensorRegressor
        regressor = TextTensorRegressor(
            clip_model=clip_model,
            tensor_dim=config.tensor_regression.tensor_dim,
            head_hidden_dims=config.tensor_regression.head_hidden_dims,
            head_dropout=config.tensor_regression.head_dropout,
            freeze_clip=config.tensor_regression.freeze_clip,
            tensor_mode=tensor_mode,
        )
    else:  # graph
        regressor = GraphTensorRegressor(
            clip_model=clip_model,
            tensor_dim=config.tensor_regression.tensor_dim,
            head_hidden_dims=config.tensor_regression.head_hidden_dims,
            head_dropout=config.tensor_regression.head_dropout,
            freeze_clip=config.tensor_regression.freeze_clip,
            tensor_mode=tensor_mode,
        )

    print(f"\n[Regression Model]")
    print(f"  Phase: {config.tensor_regression.training_phase}")
    print(f"  Modality: {config.tensor_regression.modality}")

    # Determine tensor description
    if tensor_mode == "voigt2d":
        tensor_desc = "6 components (C11,C22,C12,C66,C16,C26)"
    elif tensor_mode == "2d":
        tensor_desc = "4 components (c11,c12,c22,c33)"
    else:
        tensor_desc = "9 components (full 3x3)"
    print(f"  Tensor mode: {tensor_mode} ({tensor_desc})")

    # Get output dimension (different attribute names for different models)
    if hasattr(regressor.regression_head, 'output_dim'):
        print(f"  Model output dimension: {regressor.regression_head.output_dim}")
    elif hasattr(regressor.regression_head, 'tensor_head'):
        # ElasticTensorHead
        print(f"  Model output dimension: 6 (Voigt 2D: C11,C22,C12,C66,C16,C26)")
        print(f"  Symmetry head dimension: 4 (tri,rect,tetra,hexa)")

    print(f"  Freeze CLIP: {config.tensor_regression.freeze_clip}")
    print(f"  Head type: {config.tensor_regression.head_type}")

    # Detailed parameter breakdown
    total_params = sum(p.numel() for p in regressor.parameters())
    trainable_params = sum(p.numel() for p in regressor.parameters() if p.requires_grad)

    # CLIP encoder breakdown
    clip_total = sum(p.numel() for p in regressor.clip_model.parameters())
    clip_trainable = sum(p.numel() for p in regressor.clip_model.parameters() if p.requires_grad)

    # Text encoder breakdown
    text_total = sum(p.numel() for p in regressor.clip_model.text_encoder.parameters())
    text_trainable = sum(p.numel() for p in regressor.clip_model.text_encoder.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"    - CLIP: {clip_trainable:,} / {clip_total:,}")
    print(f"      - Text encoder: {text_trainable:,} / {text_total:,}")
    print(f"    - Regression head: {trainable_params - clip_trainable:,}")

    # For Phase 2, we need CLIP data loader
    clip_train_loader = None
    if (config.tensor_regression.training_phase == "phase2" and
        not config.tensor_regression.freeze_clip):
        print("\nPhase 2: Creating CLIP data loader for joint training")
        clip_train_loader = create_dataloader(
            dataset=train_dataset,
            tokenizer=tokenizer,
            max_len=config.model.max_seq_length,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            drop_last=True,
        )

    # Trainer
    trainer = TensorRegressionTrainer(
        model=regressor,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        clip_train_loader=clip_train_loader,
        dataset=dataset,  # Pass dataset for denormalization
    )

    # Train
    print("\n" + "="*60)
    print(f"Starting Training: {config.tensor_regression.training_phase.upper()}")
    print("="*60 + "\n")
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(cfg_like=args.cfg)

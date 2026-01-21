"""Train Graph-Text CLIP model with refactored OOP structure."""
import argparse
import random
import os

# Set environment variables before importing torch/transformers
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism to avoid fork warnings

import torch
from pathlib import Path
from torch.utils.data import Subset

from src.utils.config import Config
from src.data.tokenizer import ByteLevelTokenizer, HFTokenizerWrapper
from src.data.dataset import GraphTextDataset, create_dataloader
from src.models.clip_model import GraphTextCLIP
from src.training.trainer import Trainer


def split_dataset(dataset, val_split: float, seed: int = 42):
    """
    Split dataset into train and validation sets.

    Args:
        dataset: Full dataset
        val_split: Fraction of data for validation (0.0 - 1.0)
        seed: Random seed for reproducibility

    Returns:
        train_dataset, val_dataset (Subset objects)
    """
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Create shuffled indices
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
        default=None,
        help="Config source passed to Config.build(). e.g. 'aaa.json' or json string.",
    )
    return p.parse_args()

def main(cfg_like=None):
    # Configuration
    # config = Config.from_defaults(data_dir="data_preparation/clip_dataset")
    # config.finalize()
    config = Config.build(cfg_like, default_data_dir="data_preparation/clip_dataset")

    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", config.training.device)

    config.finalize()

    # Tokenizer
    if config.model.text_backend == "huggingface":
        tokenizer = HFTokenizerWrapper(
            model_name=config.model.text_model_name,
            max_len=config.model.max_seq_length,
            padding=config.tokenizer.tokenizer_padding,
        )
        print(f"Tokenizer: HuggingFace ({config.model.text_model_name})")
    else:
        tokenizer = ByteLevelTokenizer(
            pad_token=config.tokenizer.pad_token,
            sot_token=config.tokenizer.sot_token,
            eot_token=config.tokenizer.eot_token,
        )
        print("Tokenizer: ByteLevel")

    # Dataset
    dataset = GraphTextDataset(
        jsonl_path=str(config.paths.jsonl_path),
        graph_emb_dir=str(config.paths.graph_emb_dir),
    )
    graph_dim = dataset.graph_dim
    print(f"Graph embedding dimension: {graph_dim}")

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

    (meta_path := config.paths.run_dir / "meta.txt").write_text(
        f"graph_dim: {graph_dim}\n"
        f"total_samples: {len(dataset)}\n"
        f"train_samples: {len(train_dataset)}\n"
        f"val_samples: {len(val_dataset) if val_dataset else 0}\n"
        f"jsonl_path: {config.paths.jsonl_path}\n"
        f"graph_emb_dir: {config.paths.graph_emb_dir}\n",
        encoding="utf-8"
    )

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

    # Model
    if config.model.text_backend == "huggingface":
        model = GraphTextCLIP(
            graph_in_dim=graph_dim,
            clip_dim=config.model.clip_dim,
            text_backend="huggingface",
            text_model_name=config.model.text_model_name,
            text_pooling=config.model.text_pooling,
            freeze_text_backbone=config.model.freeze_text_backbone,
            train_text_layernorm_only=config.model.train_text_layernorm_only,
            text_dropout=config.model.text_dropout,
        )
    else:
        model = GraphTextCLIP(
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
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        full_dataset=dataset,  # For visualization during training
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(cfg_like=args.cfg)

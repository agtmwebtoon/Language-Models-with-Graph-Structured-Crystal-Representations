"""Train Graph-Text CLIP model with refactored OOP structure."""

import torch
from pathlib import Path

from src.utils.config import Config
from src.data.tokenizer import ByteLevelTokenizer
from src.data.dataset import GraphTextDataset, create_dataloader
from src.models.clip_model import GraphTextCLIP
from src.training.trainer import Trainer


def main():
    # Configuration
    config = Config.from_defaults(data_dir="data_preparation/clip_dataset")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.training.device = device
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = ByteLevelTokenizer(
        pad_token=config.tokenizer.pad_token,
        sot_token=config.tokenizer.sot_token,
        eot_token=config.tokenizer.eot_token,
    )

    # Dataset
    dataset = GraphTextDataset(
        jsonl_path=str(config.paths.jsonl_path),
        graph_emb_dir=str(config.paths.graph_emb_dir),
    )
    graph_dim = dataset.graph_dim
    print(f"Graph embedding dimension: {graph_dim}")

    # DataLoader
    train_loader = create_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        max_len=config.model.max_seq_length,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        drop_last=True,
    )

    # Model
    model = GraphTextCLIP(
        graph_in_dim=graph_dim,
        clip_dim=config.model.clip_dim,
        text_width=config.model.text_width,
        max_len=config.model.max_seq_length,
        text_layers=config.model.text_layers,
        text_heads=config.model.text_heads,
        vocab_size=config.model.vocab_size,
        dropout=config.model.dropout,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = Trainer(model=model, config=config, train_loader=train_loader)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()

"""Train Graph-Text CLIP model with refactored OOP structure."""

import torch
from pathlib import Path

from src.utils.config import Config
from src.data.tokenizer import ByteLevelTokenizer, HFTokenizerWrapper
from src.data.dataset import GraphTextDataset, create_dataloader
from src.models.clip_model import GraphTextCLIP
from src.training.trainer import Trainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(cfg_like=None):
    # Configuration
    # config = Config.from_defaults(data_dir="data_preparation/clip_dataset")
    # config.finalize()
    config = Config.build(cfg_like, default_data_dir="data_preparation/clip_dataset")

    # 2) device 설정
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", config.training.device)

    # 3) ✅ 여기서만 finalize (run_dir 생성 + dump)
    config.finalize()

    text_backend = getattr(config.model, "text_backend", "custom")

    if text_backend == "t5":
        tokenizer = HFTokenizerWrapper(
            model_name=config.model.t5_model_name,
            max_len=config.model.max_seq_length,
            padding=getattr(config.model, "t5_padding", "max_length"),
        )
        print(f"Tokenizer: HF ({tokenizer.tokenizer.name_or_path})")

    else:
        # Tokenizer
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

    graph_dim = dataset.graph_dim

    (meta_path := config.paths.run_dir / "meta.txt").write_text(
        f"graph_dim: {graph_dim}\n"
        f"jsonl_path: {config.paths.jsonl_path}\n"
        f"graph_emb_dir: {config.paths.graph_emb_dir}\n",
        encoding="utf-8"
    )

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

    if text_backend == "t5":
        model = GraphTextCLIP(
            graph_in_dim=graph_dim,
            clip_dim=config.model.clip_dim,
            text_backend="t5",
            t5_model_name=getattr(config.model, "t5_model_name", "t5-base"),
            t5_pooling=getattr(config.model, "t5_pooling", "mean"),
            freeze_t5=getattr(config.model, "freeze_t5", False),
            train_layernorm_only=getattr(config.model, "train_layernorm_only", False),
            t5_dropout=getattr(config.model, "t5_dropout", 0.0)
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
            dropout=config.model.dropout
        )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = Trainer(model=model, config=config, train_loader=train_loader)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()

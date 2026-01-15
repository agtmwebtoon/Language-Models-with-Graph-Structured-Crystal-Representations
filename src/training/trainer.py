"""Training framework for Graph-Text CLIP model."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

from ..models.clip_model import GraphTextCLIP
from ..utils.config import Config


class Trainer:
    """Unified training framework for CLIP model."""

    def __init__(
        self,
        model: GraphTextCLIP,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(config.training.device)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.current_epoch = 0
        self.global_step = 0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(self.train_loader, 1):
            graph = batch["graph"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn = batch["attention_mask"].to(self.device, non_blocking=True)

            loss, logits = self.model(graph, input_ids, attn)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if step % self.config.training.log_interval == 0:
                acc_g2t, acc_t2g = self.model.compute_retrieval_accuracy(logits.detach())
                print(
                    f"[Epoch {self.current_epoch} | Step {step}] "
                    f"loss={loss.item():.4f} "
                    f"acc(g→t)={acc_g2t:.3f} "
                    f"acc(t→g)={acc_t2g:.3f}"
                )

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_acc_g2t = 0.0
        total_acc_t2g = 0.0
        num_batches = 0

        for batch in self.val_loader:
            graph = batch["graph"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn = batch["attention_mask"].to(self.device, non_blocking=True)

            loss, logits = self.model(graph, input_ids, attn)
            acc_g2t, acc_t2g = self.model.compute_retrieval_accuracy(logits)

            total_loss += loss.item()
            total_acc_g2t += acc_g2t
            total_acc_t2g += acc_t2g
            num_batches += 1

        return {
            "val_loss": total_loss / max(1, num_batches),
            "val_acc_g2t": total_acc_g2t / max(1, num_batches),
            "val_acc_t2g": total_acc_t2g / max(1, num_batches),
        }

    def train(self, epochs: Optional[int] = None):
        """Run full training loop."""
        epochs = epochs or self.config.training.epochs
        save_path = self.config.paths.checkpoint_path

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_loss = self.train_epoch()

            print(f"Epoch {epoch} avg loss = {train_loss:.4f}")

            # Validation
            if self.val_loader:
                val_metrics = self.evaluate()
                print(f"Validation: {val_metrics}")

            # Save checkpoint
            if save_path:
                self.save_checkpoint(save_path)
                print(f"✓ Saved checkpoint: {save_path}")

        print("Training complete.")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: Path, strict: bool = True):
        """Load model checkpoint."""
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=strict)
        self.model.to(self.device)
        print(f"✓ Loaded checkpoint: {path}")

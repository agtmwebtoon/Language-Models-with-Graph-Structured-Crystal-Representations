"""Training framework for Graph-Text CLIP model."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Optional
import random

from ..models.clip_model import GraphTextCLIP
from ..utils.config import Config

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        delta: float = 0.0,
        verbose: bool = True,
    ):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        self.is_better = self._get_comparison_fn()

    def _get_comparison_fn(self):
        if self.mode == "min":
            return lambda new, best: new < best - self.delta
        else:
            return lambda new, best: new > best + self.delta

    def __call__(self, score: float, epoch: int) -> bool:
        """
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Improved! Best score: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] Stopped! Best was epoch {self.best_epoch} with score {self.best_score:.4f}")
                return True

        return False


class Trainer:
    """Unified training framework for CLIP model."""

    def __init__(
        self,
        model: GraphTextCLIP,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        full_dataset = None,  # For visualization during training
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.full_dataset = full_dataset

        self.device = torch.device(config.training.device)
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        self.current_epoch = 0
        self.global_step = 0

        # Early stopping
        self.early_stopping = None
        if config.training.early_stopping and val_loader is not None:
            self.early_stopping = EarlyStopping(
                patience=config.training.early_stopping_patience,
                mode=config.training.early_stopping_mode,
                verbose=True,
            )

        # WandB
        self.use_wandb = config.training.use_wandb
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                raise ImportError("wandb is not installed. Install with: pip install wandb")

            wandb.init(
                project=config.training.wandb_project or "graph-text-clip",
                entity=config.training.wandb_entity,
                name=config.training.wandb_run_name or config.paths.run_name,
                tags=config.training.wandb_tags,
                config={
                    "model": {
                        "clip_dim": config.model.clip_dim,
                        "text_backend": config.model.text_backend,
                        "text_model_name": config.model.text_model_name,
                    },
                    "training": {
                        "batch_size": config.training.batch_size,
                        "learning_rate": config.training.learning_rate,
                        "weight_decay": config.training.weight_decay,
                        "epochs": config.training.epochs,
                    },
                },
            )
            wandb.watch(self.model, log="gradients", log_freq=100)

    def _visualize_embeddings(self, epoch: int):
        """
        Run visualization at a given epoch.
        """
        if self.full_dataset is None:
            print(f"[Viz] Skipping: no dataset provided")
            return

        try:
            from ..visualization.embedding_extractor import EmbeddingExtractor
            from ..visualization.plots import plot_tsne_pairs, plot_similarity_heatmap, plot_diag_hist, TSNEConfig
            from ..data.dataset import create_dataloader

            print(f"[Viz] Starting visualization at epoch {epoch}...")

            # Create subset for visualization
            n_samples = min(self.config.training.visualize_n_samples, len(self.full_dataset))
            indices = random.sample(range(len(self.full_dataset)), n_samples)
            viz_dataset = Subset(self.full_dataset, indices)

            # Get tokenizer from config
            from ..data.tokenizer import ByteLevelTokenizer, HFTokenizerWrapper
            if self.config.model.text_backend == "huggingface":
                tokenizer = HFTokenizerWrapper(
                    model_name=self.config.model.text_model_name,
                    max_len=self.config.model.max_seq_length,
                    padding=self.config.tokenizer.tokenizer_padding,
                )
            else:
                tokenizer = ByteLevelTokenizer(
                    pad_token=self.config.tokenizer.pad_token,
                    sot_token=self.config.tokenizer.sot_token,
                    eot_token=self.config.tokenizer.eot_token,
                )

            viz_loader = create_dataloader(
                dataset=viz_dataset,
                tokenizer=tokenizer,
                max_len=self.config.model.max_seq_length,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=0,  # avoid multiprocessing issues
                drop_last=False,
            )

            # Extract embeddings
            extractor = EmbeddingExtractor(self.model, self.device)
            emb_batch = extractor.extract(viz_loader, max_items=n_samples)

            # Create viz directory for this epoch
            epoch_viz_dir = self.config.paths.viz_dir / f"epoch_{epoch:03d}"
            epoch_viz_dir.mkdir(parents=True, exist_ok=True)

            # Generate plots
            tsne_cfg = TSNEConfig(perplexity=30, seed=42, connect_k=200)
            plot_tsne_pairs(
                emb_batch.G, emb_batch.T,
                out_path=str(epoch_viz_dir / "tsne_pairs.png"),
                cfg=tsne_cfg
            )
            plot_similarity_heatmap(
                emb_batch.G, emb_batch.T,
                out_path=str(epoch_viz_dir / "similarity_heatmap.png")
            )
            plot_diag_hist(
                emb_batch.G, emb_batch.T,
                out_path=str(epoch_viz_dir / "diagonal_hist.png")
            )

            print(f"[Viz] Saved to {epoch_viz_dir}")

            # Log to WandB if enabled
            if self.use_wandb:
                wandb.log({
                    f"viz/tsne_epoch_{epoch}": wandb.Image(str(epoch_viz_dir / "tsne_pairs.png")),
                    f"viz/heatmap_epoch_{epoch}": wandb.Image(str(epoch_viz_dir / "similarity_heatmap.png")),
                    f"viz/hist_epoch_{epoch}": wandb.Image(str(epoch_viz_dir / "diagonal_hist.png")),
                    "epoch": epoch,
                })

        except Exception as e:
            print(f"[Viz] Error during visualization: {e}")
            import traceback
            traceback.print_exc()

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

                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/acc_g2t": acc_g2t,
                        "train/acc_t2g": acc_t2g,
                        "train/step": self.global_step,
                        "train/epoch": self.current_epoch,
                    })

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
        best_checkpoint_path = self.config.paths.checkpoint_dir / "best_model.pt"

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_loss = self.train_epoch()

            print(f"Epoch {epoch} avg loss = {train_loss:.4f}")

            # Log to WandB
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                })

            # Validation
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.evaluate()
                print(f"Validation: {val_metrics}")

                if self.use_wandb:
                    wandb.log({
                        "val/loss": val_metrics.get("val_loss"),
                        "val/acc_g2t": val_metrics.get("val_acc_g2t"),
                        "val/acc_t2g": val_metrics.get("val_acc_t2g"),
                        "epoch": epoch,
                    })

            # Save checkpoint
            if save_path:
                self.save_checkpoint(save_path)
                print(f"✓ Saved checkpoint: {save_path}")

            # Visualization
            if (self.config.training.visualize_during_training and
                epoch % self.config.training.visualize_interval == 0):
                self._visualize_embeddings(epoch)

            # Early stopping
            if self.early_stopping is not None and val_metrics:
                metric_name = self.config.training.early_stopping_metric
                score = val_metrics.get(metric_name)

                if score is not None:
                    should_stop = self.early_stopping(score, epoch)

                    # Save best model
                    if self.early_stopping.counter == 0:
                        self.save_checkpoint(best_checkpoint_path)
                        print(f"✓ Saved best checkpoint: {best_checkpoint_path}")

                        if self.use_wandb:
                            wandb.run.summary["best_epoch"] = epoch
                            wandb.run.summary[f"best_{metric_name}"] = score

                    if should_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        print("Training complete.")

        if self.use_wandb:
            wandb.finish()

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

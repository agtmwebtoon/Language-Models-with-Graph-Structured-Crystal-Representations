"""Trainer for tensor regression task."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from ..models.regression_model import GraphTensorRegressor, TextTensorRegressor
from ..models.clip_model import GraphTextCLIP
from ..utils.config import Config

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TensorRegressionTrainer:
    """
    Trainer for tensor regression using pre-trained CLIP encoders.

    Supports two training phases:
    - Phase 1: Freeze CLIP encoder, train regression head only (fast sanity check)
    - Phase 2: Fine-tune encoder + head with joint CLIP + regression loss
    """

    def __init__(
        self,
        model: GraphTensorRegressor,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        clip_train_loader: Optional[DataLoader] = None,  # For Phase 2 CLIP loss
        dataset = None,  # For denormalization
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clip_train_loader = clip_train_loader
        self.dataset = dataset  # Store dataset for denormalization

        self.device = torch.device(config.training.device)
        self.model.to(self.device)

        # Get learning rate
        lr = config.tensor_regression.regression_lr or config.training.learning_rate

        # Optimizer
        if config.tensor_regression.freeze_clip:
            # Phase 1: Only optimize regression head
            params = self.model.regression_head.parameters()
        else:
            # Phase 2: Optimize all parameters
            params = self.model.parameters()

        self.optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=config.training.weight_decay,
        )

        # Loss function
        self.loss_fn = self._get_loss_fn(config.tensor_regression.loss_fn)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # WandB
        self.use_wandb = config.training.use_wandb
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                raise ImportError("wandb is not installed. Install with: pip install wandb")

            phase = config.tensor_regression.training_phase
            modality = config.tensor_regression.modality

            wandb.init(
                project=config.training.wandb_project or "graph-text-tensor-regression",
                entity=config.training.wandb_entity,
                name=config.training.wandb_run_name or f"{phase}_{modality}_{config.paths.run_name}",
                tags=config.training.wandb_tags or [phase, modality],
                config={
                    "model": {
                        "clip_dim": config.model.clip_dim,
                        "text_backend": config.model.text_backend,
                    },
                    "regression": {
                        "tensor_dim": config.tensor_regression.tensor_dim,
                        "head_type": config.tensor_regression.head_type,
                        "training_phase": phase,
                        "modality": modality,
                        "freeze_clip": config.tensor_regression.freeze_clip,
                    },
                    "training": {
                        "batch_size": config.training.batch_size,
                        "learning_rate": lr,
                        "epochs": config.training.epochs,
                    },
                },
            )

    def _get_loss_fn(self, loss_name: str):
        """Get loss function by name."""
        if loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "mae":
            return nn.L1Loss()
        elif loss_name == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # For computing R² on training set
        all_train_preds = []
        all_train_targets = []

        # Track symmetry-specific metrics
        sym_loss_sum = 0.0
        alpha_entropy_sum = 0.0

        # Track CLIP loss for Phase 2
        clip_loss_sum = 0.0
        clip_loss_count = 0

        for step, batch in enumerate(self.train_loader, 1):
            tensor_target = batch["tensor"].to(self.device, non_blocking=True)

            # Check if model has symmetry awareness (new API)
            has_symmetry = hasattr(self.model, 'regression_head') and hasattr(self.model.regression_head, 'alpha_head')

            # Forward pass based on modality
            if self.config.tensor_regression.modality == "text":
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                crystal_systems = batch.get("group", None) if has_symmetry else None

                if has_symmetry:
                    # TextElasticRegressor2D returns (pred6, alpha, z_logits)
                    tensor_pred, alpha, z_logits = self.model(input_ids, attention_mask, crystal_systems=crystal_systems)
                else:
                    tensor_pred = self.model(input_ids, attention_mask)
                    alpha = None
            else:  # graph
                graph = batch["graph"].to(self.device, non_blocking=True)
                if has_symmetry:
                    tensor_pred, alpha, z_logits = self.model(graph)
                else:
                    tensor_pred = self.model(graph)
                    alpha = None

            # Collect predictions and targets for R² computation
            all_train_preds.append(tensor_pred.detach().cpu().numpy())
            all_train_targets.append(tensor_target.detach().cpu().numpy())

            # Compute loss
            if has_symmetry and alpha is not None:
                # Symmetry-aware loss (using ElasticSymmetryLoss)
                if not hasattr(self, 'symmetry_loss_fn'):
                    from ..models.elastic_loss_2d import ElasticSymmetryLoss
                    lambda_sym = self.config.tensor_regression.get("lambda_sym", 0.1)
                    lambda_sym_schedule = self.config.tensor_regression.get("lambda_sym_schedule", None)

                    # Debug: check config loading
                    print(f"[DEBUG Trainer] lambda_sym from config: {lambda_sym}")
                    print(f"[DEBUG Trainer] lambda_sym_schedule from config: {lambda_sym_schedule}")

                    self.symmetry_loss_fn = ElasticSymmetryLoss(
                        lambda_sym=lambda_sym,
                        lambda_sym_schedule=lambda_sym_schedule,
                        data_loss_fn=self.config.tensor_regression.loss_fn,
                    )

                # Update epoch for scheduling
                self.symmetry_loss_fn.set_epoch(self.current_epoch)

                # Compute loss
                loss, loss_dict = self.symmetry_loss_fn(tensor_pred, tensor_target, alpha)

                sym_loss_sum += loss_dict.get("sym_loss", 0.0)
                alpha_entropy_sum += alpha.detach().cpu().numpy().std()  # Track alpha diversity
            else:
                # Standard regression loss
                loss = self.loss_fn(tensor_pred, tensor_target)

            # Phase 2: Add CLIP loss if not freezing
            regression_loss_value = loss.item()  # Store before combining
            clip_loss_value = 0.0

            if (not self.config.tensor_regression.freeze_clip and
                self.clip_train_loader is not None and
                self.config.tensor_regression.training_phase == "phase2"):

                # Get a batch from CLIP data
                try:
                    clip_batch = next(self.clip_iter)
                except (StopIteration, AttributeError):
                    self.clip_iter = iter(self.clip_train_loader)
                    clip_batch = next(self.clip_iter)

                clip_graph = clip_batch["graph"].to(self.device, non_blocking=True)
                input_ids = clip_batch["input_ids"].to(self.device, non_blocking=True)
                attn_mask = clip_batch["attention_mask"].to(self.device, non_blocking=True)

                clip_loss = self.model.compute_clip_loss(clip_graph, input_ids, attn_mask)
                clip_loss_value = clip_loss.item()

                # Track CLIP loss
                clip_loss_sum += clip_loss_value
                clip_loss_count += 1

                # Weighted combination
                loss = (self.config.tensor_regression.regression_loss_weight * loss +
                        self.config.tensor_regression.clip_loss_weight * clip_loss)

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if step % self.config.training.log_interval == 0:
                if has_symmetry:
                    # Show current lambda_sym value
                    current_lambda = self.symmetry_loss_fn.get_lambda_sym() if hasattr(self, 'symmetry_loss_fn') else 0
                    avg_sym_loss = sym_loss_sum / step
                    print(f"[Epoch {self.current_epoch} | Step {step}] "
                          f"loss={loss.item():.6f} | sym_loss={avg_sym_loss:.6f} | "
                          f"λ_sym={current_lambda:.1f} | contribution={current_lambda*avg_sym_loss:.6f}")
                elif self.config.tensor_regression.training_phase == "phase2" and clip_loss_value > 0:
                    print(f"[Epoch {self.current_epoch} | Step {step}] "
                          f"loss={loss.item():.6f} (reg={regression_loss_value:.6f}, clip={clip_loss_value:.6f})")
                else:
                    print(f"[Epoch {self.current_epoch} | Step {step}] loss={loss.item():.6f}")

                if self.use_wandb:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/step": self.global_step,
                        "train/epoch": self.current_epoch,
                    }
                    if has_symmetry:
                        avg_sym_loss = sym_loss_sum / step
                        current_lambda = self.symmetry_loss_fn.get_lambda_sym() if hasattr(self, 'symmetry_loss_fn') else 0
                        log_dict["train/sym_loss"] = avg_sym_loss
                        log_dict["train/lambda_sym"] = current_lambda
                        log_dict["train/sym_contribution"] = current_lambda * avg_sym_loss
                        log_dict["train/alpha_entropy"] = alpha_entropy_sum / step
                    if self.config.tensor_regression.training_phase == "phase2" and clip_loss_value > 0:
                        log_dict["train/regression_loss"] = regression_loss_value
                        log_dict["train/clip_loss"] = clip_loss_value
                    wandb.log(log_dict)

        avg_loss = total_loss / max(1, num_batches)

        # Compute training R² and MAE
        all_train_preds = np.concatenate(all_train_preds, axis=0)
        all_train_targets = np.concatenate(all_train_targets, axis=0)

        # Denormalize if needed
        if self.dataset is not None and hasattr(self.dataset, 'denormalize_tensor'):
            all_train_preds_denorm = np.array([
                self.dataset.denormalize_tensor(pred) for pred in all_train_preds
            ])
            all_train_targets_denorm = np.array([
                self.dataset.denormalize_tensor(target) for target in all_train_targets
            ])
        else:
            all_train_preds_denorm = all_train_preds
            all_train_targets_denorm = all_train_targets

        train_mae = float(np.abs(all_train_preds_denorm - all_train_targets_denorm).mean())
        train_r2 = float(r2_score(all_train_targets_denorm.flatten(), all_train_preds_denorm.flatten()))

        # Compute average loss components for epoch summary
        avg_sym_loss = sym_loss_sum / max(1, num_batches)
        if hasattr(self, 'symmetry_loss_fn'):
            current_lambda = self.symmetry_loss_fn.get_lambda_sym()
            sym_contribution = current_lambda * avg_sym_loss
        else:
            current_lambda = 0
            sym_contribution = 0

        # Compute average CLIP loss if Phase 2
        avg_clip_loss = clip_loss_sum / max(1, clip_loss_count) if clip_loss_count > 0 else 0
        clip_contribution = self.config.tensor_regression.clip_loss_weight * avg_clip_loss if clip_loss_count > 0 else 0

        return {
            "train_loss": avg_loss,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "sym_loss": avg_sym_loss,
            "lambda_sym": current_lambda,
            "sym_contribution": sym_contribution,
            "clip_loss": avg_clip_loss,
            "clip_contribution": clip_contribution,
        }

    @torch.no_grad()
    def evaluate(self, log_samples: bool = False, num_samples_to_log: int = 5,
                 generate_parity_plot: bool = False) -> Dict[str, float]:
        """Evaluate on validation set with optional sample logging and parity plot."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # Collect all predictions and targets for parity plot
        all_preds = []
        all_targets = []

        # Check if model has symmetry awareness (new API)
        has_symmetry = hasattr(self.model, 'regression_head') and hasattr(self.model.regression_head, 'alpha_head')

        for batch in self.val_loader:
            tensor_target = batch["tensor"].to(self.device, non_blocking=True)

            # Forward pass based on modality
            if self.config.tensor_regression.modality == "text":
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                crystal_systems = batch.get("group", None) if has_symmetry else None

                if has_symmetry:
                    # TextElasticRegressor2D returns (pred6, alpha, z_logits)
                    tensor_pred, alpha, z_logits = self.model(input_ids, attention_mask, crystal_systems=crystal_systems)
                else:
                    tensor_pred = self.model(input_ids, attention_mask)
                    alpha = None
            else:  # graph
                graph = batch["graph"].to(self.device, non_blocking=True)
                if has_symmetry:
                    tensor_pred, alpha, z_logits = self.model(graph)
                else:
                    tensor_pred = self.model(graph)
                    alpha = None

            # Compute loss
            if has_symmetry and alpha is not None:
                # Use symmetry loss function
                if not hasattr(self, 'symmetry_loss_fn'):
                    from ..models.elastic_loss_2d import ElasticSymmetryLoss
                    lambda_sym = self.config.tensor_regression.get("lambda_sym", 0.1)
                    lambda_sym_schedule = self.config.tensor_regression.get("lambda_sym_schedule", None)
                    self.symmetry_loss_fn = ElasticSymmetryLoss(
                        lambda_sym=lambda_sym,
                        lambda_sym_schedule=lambda_sym_schedule,
                        data_loss_fn=self.config.tensor_regression.loss_fn,
                    )

                self.symmetry_loss_fn.set_epoch(self.current_epoch)
                loss, _ = self.symmetry_loss_fn(tensor_pred, tensor_target, alpha)
            else:
                loss = self.loss_fn(tensor_pred, tensor_target)

            total_loss += loss.item()
            num_batches += 1

            # Collect for parity plot
            if generate_parity_plot:
                all_preds.append(tensor_pred.cpu().numpy())
                all_targets.append(tensor_target.cpu().numpy())

        val_loss = total_loss / max(1, num_batches)

        # Generate parity plot if requested
        if generate_parity_plot and len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            # Denormalize if needed
            if self.dataset is not None and hasattr(self.dataset, 'denormalize_tensor'):
                all_preds_denorm = np.array([
                    self.dataset.denormalize_tensor(pred) for pred in all_preds
                ])
                all_targets_denorm = np.array([
                    self.dataset.denormalize_tensor(target) for target in all_targets
                ])
            else:
                all_preds_denorm = all_preds
                all_targets_denorm = all_targets

            # Compute metrics
            mae = float(np.abs(all_preds_denorm - all_targets_denorm).mean())
            r2 = float(r2_score(all_targets_denorm.flatten(), all_preds_denorm.flatten()))

            print(f"\nValidation Metrics (Denormalized): MAE={mae:.4f}, R²={r2:.4f}")

            # Generate and log parity plot
            self._plot_parity_during_training(
                all_targets_denorm,
                all_preds_denorm,
                mae,
                r2,
                epoch=self.current_epoch
            )

        return {"val_loss": val_loss}

    @torch.no_grad()
    def final_evaluation(self) -> Dict[str, float]:
        """
        Final evaluation on entire validation set with denormalization.
        Computes MAE and R2 score, generates parity plot.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        all_preds = []
        all_targets = []

        # Collect all predictions and targets
        for batch in self.val_loader:
            tensor_target = batch["tensor"].to(self.device, non_blocking=True)

            # Forward pass based on modality
            if self.config.tensor_regression.modality == "text":
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                tensor_pred = self.model(input_ids, attention_mask)
            else:  # graph
                graph = batch["graph"].to(self.device, non_blocking=True)
                tensor_pred = self.model(graph)

            all_preds.append(tensor_pred.cpu().numpy())
            all_targets.append(tensor_target.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)  # [N, 9]
        all_targets = np.concatenate(all_targets, axis=0)  # [N, 9]

        # Denormalize if needed
        if self.dataset is not None and hasattr(self.dataset, 'denormalize_tensor'):
            print("Denormalizing predictions and targets...")
            all_preds_denorm = np.array([
                self.dataset.denormalize_tensor(pred) for pred in all_preds
            ])
            all_targets_denorm = np.array([
                self.dataset.denormalize_tensor(target) for target in all_targets
            ])
        else:
            all_preds_denorm = all_preds
            all_targets_denorm = all_targets

        # Compute metrics
        mae = float(np.abs(all_preds_denorm - all_targets_denorm).mean())
        r2 = float(r2_score(all_targets_denorm.flatten(), all_preds_denorm.flatten()))

        print(f"\nFinal Validation Metrics (Denormalized):")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")

        # Generate parity plot
        self._plot_parity(all_targets_denorm, all_preds_denorm, mae, r2)

        # Log to WandB
        if self.use_wandb:
            wandb.log({
                "final/val_mae": mae,
                "final/val_r2": r2,
            })

        return {"mae": mae, "r2": r2}

    def _plot_parity_during_training(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        mae: float,
        r2: float,
        epoch: int,
    ):
        """Generate and log parity plot during training to WandB."""
        # Determine number of components and layout based on tensor mode
        tensor_mode = getattr(self.model, 'tensor_mode', 'full')
        num_components = predictions.shape[1]  # Use actual shape

        if tensor_mode == "voigt2d" or num_components == 6:
            # Voigt 2D: C11, C22, C12, C66, C16, C26
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            component_names = ['C11', 'C22', 'C12', 'C66', 'C16', 'C26']
        elif tensor_mode == "2d" or num_components == 4:
            # 2D materials: c11, c12, c22, c33
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            component_names = ['C11', 'C12', 'C22', 'C33']
        else:
            # Full 3x3 tensor
            num_components = 9
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()
            component_names = [
                'C11', 'C12', 'C13',
                'C21', 'C22', 'C23',
                'C31', 'C32', 'C33'
            ]

        # Plot each component separately
        for idx in range(num_components):
            ax = axes[idx]
            target_comp = targets[:, idx]
            pred_comp = predictions[:, idx]

            # Scatter plot
            ax.scatter(target_comp, pred_comp, alpha=0.3, s=10, edgecolors='none')

            # Diagonal line
            min_val = min(target_comp.min(), pred_comp.min())
            max_val = max(target_comp.max(), pred_comp.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, alpha=0.7)

            # Compute component-wise metrics
            comp_mae = float(np.abs(pred_comp - target_comp).mean())
            comp_r2 = float(r2_score(target_comp, pred_comp))

            # Labels and title
            ax.set_xlabel('Ground Truth', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            ax.set_title(f'{component_names[idx]}\nMAE={comp_mae:.4f}, R²={comp_r2:.4f}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        # Overall title
        mode_str = "2D Material" if tensor_mode == "2d" else "Full Tensor"
        fig.suptitle(f'Validation Parity Plots - {mode_str} (Epoch {epoch})\nOverall MAE={mae:.4f}, R²={r2:.4f}',
                     fontsize=16, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Log to WandB
        if self.use_wandb:
            wandb.log({
                "val/parity_plot": wandb.Image(fig),
                "val/mae_denormalized": mae,
                "val/r2": r2,
                "epoch": epoch,
            })

        plt.close()

    def _plot_parity(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        mae: float,
        r2: float,
    ):
        """Generate and save parity plot for tensor predictions (final evaluation)."""
        # Determine number of components and layout based on tensor mode
        tensor_mode = getattr(self.model, 'tensor_mode', 'full')

        if tensor_mode == "2d":
            # 2D materials: c11, c12, c22, c33
            num_components = 4
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.flatten()
            component_names = ['C11', 'C12', 'C22', 'C33']
        else:
            # Full 3x3 tensor
            num_components = 9
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()
            component_names = [
                'C11', 'C12', 'C13',
                'C21', 'C22', 'C23',
                'C31', 'C32', 'C33'
            ]

        # Plot each component separately
        for idx in range(num_components):
            ax = axes[idx]
            target_comp = targets[:, idx]
            pred_comp = predictions[:, idx]

            # Scatter plot
            ax.scatter(target_comp, pred_comp, alpha=0.3, s=10, edgecolors='none')

            # Diagonal line
            min_val = min(target_comp.min(), pred_comp.min())
            max_val = max(target_comp.max(), pred_comp.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5, alpha=0.7)

            # Compute component-wise metrics
            comp_mae = float(np.abs(pred_comp - target_comp).mean())
            comp_r2 = float(r2_score(target_comp, pred_comp))

            # Labels and title
            ax.set_xlabel('Ground Truth', fontsize=10)
            ax.set_ylabel('Predicted', fontsize=10)
            ax.set_title(f'{component_names[idx]}\nMAE={comp_mae:.4f}, R²={comp_r2:.4f}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        # Overall title
        mode_str = "2D Material" if tensor_mode == "2d" else "Full Tensor"
        fig.suptitle(f'Tensor Component Parity Plots - {mode_str}\nOverall MAE={mae:.4f}, R²={r2:.4f}',
                     fontsize=16, fontweight='bold')

        # Save plot
        plot_path = self.config.paths.viz_dir / "final_parity_plot.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved parity plot: {plot_path}")

        # Log to WandB
        if self.use_wandb:
            wandb.log({
                "final/parity_plot": wandb.Image(str(plot_path))
            })

    def train(self, epochs: Optional[int] = None):
        """Run full training loop."""
        epochs = epochs or self.config.training.epochs
        save_path = self.config.paths.checkpoint_dir / "tensor_regression_model.pt"
        best_save_path = self.config.paths.checkpoint_dir / "best_tensor_regression_model.pt"

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()

            print(f"Epoch {epoch} train loss = {train_metrics['train_loss']:.6f}, "
                  f"train MAE = {train_metrics['train_mae']:.4f}, "
                  f"train R² = {train_metrics['train_r2']:.4f}")

            # Log to WandB
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train/epoch_loss": train_metrics["train_loss"],
                    "train/mae_denormalized": train_metrics["train_mae"],
                    "train/r2": train_metrics["train_r2"],
                }

                # Add symmetry metrics if available
                if "sym_loss" in train_metrics and train_metrics["sym_loss"] > 0:
                    log_dict["train/epoch_sym_loss"] = train_metrics["sym_loss"]
                    log_dict["train/epoch_lambda_sym"] = train_metrics["lambda_sym"]
                    log_dict["train/epoch_sym_contribution"] = train_metrics["sym_contribution"]

                # Add CLIP metrics if Phase 2
                if "clip_loss" in train_metrics and train_metrics["clip_loss"] > 0:
                    log_dict["train/epoch_clip_loss"] = train_metrics["clip_loss"]
                    log_dict["train/epoch_clip_contribution"] = train_metrics["clip_contribution"]

                # Compute pure data loss (regression only)
                total_contributions = train_metrics.get("sym_contribution", 0) + train_metrics.get("clip_contribution", 0)
                log_dict["train/epoch_data_loss"] = train_metrics["train_loss"] - total_contributions

                wandb.log(log_dict)

            # Validation
            val_metrics = {}
            if self.val_loader:
                # Determine if we should generate parity plot this epoch
                generate_plot = (
                    self.config.tensor_regression.log_parity_plot and
                    epoch % self.config.tensor_regression.parity_plot_interval == 0
                )

                val_metrics = self.evaluate(
                    log_samples=False,
                    generate_parity_plot=generate_plot
                )
                print(f"Validation: {val_metrics}")

                if self.use_wandb:
                    wandb.log({
                        "val/loss": val_metrics["val_loss"],
                        "epoch": epoch,
                    })

                # Save best model
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint(best_save_path)
                    print(f"✓ Saved best model: {best_save_path}")

            # Save latest checkpoint
            if save_path:
                self.save_checkpoint(save_path)
                print(f"✓ Saved checkpoint: {save_path}")

        # Final evaluation on validation set
        if self.val_loader:
            print("\n" + "="*80)
            print("Final Evaluation on Validation Set")
            print("="*80)
            final_metrics = self.final_evaluation()

            # Update summary with final metrics
            summary = {
                "phase": self.config.tensor_regression.training_phase,
                "modality": self.config.tensor_regression.modality,
                "final_train_loss": train_metrics["train_loss"],
                "best_val_loss": self.best_val_loss,
                "epochs_trained": epochs,
                "final_val_mae": final_metrics.get("mae", None),
                "final_val_r2": final_metrics.get("r2", None),
            }
        else:
            summary = {
                "phase": self.config.tensor_regression.training_phase,
                "modality": self.config.tensor_regression.modality,
                "final_train_loss": train_metrics["train_loss"],
                "best_val_loss": self.best_val_loss if self.val_loader else None,
                "epochs_trained": epochs,
            }

        summary_path = self.config.paths.run_dir / "tensor_regression_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("Training complete.")
        if self.use_wandb:
            wandb.finish()

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save both regression head and CLIP model if not frozen
        checkpoint = {
            "regression_head": self.model.regression_head.state_dict(),
            "config": {
                "tensor_dim": self.config.tensor_regression.tensor_dim,
                "head_type": self.config.tensor_regression.head_type,
                "head_hidden_dims": self.config.tensor_regression.head_hidden_dims,
                "freeze_clip": self.config.tensor_regression.freeze_clip,
            },
            "clip_config": {
                "clip_dim": self.config.model.clip_dim,
                "text_backend": self.config.model.text_backend,
                "text_model_name": self.config.model.text_model_name,
                "text_pooling": self.config.model.text_pooling,
                "text_dropout": self.config.model.text_dropout,
                "text_width": self.config.model.text_width,
                "text_layers": self.config.model.text_layers,
                "text_heads": self.config.model.text_heads,
                "vocab_size": self.config.model.vocab_size,
                "max_seq_length": self.config.model.max_seq_length,
                "dropout": self.config.model.dropout,
            },
        }

        if not self.config.tensor_regression.freeze_clip:
            checkpoint["clip_model"] = self.model.clip_model.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        self.model.regression_head.load_state_dict(checkpoint["regression_head"])

        if "clip_model" in checkpoint:
            self.model.clip_model.load_state_dict(checkpoint["clip_model"])

        self.model.to(self.device)
        print(f"✓ Loaded checkpoint: {path}")

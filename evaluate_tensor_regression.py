"""
Evaluate trained tensor regression model on validation/test set.

Usage:
    python evaluate_tensor_regression.py \
        --checkpoint runs/phase2_graph_xxx/checkpoints/best_tensor_regression_model.pt \
        --cfg expr_setting/tensor_phase2.json \
        --clip_checkpoint runs/clip_xxx/checkpoints/graph_text_clip.pt
"""

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from sklearn.metrics import r2_score
from torch.utils.data import Subset

from src.utils.config import Config
from src.data.tokenizer import ByteLevelTokenizer, HFTokenizerWrapper
from src.data.dataset import GraphTextDataset, create_dataloader
from src.models.clip_model import GraphTextCLIP
from src.models.regression_model import GraphTensorRegressor


def split_dataset(dataset, val_split: float, seed: int = 42):
    """Split dataset into train and validation sets (same as training)."""
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
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained tensor regression checkpoint",
    )
    p.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Config JSON path (same as training)",
    )
    p.add_argument(
        "--clip_checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained CLIP checkpoint",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as checkpoint dir)",
    )
    return p.parse_args()


def plot_parity(targets, predictions, mae, r2, output_path):
    """Generate component-wise parity plots."""
    # Create 3x3 subplot for each tensor component
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    component_names = [
        'C11', 'C12', 'C13',
        'C21', 'C22', 'C23',
        'C31', 'C32', 'C33'
    ]

    # Plot each component separately
    for idx in range(9):
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
    fig.suptitle(f'Tensor Component Parity Plots\nOverall MAE={mae:.4f}, R²={r2:.4f}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved parity plot: {output_path}")


def main():
    args = parse_args()

    # Load config
    config = Config.build(args.cfg, default_data_dir="data_preparation/clip_dataset")
    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", config.training.device)
    config.finalize()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

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
    normalize_tensor = config.tensor_regression.get("normalize_tensor", True)
    full_dataset = GraphTextDataset(
        jsonl_path=str(config.paths.jsonl_path),
        graph_emb_dir=str(config.paths.graph_emb_dir),
        normalize_tensor=normalize_tensor,
    )
    graph_dim = full_dataset.graph_dim
    print(f"Graph embedding dimension: {graph_dim}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Tensor normalization: {normalize_tensor}")

    # Split dataset (same as training)
    if config.training.use_validation:
        _, val_dataset = split_dataset(
            full_dataset,
            val_split=config.training.val_split,
            seed=config.training.val_seed,
        )
        dataset = val_dataset
        print(f"Using validation split: {len(dataset)} samples (val_split={config.training.val_split}, seed={config.training.val_seed})")
    else:
        dataset = full_dataset
        print("No validation split configured, using full dataset")

    # DataLoader
    dataloader = create_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        max_len=config.model.max_seq_length,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        drop_last=False,
    )

    # Load regression checkpoint first
    print(f"\nLoading regression checkpoint from: {args.checkpoint}")
    reg_checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Determine CLIP architecture
    # Priority: 1) regression checkpoint clip_config, 2) CLIP checkpoint inference, 3) CLIP config.json
    clip_cfg = None

    if "clip_config" in reg_checkpoint:
        print("Using CLIP config from regression checkpoint")
        clip_cfg = reg_checkpoint["clip_config"]
    else:
        # Try to infer from CLIP checkpoint or fine-tuned CLIP in regression checkpoint
        print("⚠ No clip_config in regression checkpoint, inferring from CLIP weights...")

        if "clip_model" in reg_checkpoint:
            # Phase 2: infer from fine-tuned CLIP weights
            print("Inferring CLIP architecture from fine-tuned CLIP weights in regression checkpoint...")
            clip_state = reg_checkpoint["clip_model"]
        else:
            # Phase 1: infer from original CLIP checkpoint
            print(f"Loading CLIP checkpoint from: {args.clip_checkpoint}")
            clip_state = torch.load(args.clip_checkpoint, map_location="cpu")

        # Infer architecture from state_dict
        # Check text_encoder.proj.weight shape to determine clip_dim and text hidden_dim
        proj_shape = clip_state["text_encoder.proj.weight"].shape  # [clip_dim, text_hidden_dim]
        clip_dim = proj_shape[0]
        text_hidden_dim = proj_shape[1]

        # Determine backend by checking if backbone exists
        if "text_encoder.backbone.encoder.block.0.layer.0.SelfAttention.q.weight" in clip_state:
            text_backend = "huggingface"
            # For T5, hidden_dim is in d_model
            print(f"Detected HuggingFace backend (T5-like), clip_dim={clip_dim}, text_hidden_dim={text_hidden_dim}")

            # Try to load from CLIP config.json for model_name
            clip_ckpt_path = Path(args.clip_checkpoint)
            clip_config_path = clip_ckpt_path.parent.parent / "config.json"
            if clip_config_path.exists():
                clip_config = Config.from_json(clip_config_path)
                text_model_name = clip_config.model.text_model_name
                text_pooling = clip_config.model.text_pooling
                text_dropout = clip_config.model.text_dropout
            else:
                # Fallback to config passed by user
                text_model_name = config.model.text_model_name
                text_pooling = config.model.text_pooling
                text_dropout = config.model.text_dropout
                print(f"⚠ Using text_model_name from config: {text_model_name}")

            clip_model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=clip_dim,
                text_backend="huggingface",
                text_model_name=text_model_name,
                text_pooling=text_pooling,
                freeze_text_backbone=True,
                train_text_layernorm_only=False,
                text_dropout=text_dropout,
            )
        else:
            # Custom backend
            text_backend = "custom"
            print(f"Detected custom backend, clip_dim={clip_dim}, text_width={text_hidden_dim}")

            # Infer other params from state_dict
            text_width = text_hidden_dim
            # Try to infer layers from state_dict keys
            text_layers = sum(1 for k in clip_state.keys() if k.startswith("text_encoder.blocks."))

            clip_model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=clip_dim,
                text_backend="custom",
                text_width=text_width,
                max_len=config.model.max_seq_length,
                text_layers=text_layers if text_layers > 0 else 6,
                text_heads=8,  # Default
                vocab_size=256,  # Default
                dropout=0.0,
            )

        print(f"✓ Inferred CLIP architecture: {text_backend}, clip_dim={clip_dim}")

    # If we have clip_cfg from regression checkpoint, use it directly
    if clip_cfg is not None:
        if clip_cfg["text_backend"] == "huggingface":
            clip_model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=clip_cfg["clip_dim"],
                text_backend="huggingface",
                text_model_name=clip_cfg["text_model_name"],
                text_pooling=clip_cfg["text_pooling"],
                freeze_text_backbone=True,
                train_text_layernorm_only=False,
                text_dropout=clip_cfg["text_dropout"],
            )
        else:
            clip_model = GraphTextCLIP(
                graph_in_dim=graph_dim,
                clip_dim=clip_cfg["clip_dim"],
                text_backend="custom",
                text_width=clip_cfg["text_width"],
                max_len=clip_cfg["max_seq_length"],
                text_layers=clip_cfg["text_layers"],
                text_heads=clip_cfg["text_heads"],
                vocab_size=clip_cfg["vocab_size"],
                dropout=clip_cfg["dropout"],
            )
        print(f"✓ Using CLIP architecture from config: {clip_cfg['text_backend']}, clip_dim={clip_cfg['clip_dim']}")

    # Load CLIP weights
    if "clip_model" in reg_checkpoint:
        print("Loading fine-tuned CLIP weights from regression checkpoint (Phase 2)")
        clip_model.load_state_dict(reg_checkpoint["clip_model"], strict=True)
    else:
        print(f"Loading frozen CLIP weights from: {args.clip_checkpoint} (Phase 1)")
        clip_checkpoint = torch.load(args.clip_checkpoint, map_location="cpu")
        clip_model.load_state_dict(clip_checkpoint, strict=True)

    print("✓ CLIP model loaded")

    # Create regressor model
    regressor = GraphTensorRegressor(
        clip_model=clip_model,
        tensor_dim=reg_checkpoint["config"]["tensor_dim"],
        head_hidden_dims=reg_checkpoint["config"]["head_hidden_dims"],
        head_dropout=config.tensor_regression.head_dropout,
        freeze_clip=True,  # Always freeze for evaluation
    )

    # Load regression head weights
    regressor.regression_head.load_state_dict(reg_checkpoint["regression_head"])
    print("✓ Regression head loaded")

    # Move to device
    device = torch.device(config.training.device)
    regressor.to(device)
    regressor.eval()

    # Evaluate
    print("\n" + "="*80)
    print("Evaluating on dataset...")
    print("="*80)

    all_preds = []
    all_targets = []
    all_formulas = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            graph = batch["graph"].to(device, non_blocking=True)
            tensor_target = batch["tensor"].to(device, non_blocking=True)

            tensor_pred = regressor(graph)

            all_preds.append(tensor_pred.cpu().numpy())
            all_targets.append(tensor_target.cpu().numpy())
            all_formulas.extend(batch["formula"])
            all_ids.extend(batch["id"])

    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"Total samples evaluated: {len(all_preds)}")

    # Denormalize
    if normalize_tensor and hasattr(full_dataset, 'denormalize_tensor'):
        print("Denormalizing predictions and targets...")
        all_preds_denorm = np.array([
            full_dataset.denormalize_tensor(pred) for pred in all_preds
        ])
        all_targets_denorm = np.array([
            full_dataset.denormalize_tensor(target) for target in all_targets
        ])
    else:
        all_preds_denorm = all_preds
        all_targets_denorm = all_targets

    # Compute metrics
    mae = float(np.abs(all_preds_denorm - all_targets_denorm).mean())
    r2 = float(r2_score(all_targets_denorm.flatten(), all_preds_denorm.flatten()))

    print(f"\nMetrics (Denormalized):")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Save results
    results_path = output_dir / "evaluation_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Evaluation Results\n")
        f.write(f"==================\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"CLIP checkpoint: {args.clip_checkpoint}\n")
        f.write(f"Config: {args.cfg}\n\n")
        f.write(f"Total samples: {len(all_preds)}\n")
        f.write(f"Normalization: {normalize_tensor}\n\n")
        f.write(f"Metrics (Denormalized):\n")
        f.write(f"  MAE:  {mae:.4f}\n")
        f.write(f"  R²:   {r2:.4f}\n")
    print(f"✓ Saved results: {results_path}")

    # Save predictions
    predictions_path = output_dir / "predictions.npz"
    np.savez(
        predictions_path,
        predictions=all_preds_denorm,
        targets=all_targets_denorm,
        formulas=all_formulas,
        ids=all_ids,
        mae=mae,
        r2=r2,
    )
    print(f"✓ Saved predictions: {predictions_path}")

    # Generate parity plot
    plot_path = output_dir / "parity_plot.png"
    plot_parity(all_targets_denorm, all_preds_denorm, mae, r2, plot_path)

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()

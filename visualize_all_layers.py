"""
Visualize text attention across ALL layers for tensor property prediction.

Usage:
    python visualize_all_layers.py \
        --cfg expr_setting/tensor_phase2_scibert_2d_512.json \
        --checkpoint <checkpoint.pt> \
        --sample_idx 0 \
        --output_dir tensor_attention_all_layers
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.config import Config
from src.data.tokenizer import HFTokenizerWrapper
from src.data.dataset import GraphTextDataset
from src.models.clip_model import GraphTextCLIP
from src.models.regression_model import TextTensorRegressor


def extract_text_attention(model, input_ids, attention_mask):
    """Extract attention weights from text encoder."""
    text_encoder = model.clip_model.text_encoder
    if hasattr(text_encoder, 'backbone'):
        outputs = text_encoder.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        return outputs.attentions, text_encoder(input_ids, attention_mask)
    else:
        raise NotImplementedError("Only HuggingFace encoders supported")


def decode_tokens(tokenizer, input_ids):
    """Decode token ids to strings."""
    return tokenizer.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())


def plot_cls_attention_all_layers(attentions, tokens, save_path):
    """Plot CLS token attention for all layers side by side."""
    num_layers = len(attentions)
    valid_len = len(tokens)

    # Compute CLS attention for each layer
    cls_attns = []
    for layer_idx in range(num_layers):
        attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()[:valid_len, :valid_len]
        cls_attn = attn[0, :]  # CLS token attention
        cls_attns.append(cls_attn)

    cls_attns = np.array(cls_attns)  # [num_layers, valid_len]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, 10))

    display_tokens = [tok[:15] + '..' if len(tok) > 15 else tok for tok in tokens]

    sns.heatmap(
        cls_attns,
        xticklabels=display_tokens,
        yticklabels=[f'Layer {i}' for i in range(num_layers)],
        cmap='YlOrRd',
        cbar=True,
        ax=ax,
        vmin=0,
        vmax=np.percentile(cls_attns, 95)
    )

    ax.set_xlabel('Token', fontsize=13)
    ax.set_ylabel('Layer', fontsize=13)
    ax.set_title('[CLS] Token Attention Across All Layers\nWhich tokens does [CLS] attend to at each layer?',
                 fontsize=15, fontweight='bold')

    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_token_importance_all_layers(attentions, tokens, save_path, top_k=20):
    """Plot token importance evolution across all layers."""
    num_layers = len(attentions)
    valid_len = len(tokens)

    # Compute incoming attention for each layer
    importance_by_layer = []
    for attn_layer in attentions:
        attn = attn_layer[0].mean(dim=0).cpu().numpy()[:valid_len, :valid_len]
        incoming = attn.sum(axis=0)
        incoming = incoming / incoming.sum()
        importance_by_layer.append(incoming)

    importance_matrix = np.array(importance_by_layer)  # [num_layers, valid_len]

    # Find top-k most important tokens overall
    avg_importance = importance_matrix.mean(axis=0)
    top_k_idx = np.argsort(avg_importance)[-top_k:][::-1]

    # Filter to top-k tokens
    importance_matrix = importance_matrix[:, top_k_idx]
    top_tokens = [tokens[i] for i in top_k_idx]
    display_tokens = [tok[:20] + '..' if len(tok) > 20 else tok for tok in top_tokens]

    # Plot
    fig, ax = plt.subplots(figsize=(18, 10))

    sns.heatmap(
        importance_matrix.T,
        xticklabels=[f'L{i}' for i in range(num_layers)],
        yticklabels=display_tokens,
        cmap='YlOrRd',
        cbar=True,
        ax=ax,
        vmin=0,
        vmax=np.percentile(importance_matrix, 95)
    )

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Token', fontsize=12)
    ax.set_title(f'Token Importance Evolution Across All Layers (Top {top_k} Tokens)',
                 fontsize=14, fontweight='bold')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_attention_flow(attentions, tokens, save_path):
    """Plot attention flow summary across layers."""
    num_layers = len(attentions)
    valid_len = len(tokens)

    fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 8))
    if num_layers == 1:
        axes = [axes]

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()[:valid_len, :valid_len]

        sns.heatmap(
            attn,
            xticklabels=tokens if layer_idx == 0 else False,
            yticklabels=tokens if layer_idx == 0 else False,
            cmap='viridis',
            cbar=True,
            square=True,
            ax=ax,
            vmin=0,
            vmax=np.percentile(attn, 95)
        )

        ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')

        if layer_idx == 0:
            ax.set_xlabel('Key', fontsize=9)
            ax.set_ylabel('Query', fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=6)
            plt.setp(ax.get_yticklabels(), fontsize=6)

    fig.suptitle('Attention Patterns Across All Layers', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_cls_attention_individual_layers(attentions, tokens, output_dir):
    """Generate individual CLS attention plots for each layer."""
    num_layers = len(attentions)
    valid_len = len(tokens)

    for layer_idx in range(num_layers):
        attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()[:valid_len, :valid_len]
        cls_attn = attn[0, :]

        fig, ax = plt.subplots(figsize=(16, 6))
        display_tokens = [tok[:20] + '..' if len(tok) > 20 else tok for tok in tokens]

        colors = plt.cm.RdYlGn(cls_attn / cls_attn.max())
        bars = ax.bar(range(valid_len), cls_attn, color=colors, edgecolor='black', linewidth=0.5)

        ax.set_xticks(range(valid_len))
        ax.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=9)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title(f'[CLS] Token Attention - Layer {layer_idx}',
                     fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Highlight top 10
        top10_idx = np.argsort(cls_attn)[-10:]
        for idx in top10_idx:
            bars[idx].set_edgecolor('darkred')
            bars[idx].set_linewidth(2)
            ax.text(idx, cls_attn[idx] + 0.002, f'{cls_attn[idx]:.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

        plt.tight_layout()
        save_path = output_dir / f"layer_{layer_idx}_cls_attention.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Saved individual layer plots for {num_layers} layers")


def main():
    parser = argparse.ArgumentParser(description="Visualize all layers attention")
    parser.add_argument("--cfg", type=str, required=True, help="Config JSON path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index")
    parser.add_argument("--output_dir", type=str, default="tensor_attention_all_layers",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--individual_layers", action="store_true",
                       help="Generate individual plots for each layer")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"Loading config from {args.cfg}")
    config = Config.build(args.cfg, default_data_dir="data_preparation/clip_dataset")
    config.finalize()

    # Tokenizer
    tokenizer = HFTokenizerWrapper(
        model_name=config.model.text_model_name,
        max_len=config.model.max_seq_length,
        padding="max_length",
    )
    print(f"✓ Tokenizer: {config.model.text_model_name}")

    # Dataset
    tensor_mode = config.tensor_regression.get("tensor_mode", "full")
    normalize_tensor = config.tensor_regression.get("normalize_tensor", True)
    dataset = GraphTextDataset(
        jsonl_path=str(config.paths.jsonl_path),
        graph_emb_dir=str(config.paths.graph_emb_dir),
        normalize_tensor=normalize_tensor,
        tensor_mode=tensor_mode,
    )
    print(f"✓ Dataset: {len(dataset)} samples")

    # Load CLIP
    clip_checkpoint_path = config.tensor_regression.clip_checkpoint_path
    print(f"\nLoading CLIP from: {clip_checkpoint_path}")

    clip_ckpt_path = Path(clip_checkpoint_path)
    clip_config_path = clip_ckpt_path.parent.parent / "config.json"

    if clip_config_path.exists():
        clip_config = Config.from_json(clip_config_path)
        clip_model = GraphTextCLIP(
            graph_in_dim=dataset.graph_dim,
            clip_dim=clip_config.model.clip_dim,
            text_backend="huggingface",
            text_model_name=clip_config.model.text_model_name,
            text_pooling=clip_config.model.text_pooling,
            freeze_text_backbone=True,
            text_dropout=getattr(clip_config.model, "text_dropout", 0.1),
        )
    else:
        clip_model = GraphTextCLIP(
            graph_in_dim=dataset.graph_dim,
            clip_dim=config.model.clip_dim,
            text_backend="huggingface",
            text_model_name=config.model.text_model_name,
            text_pooling=config.model.text_pooling,
            freeze_text_backbone=True,
            text_dropout=getattr(config.model, "text_dropout", 0.1),
        )

    clip_checkpoint = torch.load(clip_checkpoint_path, map_location="cpu")
    clip_model.load_state_dict(clip_checkpoint, strict=True)
    print("✓ CLIP loaded")

    # Load checkpoint and infer tensor_mode
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if isinstance(checkpoint, dict) and "regression_head" in checkpoint:
        for key, value in checkpoint["regression_head"].items():
            if key.endswith(".weight") and len(value.shape) == 2:
                output_dim = value.shape[0]

        if output_dim == 4:
            tensor_mode = "2d"
            print(f"Detected tensor_mode='2d' (output_dim={output_dim})")
        elif output_dim == 9:
            tensor_mode = "full"
            print(f"Detected tensor_mode='full' (output_dim={output_dim})")

    # Build model
    model = TextTensorRegressor(
        clip_model=clip_model,
        tensor_dim=config.tensor_regression.tensor_dim,
        head_hidden_dims=config.tensor_regression.head_hidden_dims,
        head_dropout=config.tensor_regression.head_dropout,
        freeze_clip=False,
        tensor_mode=tensor_mode,
    )

    # Load weights
    if isinstance(checkpoint, dict) and "regression_head" in checkpoint:
        model.clip_model.load_state_dict(checkpoint["clip_model"], strict=True)
        model.regression_head.load_state_dict(checkpoint["regression_head"], strict=True)
        print("✓ Checkpoint loaded")

    model.eval()
    model.to(args.device)

    # Get sample
    sample = dataset[args.sample_idx]
    text = sample["text"]
    tensor_true = sample.get("tensor", None)

    print(f"\n{'='*80}")
    print(f"Sample {args.sample_idx}")
    print(f"{'='*80}")
    print(f"Text: {text[:150]}..." if len(text) > 150 else f"Text: {text}")
    if tensor_true is not None:
        print(f"True Tensor: {tensor_true}")

    # Tokenize
    input_ids, attention_mask = tokenizer.encode(text)
    input_ids = input_ids.unsqueeze(0).to(args.device)
    attention_mask = attention_mask.unsqueeze(0).to(args.device)

    tokens = decode_tokens(tokenizer, input_ids)
    valid_mask = attention_mask[0].cpu().bool()
    tokens = [tok for tok, valid in zip(tokens, valid_mask) if valid]

    print(f"\nTokens ({len(tokens)}): {tokens[:15]}..." if len(tokens) > 15 else f"\nTokens: {tokens}")

    # Predict
    with torch.no_grad():
        tensor_pred = model(input_ids, attention_mask)
        print(f"\nPredicted Tensor: {tensor_pred[0].cpu().numpy()}")

    # Extract attention
    print("\nExtracting attention from all layers...")
    with torch.no_grad():
        attentions, text_emb = extract_text_attention(model, input_ids, attention_mask)

    print(f"✓ Extracted attention from {len(attentions)} layers")
    print(f"  Shape per layer: {attentions[0].shape}")

    # Generate visualizations
    print(f"\n{'='*80}")
    print("Generating Visualizations for All Layers")
    print(f"{'='*80}\n")

    print("1. CLS attention across all layers (heatmap)...")
    plot_cls_attention_all_layers(
        attentions, tokens,
        output_dir / f"sample{args.sample_idx}_cls_attention_all_layers.png"
    )

    print("2. Token importance evolution...")
    plot_token_importance_all_layers(
        attentions, tokens,
        output_dir / f"sample{args.sample_idx}_token_importance_all_layers.png"
    )

    print("3. Attention flow across layers...")
    plot_attention_flow(
        attentions, tokens,
        output_dir / f"sample{args.sample_idx}_attention_flow.png"
    )

    if args.individual_layers:
        print("4. Individual layer CLS attention plots...")
        plot_cls_attention_individual_layers(attentions, tokens, output_dir)

    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

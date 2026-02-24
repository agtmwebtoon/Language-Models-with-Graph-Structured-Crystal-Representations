"""
Visualize text attention for tensor property prediction.

This script analyzes which text tokens (from Robocrystallographer descriptions)
have the most influence on tensor property predictions.

Usage:
    python visualize_tensor_text_attention.py \
        --cfg expr_setting/tensor_phase2_scibert_2d_512.json \
        --checkpoint path/to/checkpoint.pt \
        --output_dir tensor_attention_viz
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.config import Config
from src.data.tokenizer import ByteLevelTokenizer, HFTokenizerWrapper
from src.data.dataset import GraphTextDataset, create_dataloader
from src.models.clip_model import GraphTextCLIP
from src.models.regression_model import TextTensorRegressor


def extract_text_attention_weights(model, input_ids, attention_mask):
    """
    Extract attention weights from the text encoder in TextTensorRegressor.

    Args:
        model: TextTensorRegressor model
        input_ids: [1, L] token ids
        attention_mask: [1, L] attention mask

    Returns:
        attentions: List of attention tensors from each layer
        text_embedding: [1, clip_dim] text embedding
    """
    text_encoder = model.clip_model.text_encoder

    # For HuggingFace models, we need to enable output_attentions
    if hasattr(text_encoder, 'backbone'):
        backbone = text_encoder.backbone

        # Forward pass with attention output
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

        # Get attentions: tuple of [B, num_heads, L, L] for each layer
        attentions = outputs.attentions

        # Get pooled text embedding
        text_embedding = text_encoder(input_ids, attention_mask)

        return attentions, text_embedding
    else:
        raise NotImplementedError("Attention extraction only implemented for HuggingFace encoders")


def decode_tokens(tokenizer, input_ids):
    """Decode token ids back to strings."""
    if hasattr(tokenizer, 'tokenizer'):
        # HuggingFace tokenizer
        tokens = tokenizer.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        return tokens
    else:
        # ByteLevel tokenizer
        return [chr(idx) if idx < 256 else f"<{idx}>" for idx in input_ids[0].cpu().tolist()]


def plot_attention_heatmap(
    attentions,
    tokens,
    layer_idx=-1,
    head_idx=0,
    save_path=None,
    figsize=(14, 12)
):
    """
    Plot attention heatmap for a specific layer and head.
    """
    # Select attention from specified layer and head
    attn = attentions[layer_idx][0, head_idx].cpu().numpy()  # [L, L]

    # Filter out padding tokens
    valid_len = len(tokens)
    attn = attn[:valid_len, :valid_len]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar=True,
        square=True,
        ax=ax,
        vmin=0,
        vmax=1
    )

    ax.set_xlabel('Key Token (Attended To)', fontsize=12)
    ax.set_ylabel('Query Token (Attending From)', fontsize=12)
    ax.set_title(f'Text Attention for Tensor Prediction\nLayer {layer_idx}, Head {head_idx}',
                 fontsize=14, fontweight='bold')

    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_rollout(
    attentions,
    tokens,
    save_path=None,
    figsize=(14, 12)
):
    """
    Plot attention rollout (cumulative attention across all layers).
    This shows the aggregated information flow through the entire network.
    """
    valid_len = len(tokens)

    # Average attention across heads for each layer
    attn_matrices = []
    for attn_layer in attentions:
        # [B, num_heads, L, L] -> [L, L]
        attn_avg = attn_layer[0].mean(dim=0).cpu().numpy()[:valid_len, :valid_len]
        attn_matrices.append(attn_avg)

    # Compute attention rollout by matrix multiplication
    rollout = np.eye(valid_len)
    for attn_matrix in attn_matrices:
        # Add residual connection
        attn_matrix = attn_matrix + np.eye(valid_len)
        # Normalize
        attn_matrix = attn_matrix / attn_matrix.sum(axis=-1, keepdims=True)
        # Accumulate
        rollout = np.matmul(attn_matrix, rollout)

    # Normalize
    rollout = rollout / rollout.sum(axis=-1, keepdims=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        rollout,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar=True,
        square=True,
        ax=ax,
        vmin=0,
        vmax=np.percentile(rollout, 95)
    )

    ax.set_xlabel('Source Token', fontsize=12)
    ax.set_ylabel('Target Token', fontsize=12)
    ax.set_title('Attention Rollout for Tensor Prediction\n(Cumulative Attention Flow)',
                 fontsize=14, fontweight='bold')

    plt.xticks(rotation=90, ha='center', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention rollout to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_token_importance(
    attentions,
    tokens,
    layer_idx=-1,
    save_path=None,
    figsize=(16, 8)
):
    """
    Plot token importance for tensor prediction.
    Shows which tokens receive the most attention (important for prediction).
    """
    valid_len = len(tokens)

    # Average attention across all heads in the layer
    attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()  # [L, L]
    attn = attn[:valid_len, :valid_len]

    # Compute importance scores
    # Incoming attention: how much other tokens attend to this token
    incoming_attn = attn.sum(axis=0)
    # Outgoing attention: how much this token attends to others
    outgoing_attn = attn.sum(axis=1)

    # Normalize
    incoming_attn = incoming_attn / incoming_attn.sum()
    outgoing_attn = outgoing_attn / outgoing_attn.sum()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Truncate long tokens for display
    display_tokens = [tok[:20] + '..' if len(tok) > 20 else tok for tok in tokens]

    # Plot incoming attention (token importance)
    colors1 = plt.cm.Reds(incoming_attn / incoming_attn.max())
    bars1 = ax1.bar(range(valid_len), incoming_attn, color=colors1,
                    edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(valid_len))
    ax1.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=9)
    ax1.set_ylabel('Attention Weight', fontsize=11)
    ax1.set_title('Token Importance (Incoming Attention)\nWhich tokens are most attended to?',
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight top 5 tokens
    top5_idx = np.argsort(incoming_attn)[-5:]
    for idx in top5_idx:
        bars1[idx].set_edgecolor('darkred')
        bars1[idx].set_linewidth(2)

    # Plot outgoing attention
    colors2 = plt.cm.Blues(outgoing_attn / outgoing_attn.max())
    bars2 = ax2.bar(range(valid_len), outgoing_attn, color=colors2,
                    edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(valid_len))
    ax2.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=9)
    ax2.set_ylabel('Attention Weight', fontsize=11)
    ax2.set_title('Query Strength (Outgoing Attention)\nWhich tokens attend most to others?',
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight top 5 tokens
    top5_idx = np.argsort(outgoing_attn)[-5:]
    for idx in top5_idx:
        bars2[idx].set_edgecolor('darkblue')
        bars2[idx].set_linewidth(2)

    fig.suptitle(f'Token Importance Analysis for Tensor Prediction (Layer {layer_idx})',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved token importance plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_cls_token_attention(
    attentions,
    tokens,
    layer_idx=-1,
    save_path=None,
    figsize=(16, 6)
):
    """
    Plot CLS token attention to all other tokens.
    For models using CLS pooling, this shows which tokens contribute most to the final embedding.
    """
    valid_len = len(tokens)

    # Average attention across all heads in the layer
    attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()  # [L, L]
    attn = attn[:valid_len, :valid_len]

    # Get CLS token attention (first token)
    cls_attn = attn[0, :]  # How much CLS attends to each token

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Truncate long tokens for display
    display_tokens = [tok[:20] + '..' if len(tok) > 20 else tok for tok in tokens]

    # Color by attention weight
    colors = plt.cm.RdYlGn(cls_attn / cls_attn.max())
    bars = ax.bar(range(valid_len), cls_attn, color=colors,
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(valid_len))
    ax.set_xticklabels(display_tokens, rotation=90, ha='center', fontsize=9)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(f'[CLS] Token Attention (Layer {layer_idx})\nWhich tokens contribute most to tensor prediction?',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight top 10 tokens
    top10_idx = np.argsort(cls_attn)[-10:]
    for idx in top10_idx:
        bars[idx].set_edgecolor('darkred')
        bars[idx].set_linewidth(2)
        # Add value label
        ax.text(idx, cls_attn[idx] + 0.002, f'{cls_attn[idx]:.3f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved CLS attention plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_layer_wise_token_importance(
    attentions,
    tokens,
    save_path=None,
    figsize=(18, 10),
    top_k=15
):
    """
    Plot token importance across all layers to see how importance evolves.
    """
    valid_len = len(tokens)
    num_layers = len(attentions)

    # Compute incoming attention for each layer
    importance_by_layer = []
    for layer_idx, attn_layer in enumerate(attentions):
        attn = attn_layer[0].mean(dim=0).cpu().numpy()[:valid_len, :valid_len]
        incoming = attn.sum(axis=0)
        incoming = incoming / incoming.sum()
        importance_by_layer.append(incoming)

    importance_matrix = np.stack(importance_by_layer, axis=0)  # [num_layers, valid_len]

    # Find top-k most important tokens overall
    avg_importance = importance_matrix.mean(axis=0)
    top_k_idx = np.argsort(avg_importance)[-top_k:][::-1]

    # Filter to top-k tokens
    importance_matrix = importance_matrix[:, top_k_idx]
    top_tokens = [tokens[i] for i in top_k_idx]
    display_tokens = [tok[:20] + '..' if len(tok) > 20 else tok for tok in top_tokens]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

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
    ax.set_title(f'Token Importance Evolution Across Layers (Top {top_k} Tokens)\nFor Tensor Property Prediction',
                 fontsize=14, fontweight='bold')

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved layer-wise importance plot to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize text attention for tensor prediction")
    parser.add_argument("--cfg", type=str, required=True,
                       help="Config JSON path (e.g., expr_setting/tensor_phase2_scibert_2d_512.json)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Trained TextTensorRegressor checkpoint path")
    parser.add_argument("--output_dir", type=str, default="tensor_attention_viz",
                       help="Output directory for visualizations")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Sample index from validation set to visualize")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"Loading config from {args.cfg}")
    config = Config.build(args.cfg, default_data_dir="data_preparation/clip_dataset")
    config.finalize()

    # Setup tokenizer
    if config.model.text_backend == "huggingface":
        tokenizer = HFTokenizerWrapper(
            model_name=config.model.text_model_name,
            max_len=config.model.max_seq_length,
            padding="max_length",
        )
        print(f"✓ Tokenizer: HuggingFace ({config.model.text_model_name})")
    else:
        raise NotImplementedError("Only HuggingFace tokenizer supported for this visualization")

    # Load dataset
    tensor_mode = config.tensor_regression.get("tensor_mode", "full")
    normalize_tensor = config.tensor_regression.get("normalize_tensor", True)

    dataset = GraphTextDataset(
        jsonl_path=str(config.paths.jsonl_path),
        graph_emb_dir=str(config.paths.graph_emb_dir),
        normalize_tensor=normalize_tensor,
        tensor_mode=tensor_mode,
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Load CLIP model first (to get architecture)
    clip_checkpoint_path = config.tensor_regression.clip_checkpoint_path
    print(f"\nLoading CLIP model from: {clip_checkpoint_path}")

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
        # Use current config
        clip_model = GraphTextCLIP(
            graph_in_dim=dataset.graph_dim,
            clip_dim=config.model.clip_dim,
            text_backend="huggingface",
            text_model_name=config.model.text_model_name,
            text_pooling=config.model.text_pooling,
            freeze_text_backbone=True,
            text_dropout=getattr(config.model, "text_dropout", 0.1),
        )

    # Load CLIP weights
    clip_checkpoint = torch.load(clip_checkpoint_path, map_location="cpu")
    clip_model.load_state_dict(clip_checkpoint, strict=True)
    print("✓ CLIP model loaded")

    # Load regression checkpoint first to get architecture info
    print(f"\nLoading regression checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Infer tensor_mode from checkpoint
    if isinstance(checkpoint, dict) and "regression_head" in checkpoint:
        # Find output dimension from last layer
        for key, value in checkpoint["regression_head"].items():
            if key.endswith(".weight") and len(value.shape) == 2:
                output_dim = value.shape[0]

        if output_dim == 4:
            tensor_mode = "2d"
            print(f"Detected tensor_mode='2d' from checkpoint (output_dim={output_dim})")
        elif output_dim == 9:
            tensor_mode = "full"
            print(f"Detected tensor_mode='full' from checkpoint (output_dim={output_dim})")
        else:
            print(f"Warning: Unknown output_dim={output_dim}, using config tensor_mode={tensor_mode}")

    # Build TextTensorRegressor with correct tensor_mode
    from src.models.regression_model import TextTensorRegressor

    model = TextTensorRegressor(
        clip_model=clip_model,
        tensor_dim=config.tensor_regression.tensor_dim,  # Will be overridden by tensor_mode
        head_hidden_dims=config.tensor_regression.head_hidden_dims,
        head_dropout=config.tensor_regression.head_dropout,
        freeze_clip=False,  # We want to analyze attention
        tensor_mode=tensor_mode,
    )

    print(f"Model created with tensor_mode='{tensor_mode}', output_dim={model.regression_head.output_dim}")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "regression_head" in checkpoint:
            # Nested format: separate keys for clip_model and regression_head
            model.clip_model.load_state_dict(checkpoint["clip_model"], strict=True)
            model.regression_head.load_state_dict(checkpoint["regression_head"], strict=True)
            print("✓ Checkpoint loaded (nested format)")
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            print("✓ Checkpoint loaded (model_state_dict)")
        else:
            model.load_state_dict(checkpoint, strict=True)
            print("✓ Checkpoint loaded")
    else:
        model.load_state_dict(checkpoint, strict=True)
        print("✓ Checkpoint loaded")

    model.eval()
    model.to(args.device)
    print("✓ Model loaded successfully")

    # Get sample
    sample = dataset[args.sample_idx]
    text = sample["text"]
    tensor_true = sample.get("tensor", None)

    print(f"\n{'='*80}")
    print(f"Sample {args.sample_idx}")
    print(f"{'='*80}")
    print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
    if tensor_true is not None:
        print(f"True Tensor: {tensor_true}")
    print(f"{'='*80}\n")

    # Tokenize
    input_ids, attention_mask = tokenizer.encode(text)
    input_ids = input_ids.unsqueeze(0).to(args.device)
    attention_mask = attention_mask.unsqueeze(0).to(args.device)

    # Decode tokens
    tokens = decode_tokens(tokenizer, input_ids)
    valid_mask = attention_mask[0].cpu().bool()
    tokens = [tok for tok, valid in zip(tokens, valid_mask) if valid]

    print(f"Tokens ({len(tokens)}): {tokens}\n")

    # Forward pass and get prediction
    with torch.no_grad():
        tensor_pred = model(input_ids, attention_mask)
        print(f"Predicted Tensor: {tensor_pred[0].cpu().numpy()}")

    # Extract attention weights
    print("\nExtracting attention weights...")
    with torch.no_grad():
        attentions, text_emb = extract_text_attention_weights(model, input_ids, attention_mask)

    print(f"✓ Extracted attention from {len(attentions)} layers")
    print(f"  Attention shape: {attentions[0].shape} (batch, heads, seq_len, seq_len)")
    print(f"  Text embedding shape: {text_emb.shape}")

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80 + "\n")

    # 1. Attention heatmap (last layer, head 0)
    print("1. Generating attention heatmap (last layer, head 0)...")
    plot_attention_heatmap(
        attentions,
        tokens,
        layer_idx=-1,
        head_idx=0,
        save_path=output_dir / f"sample{args.sample_idx}_attention_heatmap.png"
    )

    # 2. Attention rollout
    print("2. Generating attention rollout...")
    plot_attention_rollout(
        attentions,
        tokens,
        save_path=output_dir / f"sample{args.sample_idx}_attention_rollout.png"
    )

    # 3. Token importance
    print("3. Generating token importance analysis...")
    plot_token_importance(
        attentions,
        tokens,
        layer_idx=-1,
        save_path=output_dir / f"sample{args.sample_idx}_token_importance.png"
    )

    # 4. CLS token attention (if using first token pooling)
    print("4. Generating CLS token attention...")
    plot_cls_token_attention(
        attentions,
        tokens,
        layer_idx=-1,
        save_path=output_dir / f"sample{args.sample_idx}_cls_attention.png"
    )

    # 5. Layer-wise token importance
    print("5. Generating layer-wise token importance...")
    plot_layer_wise_token_importance(
        attentions,
        tokens,
        save_path=output_dir / f"sample{args.sample_idx}_layerwise_importance.png"
    )

    print(f"\n{'='*80}")
    print(f"✓ All visualizations saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

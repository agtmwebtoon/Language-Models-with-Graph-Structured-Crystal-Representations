"""
Visualize attention distributions for tokenized text in the trained model.

Usage:
    python visualize_text_attention.py --cfg expr_setting/tensor_phase2.json \
        --checkpoint data_preparation/clip_dataset/runs/xxx/checkpoints/best_model.pt \
        --text "MoS2 is a two-dimensional semiconductor" \
        --output attention_viz.png
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.config import Config
from src.data.tokenizer import ByteLevelTokenizer, HFTokenizerWrapper
from src.models.clip_model import GraphTextCLIP


def extract_attention_weights(model, input_ids, attention_mask):
    """
    Extract attention weights from the text encoder.

    Args:
        model: GraphTextCLIP model
        input_ids: [1, L] token ids
        attention_mask: [1, L] attention mask

    Returns:
        attentions: List of attention tensors from each layer
        tokens: List of token strings
    """
    text_encoder = model.text_encoder

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

        return attentions
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
    figsize=(12, 10)
):
    """
    Plot attention heatmap for a specific layer and head.

    Args:
        attentions: List of attention tensors [B, num_heads, L, L]
        tokens: List of token strings
        layer_idx: Which layer to visualize (-1 for last layer)
        head_idx: Which attention head to visualize
        save_path: Path to save the figure
        figsize: Figure size
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
        ax=ax
    )

    ax.set_xlabel('Key (to)', fontsize=12)
    ax.set_ylabel('Query (from)', fontsize=12)
    ax.set_title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}', fontsize=14)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_heads(
    attentions,
    tokens,
    layer_idx=-1,
    save_path=None,
    max_heads=8,
    figsize=(20, 16)
):
    """
    Plot attention heatmaps for multiple heads in a layer.

    Args:
        attentions: List of attention tensors [B, num_heads, L, L]
        tokens: List of token strings
        layer_idx: Which layer to visualize (-1 for last layer)
        save_path: Path to save the figure
        max_heads: Maximum number of heads to plot
        figsize: Figure size
    """
    attn_layer = attentions[layer_idx][0].cpu().numpy()  # [num_heads, L, L]
    num_heads = min(attn_layer.shape[0], max_heads)

    # Filter out padding tokens
    valid_len = len(tokens)

    # Create subplots
    rows = int(np.sqrt(num_heads))
    cols = int(np.ceil(num_heads / rows))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_heads > 1 else [axes]

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        attn_head = attn_layer[head_idx, :valid_len, :valid_len]

        sns.heatmap(
            attn_head,
            xticklabels=tokens if head_idx >= num_heads - cols else [],
            yticklabels=tokens if head_idx % cols == 0 else [],
            cmap='viridis',
            cbar=True,
            square=True,
            ax=ax
        )

        ax.set_title(f'Head {head_idx}', fontsize=10)

        if head_idx >= num_heads - cols:
            ax.set_xlabel('Key', fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        if head_idx % cols == 0:
            ax.set_ylabel('Query', fontsize=9)
            plt.setp(ax.get_yticklabels(), fontsize=7)

    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Attention Heads - Layer {layer_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved multi-head attention heatmap to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_rollout(
    attentions,
    tokens,
    save_path=None,
    figsize=(12, 10)
):
    """
    Plot attention rollout (cumulative attention across all layers).

    Attention rollout computes the cumulative attention flow from input to output
    by multiplying attention matrices across layers.

    Args:
        attentions: List of attention tensors [B, num_heads, L, L]
        tokens: List of token strings
        save_path: Path to save the figure
        figsize: Figure size
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
        ax=ax
    )

    ax.set_xlabel('Token (source)', fontsize=12)
    ax.set_ylabel('Token (destination)', fontsize=12)
    ax.set_title('Attention Rollout (Cumulative Attention Flow)', fontsize=14)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention rollout to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_text_overlay(
    attentions,
    tokens,
    layer_idx=-1,
    save_path=None,
    figsize=(14, 8)
):
    """
    Plot attention weights overlaid on text tokens.
    Shows which tokens each token attends to using colored highlights.

    Args:
        attentions: List of attention tensors [B, num_heads, L, L]
        tokens: List of token strings
        layer_idx: Which layer to visualize (-1 for last layer)
        save_path: Path to save the figure
        figsize: Figure size
    """
    valid_len = len(tokens)

    # Average attention across all heads in the layer
    attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()  # [L, L]
    attn = attn[:valid_len, :valid_len]

    # Create figure with subplots for each query token
    num_tokens = len(tokens)
    fig, axes = plt.subplots(num_tokens, 1, figsize=(figsize[0], figsize[1] * num_tokens / 8))

    if num_tokens == 1:
        axes = [axes]

    for query_idx in range(num_tokens):
        ax = axes[query_idx]
        ax.set_xlim(0, num_tokens)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Get attention weights for this query token
        attn_weights = attn[query_idx]

        # Plot each token with color based on attention weight
        for key_idx in range(num_tokens):
            weight = attn_weights[key_idx]
            color = plt.cm.YlOrRd(weight)  # Yellow to Red colormap

            # Draw colored box
            rect = plt.Rectangle(
                (key_idx, 0.1), 1, 0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5
            )
            ax.add_patch(rect)

            # Add token text
            token_text = tokens[key_idx]
            # Truncate long tokens
            if len(token_text) > 10:
                token_text = token_text[:8] + '..'

            ax.text(
                key_idx + 0.5, 0.5,
                token_text,
                ha='center', va='center',
                fontsize=9,
                fontweight='bold' if key_idx == query_idx else 'normal',
                color='white' if weight > 0.5 else 'black'
            )

            # Add attention weight on top
            ax.text(
                key_idx + 0.5, 0.95,
                f'{weight:.2f}',
                ha='center', va='top',
                fontsize=7,
                color='darkblue'
            )

        # Label the query token
        query_token = tokens[query_idx]
        if len(query_token) > 15:
            query_token = query_token[:13] + '..'
        ax.text(
            -0.5, 0.5,
            f'"{query_token}"',
            ha='right', va='center',
            fontsize=10,
            fontweight='bold'
        )

    fig.suptitle(
        f'Attention Distribution per Token (Layer {layer_idx}, Averaged across Heads)',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.02, aspect=40)
    cbar.set_label('Attention Weight', fontsize=11)

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved text overlay visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_attention_word_importance(
    attentions,
    tokens,
    layer_idx=-1,
    save_path=None,
    figsize=(14, 6)
):
    """
    Plot aggregated attention weights as word importance scores.
    Shows a bar chart of how much attention each token receives overall.

    Args:
        attentions: List of attention tensors [B, num_heads, L, L]
        tokens: List of token strings
        layer_idx: Which layer to visualize (-1 for last layer)
        save_path: Path to save the figure
        figsize: Figure size
    """
    valid_len = len(tokens)

    # Average attention across all heads in the layer
    attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()  # [L, L]
    attn = attn[:valid_len, :valid_len]

    # Aggregate attention: sum of incoming attention for each token
    incoming_attn = attn.sum(axis=0)  # How much attention each token receives
    outgoing_attn = attn.sum(axis=1)  # How much attention each token gives

    # Normalize
    incoming_attn = incoming_attn / incoming_attn.sum()
    outgoing_attn = outgoing_attn / outgoing_attn.sum()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Truncate long tokens for display
    display_tokens = [tok[:15] + '..' if len(tok) > 15 else tok for tok in tokens]

    # Plot incoming attention (how important is each token as a key)
    colors1 = plt.cm.YlOrRd(incoming_attn / incoming_attn.max())
    bars1 = ax1.bar(range(valid_len), incoming_attn, color=colors1, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(range(valid_len))
    ax1.set_xticklabels(display_tokens, rotation=45, ha='right')
    ax1.set_ylabel('Attention Weight', fontsize=11)
    ax1.set_title('Incoming Attention (Token Importance as Key)', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for idx, (bar, val) in enumerate(zip(bars1, incoming_attn)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.3f}',
            ha='center', va='bottom',
            fontsize=8
        )

    # Plot outgoing attention (how much attention each token gives)
    colors2 = plt.cm.YlGnBu(outgoing_attn / outgoing_attn.max())
    bars2 = ax2.bar(range(valid_len), outgoing_attn, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(valid_len))
    ax2.set_xticklabels(display_tokens, rotation=45, ha='right')
    ax2.set_ylabel('Attention Weight', fontsize=11)
    ax2.set_title('Outgoing Attention (Token Importance as Query)', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for idx, (bar, val) in enumerate(zip(bars2, outgoing_attn)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.3f}',
            ha='center', va='bottom',
            fontsize=8
        )

    fig.suptitle(
        f'Token Importance Analysis (Layer {layer_idx}, Averaged across Heads)',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved word importance visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize text attention")
    parser.add_argument("--cfg", type=str, required=True, help="Config JSON path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--text", type=str, required=True, help="Text to visualize")
    parser.add_argument("--output_dir", type=str, default="attention_viz", help="Output directory")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index (-1 for last)")
    parser.add_argument("--head", type=int, default=0, help="Attention head index")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = Config.build(args.cfg, default_data_dir="data_preparation/clip_dataset")
    config.finalize()

    # Setup tokenizer
    if config.model.text_backend == "huggingface":
        tokenizer = HFTokenizerWrapper(
            model_name=config.model.text_model_name,
            max_len=config.model.max_seq_length,
            padding="max_length",
        )
    else:
        tokenizer = ByteLevelTokenizer(
            pad_token=config.tokenizer.pad_token,
            sot_token=config.tokenizer.sot_token,
            eot_token=config.tokenizer.eot_token,
        )

    # Load model checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Build model with same architecture
    # Note: You may need to adjust graph_in_dim based on your dataset
    graph_in_dim = 768  # Default, adjust if needed

    if config.model.text_backend == "huggingface":
        model = GraphTextCLIP(
            graph_in_dim=graph_in_dim,
            clip_dim=config.model.clip_dim,
            text_backend="huggingface",
            text_model_name=config.model.text_model_name,
            text_pooling=config.model.text_pooling,
            freeze_text_backbone=False,
            text_dropout=config.model.text_dropout,
        )
    else:
        model = GraphTextCLIP(
            graph_in_dim=graph_in_dim,
            clip_dim=config.model.clip_dim,
            text_backend="custom",
            text_width=config.model.text_width,
            max_len=config.model.max_seq_length,
            text_layers=config.model.text_layers,
            text_heads=config.model.text_heads,
            vocab_size=config.model.vocab_size,
            dropout=config.model.dropout,
        )

    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    model.to(args.device)

    print(f"✓ Model loaded successfully")

    # Tokenize text
    print(f"\nText: {args.text}")
    input_ids, attention_mask = tokenizer.encode(args.text, config.model.max_seq_length)
    input_ids = input_ids.unsqueeze(0).to(args.device)  # [1, L]
    attention_mask = attention_mask.unsqueeze(0).to(args.device)  # [1, L]

    # Decode tokens for visualization
    tokens = decode_tokens(tokenizer, input_ids)

    # Filter tokens to valid length (remove padding)
    valid_mask = attention_mask[0].cpu().bool()
    tokens = [tok for tok, valid in zip(tokens, valid_mask) if valid]

    print(f"Tokens ({len(tokens)}): {tokens}")

    # Extract attention weights
    print("\nExtracting attention weights...")
    with torch.no_grad():
        attentions = extract_attention_weights(model, input_ids, attention_mask)

    print(f"✓ Extracted attention from {len(attentions)} layers")
    print(f"  Attention shape: {attentions[0].shape} (batch, heads, seq_len, seq_len)")

    # Visualize single head
    print(f"\nGenerating attention heatmap (Layer {args.layer}, Head {args.head})...")
    plot_attention_heatmap(
        attentions,
        tokens,
        layer_idx=args.layer,
        head_idx=args.head,
        save_path=output_dir / f"attention_L{args.layer}_H{args.head}.png"
    )

    # Visualize all heads in a layer
    print(f"\nGenerating multi-head attention heatmap (Layer {args.layer})...")
    plot_attention_heads(
        attentions,
        tokens,
        layer_idx=args.layer,
        save_path=output_dir / f"attention_heads_L{args.layer}.png"
    )

    # Visualize attention rollout
    print("\nGenerating attention rollout...")
    plot_attention_rollout(
        attentions,
        tokens,
        save_path=output_dir / "attention_rollout.png"
    )

    # Visualize text overlay
    print(f"\nGenerating text overlay visualization (Layer {args.layer})...")
    plot_attention_text_overlay(
        attentions,
        tokens,
        layer_idx=args.layer,
        save_path=output_dir / f"attention_text_overlay_L{args.layer}.png"
    )

    # Visualize word importance
    print(f"\nGenerating word importance visualization (Layer {args.layer})...")
    plot_attention_word_importance(
        attentions,
        tokens,
        layer_idx=args.layer,
        save_path=output_dir / f"attention_word_importance_L{args.layer}.png"
    )

    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

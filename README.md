# Material Representation: Graph-Text CLIP

A contrastive learning framework for aligning graph-based structural representations and text descriptions of crystalline materials. This project implements a CLIP-style model that learns a shared embedding space between material structures (encoded as graphs via Equiformer) and their natural language descriptions.

## Overview

This project uses contrastive learning to create a unified representation space where:
- **Graph embeddings** capture the 3D structural information of crystalline materials
- **Text embeddings** encode natural language descriptions of material properties
- The model learns to align these two modalities using symmetric cross-entropy loss

### Key Features

- **Dual Encoder Architecture**: Separate encoders for graph and text modalities
- **Multiple Text Backends**: Support for custom Transformer encoder or pre-trained T5 models
- **Pre-computed Graph Embeddings**: Uses Equiformer V2 for extracting structural features
- **Flexible Configuration**: Easy-to-modify configuration system with dataclasses
- **Modular Design**: Clean separation of concerns (models, data, training, visualization)
- **Type-Safe**: Full type annotations throughout the codebase

## Project Structure

```
material_representation_graph_text/
├── src/
│   ├── models/              # Model architectures
│   │   ├── components.py      # Attention, transformer building blocks
│   │   ├── encoders.py        # GraphEncoder, TextEncoder, T5Encoder
│   │   └── clip_model.py      # GraphTextCLIP main model
│   ├── data/                # Data handling
│   │   ├── tokenizer.py       # ByteLevelTokenizer, HFTokenizerWrapper
│   │   └── dataset.py         # GraphTextDataset, dataloaders
│   ├── training/            # Training framework
│   │   └── trainer.py         # Trainer class with logging & checkpointing
│   ├── utils/               # Utilities
│   │   ├── config.py          # Configuration management
│   │   ├── run_naming.py      # Experiment naming utilities
│   │   └── config_dump.py     # Config serialization
│   └── visualization/       # Visualization tools
│       ├── embedding_extractor.py
│       ├── plots.py
│       └── visualizer.py
├── data_preparation/        # Data preprocessing scripts
│   ├── build_graph_text_dataset.py  # Extract graph embeddings from CIF files
│   └── graph_featurizer.py          # Graph featurization utilities
├── expr_setting/            # Experiment configurations
│   ├── run_spec.json          # Training configuration specs
│   └── visualization_cfg.json # Visualization settings
├── train.py                 # Main training script
├── visualize_embeddings.py  # Embedding visualization script
├── environment.yml          # Conda environment specification
└── README.md               # This file
```

## Architecture

### Graph Encoder
- Takes pre-computed graph embeddings from Equiformer V2
- Projects to shared CLIP embedding space via linear layer
- L2-normalizes outputs

### Text Encoder (Two Options)

**Option 1: Custom Transformer**
- Byte-level tokenization (256 vocab size)
- Learnable positional embeddings
- Multi-head self-attention layers
- Mean pooling over valid tokens

**Option 2: Pre-trained T5**
- Uses HuggingFace `T5EncoderModel`
- Supports mean/first/last token pooling
- Options for freezing or fine-tuning (LayerNorm-only mode available)

### Contrastive Learning
- CLIP-style symmetric cross-entropy loss
- Learnable temperature parameter
- Bidirectional graph↔text retrieval

## Installation

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate equiformer_v2
```

### 2. Dependencies

Key dependencies include:
- PyTorch 1.13.1 (CUDA 11.6)
- PyG (PyTorch Geometric)
- Transformers (for T5 backend)
- ASE (Atomic Simulation Environment)
- E3NN (for Equiformer)

## Usage

### 1. Data Preparation

Extract graph embeddings from CIF files using Equiformer:

```bash
cd data_preparation

python build_graph_text_dataset.py \
    --csv your_material_data.csv \
    --cif_dir path/to/cif/files \
    --out_dir clip_dataset \
    --equiformer_root path/to/equiformer_v2 \
    --checkpoint path/to/equiformer_checkpoint.pt \
    --pooling max \
    --device cuda
```

**Input CSV Format:**

| Column | Description |
|--------|-------------|
| id | Unique material identifier |
| formula | Chemical formula (e.g., "Ni4S5Se3") |
| description | Natural language description of material |
| tensor | Optional: tensor data for supervised tasks |

**Outputs:**
- `clip_dataset/dataset.jsonl` - Dataset records linking embeddings to text
- `clip_dataset/graph_emb/` - Directory of `.npy` files containing graph embeddings
- `clip_dataset/missing_cif.txt` - Log of missing/failed CIF files

### 2. Training

**Using default configuration:**
```bash
python train.py --cfg expr_setting/run_spec.json
```


### 3. Visualization

Visualize embeddings with t-SNE:

```bash
python visualize_embeddings.py --cfg expr_setting/visualization_cfg.json
```

Outputs saved to `data_preparation/clip_dataset/runs/{run_name}/viz/`:
- `tsne_*.png` - t-SNE plots of graph and text embeddings
- `similarity_metrics.json` - Quantitative alignment metrics

## Configuration

The configuration system uses dataclasses for type safety. Key configuration sections:

### ModelConfig
```python
clip_dim: int = 512                    # Shared embedding dimension
text_backend: Literal["custom", "t5"]  # Text encoder type
text_width: int = 512                  # Custom encoder width
text_layers: int = 6                   # Custom encoder depth
text_heads: int = 8                    # Attention heads
max_seq_length: int = 256              # Max text length
vocab_size: int = 256                  # Byte-level vocab
dropout: float = 0.0

# T5-specific
t5_model_name: str = "t5-base"
t5_pooling: Literal["mean", "first", "last"] = "mean"
freeze_t5: bool = True
train_layernorm_only: bool = True
```

### TrainingConfig
```python
batch_size: int = 64
learning_rate: float = 2e-4
weight_decay: float = 0.01
epochs: int = 50
num_workers: int = 4
device: str = "cuda"
log_interval: int = 50
```

### Experiment Naming

Runs are automatically named based on configuration:
```
{text_model_id}_{graph_backbone_id}_{suffix}
```

Examples:
- `t5-base_equiformer_001`
- `customTx_equiformer_002`

## Model Training Details

### Loss Function
Symmetric CLIP loss:
```python
loss = (CE(graph→text) + CE(text→graph)) / 2
```

### Metrics Tracked
- Contrastive loss
- Graph-to-text retrieval accuracy (Top-1)
- Text-to-graph retrieval accuracy (Top-1)
- Cosine similarity statistics

### Checkpointing
Models saved to `data_preparation/clip_dataset/runs/{run_name}/checkpoints/graph_text_clip.pt`

## Advanced Usage

### Custom Text Encoder

To use a custom transformer instead of T5:

```python
config.model.text_backend = "custom"
config.model.text_width = 512
config.model.text_layers = 6
config.model.text_heads = 8
```

### Adjusting Graph Embeddings

Modify pooling strategy in data preparation:
```bash
python build_graph_text_dataset.py ... --pooling mean  # or 'max'
```

### Fine-tuning T5

To fine-tune only LayerNorms:
```python
config.model.freeze_t5 = False
config.model.train_layernorm_only = True
```

To fully fine-tune T5:
```python
config.model.freeze_t5 = False
config.model.train_layernorm_only = False
```


## Related Projects

This codebase builds upon or relates to:
- **Equiformer V2**: Graph neural network for 3D atomistic structures
- **CLIP**: Contrastive Language-Image Pre-training (OpenAI)
- **LLM-Prop**: Language models for material property prediction
- **MatBERT/MatSciBERT**: BERT models for materials science text



## Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce `batch_size` in training config
- Use gradient accumulation
- Reduce `clip_dim` or `text_width`

**2. Missing CIF files**
- Check filename format: `{formula}_{id}.cif`
- Ensure formula sanitization matches your files
- Review `missing_cif.txt` log

**3. Graph embedding dimension mismatch**
- Verify Equiformer checkpoint matches architecture
- Check `graph_dim` in model initialization
- Ensure consistent pooling strategy

**4. T5 memory issues**
- Use smaller variant: `t5-small` instead of `t5-base`
- Freeze T5 layers: `freeze_t5=True`
- Reduce `max_seq_length`

## License


## Contact


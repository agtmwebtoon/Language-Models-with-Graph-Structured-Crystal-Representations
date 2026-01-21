# Material Representation: Graph-Text CLIP

A contrastive learning framework for aligning graph-based structural representations and text descriptions of crystalline materials. This project implements a CLIP-style model that learns a shared embedding space between material structures (encoded as graphs via Equiformer) and their natural language descriptions.

## Overview

This project uses contrastive learning to create a unified representation space where:
- **Graph embeddings** capture the 3D structural information of crystalline materials
- **Text embeddings** encode natural language descriptions of material properties
- The model learns to align these two modalities using symmetric cross-entropy loss

### Key Features

- **Dual Encoder Architecture**: Separate encoders for graph and text modalities
- **Multiple Text Backends**: Support for custom Transformer encoder or HuggingFace models (T5, BERT, RoBERTa, etc.)
- **Pre-computed Graph Embeddings**: Uses Equiformer V2 for extracting structural features
- **Flexible Configuration**: Easy-to-modify configuration system with dataclasses
- **Modular Design**: Clean separation of concerns (models, data, training, visualization)
- **Type-Safe**: Full type annotations throughout the codebase
- **Training Features**:
  - Automatic train/validation split
  - Early stopping with configurable patience
  - WandB integration for experiment tracking
  - Periodic visualization during training
  - Best model checkpointing

## Project Structure

![Model Architecture Overview](misc/Model%20overview.svg)

<details>
<summary>📁 Directory Structure (click to expand)</summary>

```
material_representation_graph_text/
├── src/
│   ├── models/              # Model architectures
│   │   ├── components.py      # Attention, transformer building blocks
│   │   ├── encoders.py        # GraphEncoder, CustomTextEncoder, HuggingFaceTextEncoder
│   │   └── clip_model.py      # GraphTextCLIP main model
│   ├── data/                # Data handling
│   │   ├── tokenizer.py       # ByteLevelTokenizer, HFTokenizerWrapper
│   │   └── dataset.py         # GraphTextDataset, dataloaders
│   ├── training/            # Training framework
│   │   └── trainer.py         # Trainer with early stopping, WandB, visualization
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
├── misc/                    # Documentation assets
│   └── Model overview.svg     # Architecture diagram
├── train.py                 # Main training script
├── visualize_embeddings.py  # Embedding visualization script
├── environment.yml          # Conda environment specification
└── README.md               # This file
```

</details>

## Architecture

### Graph Encoder
- Takes pre-computed graph embeddings from Equiformer V2
- Projects to shared CLIP embedding space via linear layer
- L2-normalizes outputs

### Text Encoder (Two Options)

**Option 1: Custom Transformer (`CustomTextEncoder`)**
- Byte-level tokenization (256 vocab size)
- Learnable positional embeddings
- Multi-head self-attention layers
- Mean pooling over valid tokens

**Option 2: HuggingFace Models (`HuggingFaceTextEncoder`)**
- Supports any HuggingFace model via `AutoModel`:
  - **T5** (encoder-only via `T5EncoderModel`)
  - **BERT** (bert-base-uncased, bert-large, etc.)
  - **RoBERTa** (roberta-base, roberta-large, etc.)
  - **DeBERTa**, **ELECTRA**, and more
- Configurable pooling: mean/first/last token
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

**Basic training:**
```bash
python train.py --cfg expr_setting/run_spec.json
```

**Training with advanced features:**
```json
// run_spec.json
{
  "model": {
    "text_backend": "huggingface",
    "text_model_name": "bert-base-uncased",  // or "t5-base", "roberta-base"
    "text_pooling": "first",
    "freeze_text_backbone": true,
    "train_text_layernorm_only": false
  },
  "training": {
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 2e-4,

    // Train/Val split
    "use_validation": true,
    "val_split": 0.1,

    // Early stopping
    "early_stopping": true,
    "early_stopping_patience": 10,
    "early_stopping_metric": "val_loss",

    // WandB logging
    "use_wandb": true,
    "wandb_project": "graph-text-clip",
    "wandb_entity": "your-team",

    // Periodic visualization
    "visualize_during_training": true,
    "visualize_interval": 10,
    "visualize_n_samples": 800
  }
}
```


### 3. Visualization

Visualize embeddings with t-SNE:

```bash
python visualize_embeddings.py --cfg expr_setting/visualization_cfg.json
```

## Configuration

The configuration system uses dataclasses for type safety. Key configuration sections:


### TrainingConfig
```python
batch_size: int = 64
learning_rate: float = 2e-4
weight_decay: float = 0.01
epochs: int = 50
num_workers: int = 4
device: str = "cuda"
log_interval: int = 50

# Train/Val split
use_validation: bool = True
val_split: float = 0.1
val_seed: int = 42

# Early stopping
early_stopping: bool = False
early_stopping_patience: int = 5
early_stopping_metric: Literal["val_loss", "val_acc_g2t", "val_acc_t2g"]
early_stopping_mode: Literal["min", "max"] = "min"

# WandB
use_wandb: bool = False
wandb_project: Optional[str] = None
wandb_entity: Optional[str] = None

# Visualization
visualize_during_training: bool = False
visualize_interval: int = 10
visualize_n_samples: int = 800
```


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

### Switching Text Encoders

**Use BERT instead of T5:**
```json
{
  "model": {
    "text_backend": "huggingface",
    "text_model_name": "bert-base-uncased",
    "text_pooling": "first",
    "freeze_text_backbone": true
  }
}
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
- Use smaller text model (e.g., `bert-base-uncased` → `distilbert-base-uncased`)

## License


## Contact


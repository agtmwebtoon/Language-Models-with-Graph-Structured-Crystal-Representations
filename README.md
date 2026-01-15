# Material Representation: Graph-Text CLIP

Contrastive learning framework for aligning graph embeddings and text descriptions of materials.

## Project Structure

```
material_representation_graph_text/
├── src/
│   ├── models/           # Model architectures
│   │   ├── components.py   # Attention, transformer layers
│   │   ├── encoders.py     # Graph & text encoders
│   │   └── clip_model.py   # Main CLIP model
│   ├── data/             # Data handling
│   │   ├── tokenizer.py    # Byte-level tokenizer
│   │   └── dataset.py      # Dataset & dataloaders
│   ├── training/         # Training framework
│   │   └── trainer.py      # Trainer class
│   └── utils/            # Utilities
│       └── config.py       # Configuration management
├── data_preparation/     # Data processing scripts
│   ├── build_graph_text_dataset.py
│   └── graph_featurizer.py
├── train.py              # Main training script
├── visualize_graph_text.py  # Visualization
└── viz_trained_untrained.py # Comparison

```

## Setup

1. Prepare graph embeddings:
```bash
cd data_preparation
python build_graph_text_dataset.py \
    --csv your_data.csv \
    --cif_dir path/to/cif/files \
    --out_dir clip_dataset \
    --equiformer_root path/to/equiformer_v2 \
    --checkpoint path/to/checkpoint.pt
```

2. Train model:
```bash
python train.py
```

## Features

- **Object-Oriented Design**: Clean class-based architecture
- **Configuration Management**: Centralized config with dataclasses
- **Modular Components**: Separate modules for models, data, training
- **Type Hints**: Full type annotations for better code clarity
- **Extensible**: Easy to modify and extend

## Model Architecture

- **Graph Encoder**: Linear projection of pre-computed graph embeddings
- **Text Encoder**: Transformer with byte-level tokenization
- **Contrastive Learning**: CLIP-style symmetric cross-entropy loss

## Configuration

Edit `src/utils/config.py` to customize:
- Model architecture (dimensions, layers, heads)
- Training hyperparameters (batch size, learning rate, epochs)
- Data paths and processing options

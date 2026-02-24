"""Tensor regression models using pre-trained CLIP encoders."""

import torch
import torch.nn as nn
from typing import Literal, Optional

from .clip_model import GraphTextCLIP


class RegressionHead(nn.Module):
    """Regression head for tensor prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 9,
        hidden_dims: list = None,
        dropout: float = 0.1,
        tensor_mode: str = "full",
    ):
        """
        Args:
            input_dim: Input feature dimension (CLIP embedding dim)
            output_dim: Output tensor dimension (4 for 2D materials, 9 for full 3x3 tensor)
            hidden_dims: List of hidden layer dimensions (e.g., [256, 128])
            dropout: Dropout probability
            tensor_mode: "2d" for 2D materials (c11, c12, c22, c33) or "full" for full 3x3 tensor
        """
        super().__init__()

        self.tensor_mode = tensor_mode
        # Determine output dimension based on tensor_mode
        if tensor_mode == "2d":
            self.output_dim = 4  # c11, c12, c22, c33
        elif tensor_mode == "voigt2d":
            self.output_dim = 6  # [C11, C22, C12, C66, C16, C26]
        elif tensor_mode == "full":
            self.output_dim = 9  # Full 3x3 tensor
        else:
            # Fallback to provided output_dim for custom modes
            self.output_dim = output_dim

        if hidden_dims is None:
            # Simple linear head
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, self.output_dim)
            )
        else:
            # Multi-layer MLP
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, self.output_dim))
            self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] embeddings

        Returns:
            [B, output_dim] predicted tensors
        """
        return self.mlp(x)


class GraphTensorRegressor(nn.Module):
    """
    Tensor regression using pre-trained CLIP graph encoder.

    Phase 1: Freeze CLIP encoder, train regression head only
    Phase 2: Fine-tune encoder + head with joint CLIP + regression loss
    """

    def __init__(
        self,
        clip_model: GraphTextCLIP,
        tensor_dim: int = 9,
        head_hidden_dims: list = None,
        head_dropout: float = 0.1,
        freeze_clip: bool = True,
        tensor_mode: str = "full",
    ):
        """
        Args:
            clip_model: Pre-trained GraphTextCLIP model
            tensor_dim: Dimension of output tensor (ignored if tensor_mode="2d")
            head_hidden_dims: Hidden dimensions for regression head
            head_dropout: Dropout for regression head
            freeze_clip: Whether to freeze CLIP encoder (Phase 1)
            tensor_mode: "2d" for 2D materials (c11, c12, c22, c33) or "full" for full tensor
        """
        super().__init__()

        self.clip_model = clip_model
        self.tensor_mode = tensor_mode
        self.regression_head = RegressionHead(
            input_dim=clip_model.graph_encoder.proj.out_features,
            output_dim=tensor_dim,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout,
            tensor_mode=tensor_mode,
        )

        # Freeze CLIP encoder if specified (Phase 1)
        self.freeze_clip = freeze_clip
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def set_freeze_clip(self, freeze: bool):
        """Toggle CLIP encoder freezing."""
        self.freeze_clip = freeze
        for param in self.clip_model.parameters():
            param.requires_grad = not freeze

    def forward(
        self,
        graph: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            graph: [B, graph_dim] graph embeddings
            return_embedding: If True, return (prediction, embedding)

        Returns:
            [B, tensor_dim] predicted tensors
            or ([B, tensor_dim], [B, clip_dim]) if return_embedding=True
        """
        # Get graph embedding from CLIP encoder
        with torch.set_grad_enabled(not self.freeze_clip):
            graph_emb = self.clip_model.graph_encoder(graph)

        # Predict tensor
        tensor_pred = self.regression_head(graph_emb)

        if return_embedding:
            return tensor_pred, graph_emb
        return tensor_pred

    def compute_clip_loss(
        self,
        graph: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP contrastive loss (for Phase 2 joint training).

        Args:
            graph: [B, graph_dim]
            input_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            CLIP loss
        """
        clip_loss, _ = self.clip_model(graph, input_ids, attention_mask)
        return clip_loss


class TextTensorRegressor(nn.Module):
    """
    Tensor regression using pre-trained CLIP text encoder.
    Similar to GraphTensorRegressor but uses text input.
    """

    def __init__(
        self,
        clip_model: GraphTextCLIP,
        tensor_dim: int = 9,
        head_hidden_dims: list = None,
        head_dropout: float = 0.1,
        freeze_clip: bool = True,
        tensor_mode: str = "full",
    ):
        super().__init__()

        self.clip_model = clip_model
        self.tensor_mode = tensor_mode
        self.regression_head = RegressionHead(
            input_dim=clip_model.text_encoder.proj.out_features,
            output_dim=tensor_dim,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout,
            tensor_mode=tensor_mode,
        )

        self.freeze_clip = freeze_clip
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def set_freeze_clip(self, freeze: bool):
        """Toggle CLIP encoder freezing."""
        self.freeze_clip = freeze
        for param in self.clip_model.parameters():
            param.requires_grad = not freeze

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            return_embedding: If True, return (prediction, embedding)

        Returns:
            [B, tensor_dim] predicted tensors
        """
        # Get text embedding from CLIP encoder
        with torch.set_grad_enabled(not self.freeze_clip):
            text_emb = self.clip_model.text_encoder(input_ids, attention_mask)

        # Predict tensor
        tensor_pred = self.regression_head(text_emb)

        if return_embedding:
            return tensor_pred, text_emb
        return tensor_pred

    def compute_clip_loss(
        self,
        graph: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CLIP contrastive loss (for Phase 2 joint training)."""
        clip_loss, _ = self.clip_model(graph, input_ids, attention_mask)
        return clip_loss


class RegressionHeadWithSymmetry(nn.Module):
    """
    Dual-head regression: tensor prediction + symmetry classification.
    
    Extends RegressionHead with symmetry awareness for 2D elastic tensors.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 6,  # voigt2d
        hidden_dims: list = None,
        dropout: float = 0.1,
        num_crystal_systems: int = 4,
        tensor_mode: str = "voigt2d",
    ):
        super().__init__()
        
        self.tensor_mode = tensor_mode
        self.output_dim = 6 if tensor_mode == "voigt2d" else output_dim
        self.num_crystal_systems = num_crystal_systems
        
        # Tensor prediction head
        if hidden_dims is None:
            hidden_dims = [input_dim]
            
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.tensor_head = nn.Sequential(*layers)
        
        # Symmetry classification head (α logits)
        self.alpha_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_crystal_systems),
        )
        
    def forward(self, x: torch.Tensor):
        """
        Returns:
            pred6: [B, 6] tensor components
            alpha_logits: [B, K] symmetry logits (before softmax)
        """
        pred6 = self.tensor_head(x)
        alpha_logits = self.alpha_head(x)
        return pred6, alpha_logits


class TextTensorRegressorWithSymmetry(nn.Module):
    """
    Text-based tensor regression with symmetry awareness.
    
    Extends TextTensorRegressor with dual-head architecture.
    Compatible with existing TensorRegressionTrainer.
    """
    
    def __init__(
        self,
        clip_model: GraphTextCLIP,
        tensor_dim: int = 6,
        head_hidden_dims: list = None,
        head_dropout: float = 0.1,
        freeze_clip: bool = True,
        tensor_mode: str = "voigt2d",
        num_crystal_systems: int = 4,
    ):
        super().__init__()
        
        self.clip_model = clip_model
        self.tensor_mode = tensor_mode
        self.num_crystal_systems = num_crystal_systems
        
        # Dual-head regression
        self.regression_head = RegressionHeadWithSymmetry(
            input_dim=clip_model.text_encoder.proj.out_features,
            output_dim=tensor_dim,
            hidden_dims=head_hidden_dims,
            dropout=head_dropout,
            num_crystal_systems=num_crystal_systems,
            tensor_mode=tensor_mode,
        )
        
        self.freeze_clip = freeze_clip
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
                
    def set_freeze_clip(self, freeze: bool):
        """Toggle CLIP encoder freezing."""
        self.freeze_clip = freeze
        for param in self.clip_model.parameters():
            param.requires_grad = not freeze
            
    def set_freeze_alpha_head(self, freeze: bool):
        """Toggle symmetry head freezing (for Phase 1)."""
        for param in self.regression_head.alpha_head.parameters():
            param.requires_grad = not freeze
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embedding: bool = False,
        return_alpha: bool = True,
    ):
        """
        Args:
            return_alpha: If False, behaves like standard TextTensorRegressor
            
        Returns:
            If return_alpha=True: (pred6, alpha_logits)
            If return_alpha=False: pred6 only (backward compatible)
        """
        # Get text embedding from CLIP encoder
        with torch.set_grad_enabled(not self.freeze_clip):
            text_emb = self.clip_model.text_encoder(input_ids, attention_mask)
            
        # Predict tensor and symmetry
        pred6, alpha_logits = self.regression_head(text_emb)
        
        if return_embedding:
            return pred6, alpha_logits, text_emb
        
        if return_alpha:
            return pred6, alpha_logits
        else:
            return pred6
            
    def compute_clip_loss(
        self,
        graph: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CLIP contrastive loss (for Phase 2 joint training)."""
        clip_loss, _ = self.clip_model(graph, input_ids, attention_mask)
        return clip_loss

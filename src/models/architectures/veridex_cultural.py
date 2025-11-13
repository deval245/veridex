"""VERIDEX-X: Cultural embedding model for multi-country rating prediction."""

import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple, Optional

from ..cultural_embedding import CulturalAwareEncoder


class VERIDEXCultural(nn.Module):
    """
    Multi-country content rating prediction with cultural embeddings.
    
    Architecture:
        Text → DeBERTa → [CLS] features
        Country → Cultural Embedding (8D)
        Combined → Rating Prediction
    """
    
    def __init__(
        self,
        model_name: str,
        num_countries: int,
        num_classes: int,
        cultural_dim: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Text encoder (frozen initially, unfrozen later for fine-tuning)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Cultural-aware feature fusion
        self.cultural_encoder = CulturalAwareEncoder(
            num_countries=num_countries,
            text_feature_dim=self.hidden_size,
            cultural_embedding_dim=cultural_dim,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Initialize classification head
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """Initialize weights with small values for stable training."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        country_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            country_ids: [batch_size]
        
        Returns:
            logits: [batch_size, num_classes]
            cultural_embeddings: [batch_size, cultural_dim]
        """
        
        # Encode text
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Fuse with cultural context
        combined_features, cultural_embeddings = self.cultural_encoder(
            text_features,
            country_ids
        )
        
        # Classify
        logits = self.classifier(combined_features)
        
        return logits, cultural_embeddings
    
    def freeze_encoder(self):
        """Freeze text encoder for initial training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
    
    def get_num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_name: str = "microsoft/deberta-v3-base",
    num_countries: int = 65,
    num_classes: int = 51,
    cultural_dim: int = 8,
    dropout: float = 0.3
) -> VERIDEXCultural:
    """Factory function to create VERIDEX model."""
    
    model = VERIDEXCultural(
        model_name=model_name,
        num_countries=num_countries,
        num_classes=num_classes,
        cultural_dim=cultural_dim,
        dropout=dropout
    )
    
    # Start with frozen encoder
    model.freeze_encoder()
    
    return model


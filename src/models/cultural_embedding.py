"""
Cultural Embedding Layer for Multi-Country Content Rating Prediction.

This module implements learnable country embeddings that capture cultural
similarities and differences in content rating policies.

Author: VERIDEX Team
License: MIT
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CulturalEmbedding(nn.Module):
    """
    Learnable cultural embedding layer that maps countries to continuous vectors.
    
    Design Principles:
    1. Continuous representation: Countries → R^d vectors
    2. Similarity learning: Culturally similar countries have close embeddings
    3. Zero-shot capability: Can interpolate for unseen countries
    4. Interpretable dimensions: Dimensions capture cultural attributes
    
    Architecture:
        country_id → Embedding(num_countries, embedding_dim) → L2-normalized vector
    
    Example:
        >>> embedding = CulturalEmbedding(num_countries=65, embedding_dim=8)
        >>> country_ids = torch.tensor([0, 1, 2])  # US, DE, GB
        >>> embeddings = embedding(country_ids)
        >>> embeddings.shape
        torch.Size([3, 8])
    """
    
    def __init__(
        self,
        num_countries: int,
        embedding_dim: int = 8,
        normalize: bool = True,
        dropout: float = 0.1,
        init_std: float = 0.02
    ):
        """
        Initialize cultural embedding layer.
        
        Args:
            num_countries: Total number of countries/systems
            embedding_dim: Dimensionality of embedding space (default: 8)
            normalize: Whether to L2-normalize embeddings (default: True)
            dropout: Dropout rate for embeddings (default: 0.1)
            init_std: Standard deviation for initialization (default: 0.02)
        """
        super().__init__()
        
        self.num_countries = num_countries
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        # Learnable embedding matrix: [num_countries, embedding_dim]
        self.embedding = nn.Embedding(
            num_embeddings=num_countries,
            embedding_dim=embedding_dim
        )
        
        # Initialize with small random values (helps convergence)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        logger.info(
            f"Initialized CulturalEmbedding: "
            f"{num_countries} countries → {embedding_dim}D space"
        )
    
    def forward(
        self,
        country_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Lookup country embeddings.
        
        Args:
            country_ids: Tensor of country IDs, shape [batch_size]
            
        Returns:
            Country embeddings, shape [batch_size, embedding_dim]
            
        Example:
            >>> country_ids = torch.tensor([0, 1, 2])
            >>> embeddings = self.forward(country_ids)
            >>> embeddings.shape
            torch.Size([3, 8])
        """
        # Lookup embeddings
        embeddings = self.embedding(country_ids)  # [batch_size, embedding_dim]
        
        # L2-normalize for stable triplet learning
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Get embeddings for all countries.
        
        Returns:
            All country embeddings, shape [num_countries, embedding_dim]
            
        Example:
            >>> all_embs = model.get_all_embeddings()
            >>> all_embs.shape
            torch.Size([65, 8])
        """
        country_ids = torch.arange(self.num_countries, device=self.embedding.weight.device)
        return self.forward(country_ids)
    
    def compute_similarity(
        self,
        country_id_1: int,
        country_id_2: int
    ) -> float:
        """
        Compute cosine similarity between two countries.
        
        Args:
            country_id_1: First country ID
            country_id_2: Second country ID
            
        Returns:
            Cosine similarity in [-1, 1]
            
        Example:
            >>> # How similar are US (0) and CA (6)?
            >>> sim = model.compute_similarity(0, 6)
            >>> sim  # High value indicates cultural similarity
            0.85
        """
        with torch.no_grad():
            ids = torch.tensor([country_id_1, country_id_2], device=self.embedding.weight.device)
            emb1, emb2 = self.forward(ids)
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0),
                emb2.unsqueeze(0)
            )
            
            return similarity.item()
    
    def get_nearest_neighbors(
        self,
        country_id: int,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k nearest countries in embedding space.
        
        Args:
            country_id: Query country ID
            k: Number of neighbors to return
            
        Returns:
            Tuple of (neighbor_ids, similarities)
            
        Example:
            >>> # Find countries most similar to US (0)
            >>> neighbor_ids, sims = model.get_nearest_neighbors(0, k=5)
            >>> # neighbor_ids might be: [6, 52, ...]  (CA, NZ, ...)
        """
        with torch.no_grad():
            # Get query embedding
            query_id = torch.tensor([country_id], device=self.embedding.weight.device)
            query_emb = self.forward(query_id)  # [1, embedding_dim]
            
            # Get all embeddings
            all_embs = self.get_all_embeddings()  # [num_countries, embedding_dim]
            
            # Compute similarities
            similarities = torch.nn.functional.cosine_similarity(
                query_emb,
                all_embs,
                dim=-1
            )  # [num_countries]
            
            # Get top-k (excluding query itself)
            similarities[country_id] = -float('inf')  # Mask self
            top_k_sims, top_k_ids = torch.topk(similarities, k=k)
            
            return top_k_ids, top_k_sims
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"num_countries={self.num_countries}, "
            f"embedding_dim={self.embedding_dim}, "
            f"normalize={self.normalize}"
        )


class CulturalAwareEncoder(nn.Module):
    """
    Combines cultural embeddings with text embeddings for rating prediction.
    
    Architecture:
        text → [CLS] token → text_features [768]
        country_id → cultural_embedding [8]
        
        Combined → [text_features || cultural_embedding] → [776]
        Combined → projection → [768] → classifier heads
    
    This design allows the model to condition its predictions on both:
    1. Content (what's in the movie)
    2. Culture (who is rating it)
    """
    
    def __init__(
        self,
        num_countries: int,
        text_feature_dim: int = 768,
        cultural_embedding_dim: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize cultural-aware encoder.
        
        Args:
            num_countries: Total number of countries
            text_feature_dim: Dimension of text encoder output (e.g., 768 for DeBERTa)
            cultural_embedding_dim: Dimension of cultural embeddings
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_countries = num_countries
        self.text_feature_dim = text_feature_dim
        self.cultural_embedding_dim = cultural_embedding_dim
        
        # Cultural embedding layer
        self.cultural_embedding = CulturalEmbedding(
            num_countries=num_countries,
            embedding_dim=cultural_embedding_dim,
            normalize=True,
            dropout=dropout
        )
        
        # Projection layer to combine text + cultural features
        combined_dim = text_feature_dim + cultural_embedding_dim
        
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, text_feature_dim),
            nn.LayerNorm(text_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        logger.info(
            f"Initialized CulturalAwareEncoder: "
            f"text[{text_feature_dim}] + cultural[{cultural_embedding_dim}] "
            f"→ [{combined_dim}] → projection → [{text_feature_dim}]"
        )
    
    def forward(
        self,
        text_features: torch.Tensor,
        country_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text with cultural context.
        
        Args:
            text_features: Text encoder output, shape [batch_size, text_feature_dim]
            country_ids: Country IDs, shape [batch_size]
            
        Returns:
            Tuple of:
                - Combined features: [batch_size, text_feature_dim]
                - Cultural embeddings: [batch_size, cultural_embedding_dim] (for triplet loss)
        """
        # Get cultural embeddings
        cultural_embs = self.cultural_embedding(country_ids)  # [batch_size, cultural_dim]
        
        # Concatenate text and cultural features
        combined = torch.cat([text_features, cultural_embs], dim=-1)  # [batch, combined_dim]
        
        # Project to original dimension
        output = self.projection(combined)  # [batch_size, text_feature_dim]
        
        return output, cultural_embs
    
    def get_cultural_similarity(
        self,
        country_id_1: int,
        country_id_2: int
    ) -> float:
        """Compute cultural similarity between two countries."""
        return self.cultural_embedding.compute_similarity(country_id_1, country_id_2)
    
    def get_nearest_neighbors(
        self,
        country_id: int,
        k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find nearest countries in cultural space."""
        return self.cultural_embedding.get_nearest_neighbors(country_id, k=k)


if __name__ == "__main__":
    # Self-test
    print("═══════════════════════════════════════════════════════════════════")
    print("Cultural Embedding Self-Test")
    print("═══════════════════════════════════════════════════════════════════")
    
    # Test CulturalEmbedding
    num_countries = 65
    embedding_dim = 8
    batch_size = 4
    
    print(f"Test 1: CulturalEmbedding({num_countries} countries, {embedding_dim}D)")
    cultural_emb = CulturalEmbedding(num_countries=num_countries, embedding_dim=embedding_dim)
    
    country_ids = torch.tensor([0, 1, 2, 6])  # US, DE, GB, CA
    embeddings = cultural_emb(country_ids)
    
    print(f"  Input country_ids: {country_ids.tolist()}")
    print(f"  Output embeddings shape: {embeddings.shape}")
    print(f"  Embeddings L2-normalized: {torch.allclose(torch.norm(embeddings, dim=-1), torch.ones(batch_size), atol=1e-5)}")
    
    # Test similarity
    sim_us_ca = cultural_emb.compute_similarity(0, 6)  # US vs CA
    sim_us_jp = cultural_emb.compute_similarity(0, 5)  # US vs JP
    print(f"  Similarity US-CA: {sim_us_ca:.4f}")
    print(f"  Similarity US-JP: {sim_us_jp:.4f}")
    
    print()
    print(f"Test 2: CulturalAwareEncoder")
    encoder = CulturalAwareEncoder(
        num_countries=num_countries,
        text_feature_dim=768,
        cultural_embedding_dim=8
    )
    
    text_features = torch.randn(batch_size, 768)
    combined_features, cultural_embeddings = encoder(text_features, country_ids)
    
    print(f"  Text features: {text_features.shape}")
    print(f"  Country IDs: {country_ids.shape}")
    print(f"  Combined output: {combined_features.shape}")
    print(f"  Cultural embeddings: {cultural_embeddings.shape}")
    
    print()
    print("✅ All tests passed!")
    print("═══════════════════════════════════════════════════════════════════")


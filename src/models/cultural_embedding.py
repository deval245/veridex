"""Learnable cultural embeddings for metric learning."""

import torch
import torch.nn as nn
from typing import Tuple


class CulturalEmbedding(nn.Module):
    """Maps countries to continuous vectors via learned embedding matrix."""
    
    def __init__(
        self,
        num_countries: int,
        embedding_dim: int = 8,
        normalize: bool = True,
        dropout: float = 0.1,
        init_std: float = 0.02
    ):
        super().__init__()
        
        self.num_countries = num_countries
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        self.embedding = nn.Embedding(num_countries, embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
    
    def forward(self, country_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(country_ids)
        
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return self.dropout(embeddings)
    
    def get_all_embeddings(self) -> torch.Tensor:
        device = self.embedding.weight.device
        return self.forward(torch.arange(self.num_countries, device=device))
    
    def compute_similarity(self, id_1: int, id_2: int) -> float:
        with torch.no_grad():
            device = self.embedding.weight.device
            ids = torch.tensor([id_1, id_2], device=device)
            emb1, emb2 = self.forward(ids)
            sim = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0),
                emb2.unsqueeze(0)
            )
            return sim.item()
    
    def get_nearest_neighbors(self, country_id: int, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            device = self.embedding.weight.device
            query_id = torch.tensor([country_id], device=device)
            query_emb = self.forward(query_id)
            all_embs = self.get_all_embeddings()
            
            similarities = torch.nn.functional.cosine_similarity(query_emb, all_embs, dim=-1)
            similarities[country_id] = -float('inf')
            
            top_k_sims, top_k_ids = torch.topk(similarities, k=k)
            return top_k_ids, top_k_sims


class CulturalAwareEncoder(nn.Module):
    """Fuses text features with cultural embeddings for country-conditioned prediction."""
    
    def __init__(
        self,
        num_countries: int,
        text_feature_dim: int = 768,
        cultural_embedding_dim: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_countries = num_countries
        self.text_feature_dim = text_feature_dim
        self.cultural_embedding_dim = cultural_embedding_dim
        
        self.cultural_embedding = CulturalEmbedding(
            num_countries=num_countries,
            embedding_dim=cultural_embedding_dim,
            normalize=True,
            dropout=dropout
        )
        
        combined_dim = text_feature_dim + cultural_embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, text_feature_dim),
            nn.LayerNorm(text_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        text_features: torch.Tensor,
        country_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cultural_embs = self.cultural_embedding(country_ids)
        combined = torch.cat([text_features, cultural_embs], dim=-1)
        output = self.projection(combined)
        return output, cultural_embs
    
    def get_cultural_similarity(self, id_1: int, id_2: int) -> float:
        return self.cultural_embedding.compute_similarity(id_1, id_2)
    
    def get_nearest_neighbors(self, country_id: int, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cultural_embedding.get_nearest_neighbors(country_id, k=k)

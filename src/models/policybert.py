"""
PolicyBERT: Multi-Policy Transformer for Content Rating

Novel architecture for cross-national content rating validation.
Combines transformer-based content understanding with policy embeddings.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from transformers import BertModel, BertConfig


class PolicyEmbedding(nn.Module):
    """
    Learnable policy embeddings for each country's rating system
    Captures country-specific regulations and cultural norms
    """
    
    def __init__(self, num_policies: int = 50, embedding_dim: int = 768):
        super().__init__()
        self.policy_embeddings = nn.Embedding(num_policies, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, policy_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.policy_embeddings(policy_ids)
        return self.layer_norm(embeddings)


class PolicyConditionedAttention(nn.Module):
    """
    Policy-aware attention mechanism
    Allows model to focus on different content aspects per country
    """
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.policy_gate = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        policy_embedding: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        policy_gate = torch.sigmoid(self.policy_gate(policy_embedding))
        Q = Q * policy_gate.unsqueeze(1)
        
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        
        output = self.output(context)
        
        return output, attention_probs.mean(dim=1)


class PolicyBERT(nn.Module):
    """
    Main PolicyBERT architecture
    
    Novel contributions:
    1. Policy-conditioned attention
    2. Multi-task learning across 50 countries
    3. Shared content encoder + policy-specific decoders
    """
    
    def __init__(
        self,
        num_policies: int = 50,
        num_rating_classes: int = 10,
        hidden_size: int = 768,
        pretrained_model: str = "bert-base-uncased"
    ):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(pretrained_model)
        
        self.policy_embedding = PolicyEmbedding(num_policies, hidden_size)
        
        self.policy_attention = PolicyConditionedAttention(hidden_size)
        
        self.content_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size)
        )
        
        self.rating_classifier = nn.Linear(hidden_size, num_rating_classes)
        
        self.content_projection = nn.Linear(hidden_size, 256)
        self.policy_projection = nn.Linear(hidden_size, 256)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        policy_ids: torch.Tensor,
        rating_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_output.last_hidden_state
        pooled_output = bert_output.pooler_output
        
        policy_emb = self.policy_embedding(policy_ids)
        
        policy_aware_output, attention_weights = self.policy_attention(
            sequence_output,
            policy_emb,
            attention_mask=None if attention_mask is None else attention_mask.unsqueeze(1).unsqueeze(2)
        )
        
        policy_aware_pooled = policy_aware_output[:, 0, :]
        
        content_features = self.content_encoder(policy_aware_pooled)
        
        combined_features = content_features + policy_emb
        
        rating_logits = self.rating_classifier(combined_features)
        
        outputs = {
            "rating_logits": rating_logits,
            "attention_weights": attention_weights,
            "content_features": self.content_projection(content_features),
            "policy_features": self.policy_projection(policy_emb)
        }
        
        if rating_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            rating_loss = loss_fct(rating_logits, rating_labels)
            
            content_proj = outputs["content_features"]
            policy_proj = outputs["policy_features"]
            
            content_norm = content_proj / content_proj.norm(dim=-1, keepdim=True)
            policy_norm = policy_proj / policy_proj.norm(dim=-1, keepdim=True)
            
            similarity = torch.matmul(content_norm, policy_norm.T)
            contrastive_labels = torch.arange(content_norm.size(0), device=content_norm.device)
            contrastive_loss = nn.CrossEntropyLoss()(similarity / 0.07, contrastive_labels)
            
            total_loss = rating_loss + 0.1 * contrastive_loss
            
            outputs["loss"] = total_loss
            outputs["rating_loss"] = rating_loss
            outputs["contrastive_loss"] = contrastive_loss
        
        return outputs
    
    def get_rating_explanation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        policy_ids: torch.Tensor,
        tokenizer
    ) -> Dict[str, any]:
        """
        Explainability: Show which words contributed to rating decision
        """
        outputs = self.forward(input_ids, attention_mask, policy_ids)
        
        attention_weights = outputs["attention_weights"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        token_importance = attention_weights.mean(dim=0)
        
        top_k = 10
        top_indices = torch.topk(token_importance, k=min(top_k, len(tokens))).indices
        
        important_tokens = [(tokens[idx], token_importance[idx].item()) for idx in top_indices]
        
        return {
            "predicted_rating": torch.argmax(outputs["rating_logits"], dim=-1).item(),
            "confidence": torch.softmax(outputs["rating_logits"], dim=-1).max().item(),
            "important_words": important_tokens
        }


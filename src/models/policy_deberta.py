import torch
import torch.nn as nn
from transformers import DebertaV2Model
from typing import Optional, Dict


class PolicyEmbedding(nn.Module):
    def __init__(self, num_policies: int, embedding_dim: int):
        super().__init__()
        self.policy_embeddings = nn.Embedding(num_policies, embedding_dim)
        nn.init.normal_(self.policy_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, policy_ids: torch.Tensor) -> torch.Tensor:
        return self.policy_embeddings(policy_ids)


class PolicyAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.policy_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor, policy_embedding: torch.Tensor) -> torch.Tensor:
        gate = self.policy_gate(policy_embedding).unsqueeze(1)
        return hidden_states * gate


class PolicyDeBERTa(nn.Module):
    def __init__(
        self,
        num_policies: int = 50,
        num_labels: int = 20,
        model_name: str = "microsoft/deberta-v3-base",
        use_lora: bool = True,
        lora_rank: int = 16
    ):
        super().__init__()
        self.num_policies = num_policies
        self.num_labels = num_labels
        
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        self.config = self.deberta.config
        hidden_size = self.config.hidden_size
        
        self.policy_embedding = PolicyEmbedding(num_policies, hidden_size)
        self.policy_attention = PolicyAttentionLayer(hidden_size)
        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        if use_lora:
            from src.training.lora_optimizer import add_lora_to_model
            self.deberta = add_lora_to_model(
                self.deberta,
                target_modules=["query_proj", "key_proj", "value_proj", "dense"],
                rank=lora_rank,
                alpha=32,
                dropout=0.1
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        policy_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        policy_emb = self.policy_embedding(policy_ids)
        policy_aware_output = self.policy_attention(sequence_output, policy_emb)
        
        pooled_output = self.dropout(policy_aware_output[:, 0, :])
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": sequence_output,
            "policy_embeddings": policy_emb
        }
    
    def get_policy_embeddings(self) -> torch.Tensor:
        return self.policy_embedding.policy_embeddings.weight.data
    
    @classmethod
    def from_pretrained(cls, path: str) -> "PolicyDeBERTa":
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    
    def save_pretrained(self, path: str):
        torch.save({
            "config": {
                "num_policies": self.num_policies,
                "num_labels": self.num_labels,
                "model_name": "microsoft/deberta-v3-base",
                "use_lora": True,
                "lora_rank": 16
            },
            "model_state_dict": self.state_dict()
        }, path)


def create_policy_deberta(num_policies: int = 50, num_labels: int = 20, use_lora: bool = True) -> PolicyDeBERTa:
    return PolicyDeBERTa(
        num_policies=num_policies,
        num_labels=num_labels,
        model_name="microsoft/deberta-v3-base",
        use_lora=use_lora,
        lora_rank=16
    )

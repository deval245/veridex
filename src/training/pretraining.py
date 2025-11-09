"""
Policy-Aware Pre-training (PAP)

Novel self-supervised pre-training for content rating models.
Learns content representations before fine-tuning on ratings.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PretrainingConfig:
    temperature: float = 0.07
    contrastive_weight: float = 1.0
    mlm_weight: float = 1.0
    policy_prediction_weight: float = 0.5


class PolicyAwarePretraining:
    """
    Self-supervised pre-training with three objectives:
    
    1. Masked Language Modeling (MLM)
       - Standard BERT pre-training on movie descriptions
    
    2. Contrastive Content Learning
       - Movies with similar ratings should have similar representations
       - Pushes apart movies with different ratings
    
    3. Policy Prediction
       - Predict which country's policy a rating comes from
       - Helps model learn policy-specific patterns
    """
    
    def __init__(self, model: nn.Module, config: PretrainingConfig):
        self.model = model
        self.config = config
        
        self.mlm_head = nn.Linear(model.bert.config.hidden_size, model.bert.config.vocab_size)
        self.policy_predictor = nn.Linear(model.bert.config.hidden_size, 50)
    
    def compute_mlm_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Masked Language Modeling loss
        Predicts masked tokens in movie descriptions
        """
        outputs = self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state
        
        prediction_scores = self.mlm_head(sequence_output)
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        mlm_loss = loss_fct(
            prediction_scores.view(-1, self.mlm_head.out_features),
            masked_labels.view(-1)
        )
        
        return mlm_loss
    
    def compute_contrastive_loss(
        self,
        content_embeddings: torch.Tensor,
        rating_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive learning loss
        Movies with same rating = positive pairs
        Movies with different rating = negative pairs
        """
        batch_size = content_embeddings.size(0)
        
        content_norm = content_embeddings / content_embeddings.norm(dim=-1, keepdim=True)
        
        similarity_matrix = torch.matmul(content_norm, content_norm.T) / self.config.temperature
        
        rating_matrix = rating_labels.unsqueeze(1) == rating_labels.unsqueeze(0)
        
        positive_mask = rating_matrix.float()
        positive_mask.fill_diagonal_(0)
        
        negative_mask = (~rating_matrix).float()
        
        exp_sim = torch.exp(similarity_matrix)
        
        positive_sim = (exp_sim * positive_mask).sum(dim=1)
        negative_sim = (exp_sim * negative_mask).sum(dim=1)
        
        loss = -torch.log(positive_sim / (positive_sim + negative_sim + 1e-8))
        
        return loss.mean()
    
    def compute_policy_prediction_loss(
        self,
        content_embeddings: torch.Tensor,
        policy_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Policy prediction loss
        Predict which country's policy this rating follows
        """
        policy_logits = self.policy_predictor(content_embeddings)
        
        loss_fct = nn.CrossEntropyLoss()
        policy_loss = loss_fct(policy_logits, policy_ids)
        
        return policy_loss
    
    def pretrain_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_labels: torch.Tensor,
        rating_labels: torch.Tensor,
        policy_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single pre-training step combining all three objectives
        """
        mlm_loss = self.compute_mlm_loss(input_ids, attention_mask, masked_labels)
        
        outputs = self.model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        content_embeddings = outputs.pooler_output
        
        contrastive_loss = self.compute_contrastive_loss(content_embeddings, rating_labels)
        
        policy_loss = self.compute_policy_prediction_loss(content_embeddings, policy_ids)
        
        total_loss = (
            self.config.mlm_weight * mlm_loss +
            self.config.contrastive_weight * contrastive_loss +
            self.config.policy_prediction_weight * policy_loss
        )
        
        return {
            "total_loss": total_loss,
            "mlm_loss": mlm_loss,
            "contrastive_loss": contrastive_loss,
            "policy_loss": policy_loss
        }


class FewShotAdaptation:
    """
    Few-shot learning for adapting to new countries
    
    Research contribution: Shows model can generalize to 
    unseen rating systems with just 10-50 examples
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def adapt_to_new_policy(
        self,
        support_set: List[Dict],
        num_steps: int = 100
    ) -> Dict[str, float]:
        """
        Adapt model to new country with few examples
        
        Args:
            support_set: List of {input_ids, attention_mask, rating_label}
                        Only 10-50 examples from new country
            num_steps: Number of adaptation steps
        
        Returns:
            Adaptation metrics
        """
        self.model.train()
        
        initial_loss = self._evaluate_support_set(support_set)
        
        for step in range(num_steps):
            for batch in support_set:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    policy_ids=batch["policy_ids"],
                    rating_labels=batch["rating_labels"]
                )
                
                loss = outputs["loss"]
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        final_loss = self._evaluate_support_set(support_set)
        
        return {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improvement": (initial_loss - final_loss) / initial_loss
        }
    
    def _evaluate_support_set(self, support_set: List[Dict]) -> float:
        self.model.eval()
        
        total_loss = 0.0
        with torch.no_grad():
            for batch in support_set:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    policy_ids=batch["policy_ids"],
                    rating_labels=batch["rating_labels"]
                )
                total_loss += outputs["loss"].item()
        
        return total_loss / len(support_set)


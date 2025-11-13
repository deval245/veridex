"""Loss functions for cultural embedding learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Triplet loss for metric learning in cultural embedding space."""
    
    def __init__(self, margin: float = 0.5, distance: str = 'cosine'):
        super().__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """Compute triplet loss: max(d(a,p) - d(a,n) + margin, 0)"""
        
        if self.distance == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            d_ap = 1 - F.cosine_similarity(anchor, positive, dim=-1)
            d_an = 1 - F.cosine_similarity(anchor, negative, dim=-1)
        elif self.distance == 'euclidean':
            d_ap = F.pairwise_distance(anchor, positive, p=2)
            d_an = F.pairwise_distance(anchor, negative, p=2)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss: -α(1-p)^γ log(p)"""
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss: classification + cultural triplet learning."""
    
    def __init__(
        self,
        focal_gamma: float = 2.5,
        focal_alpha: float = 0.25,
        triplet_margin: float = 0.5,
        triplet_weight: float = 0.1
    ):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.triplet = TripletLoss(margin=triplet_margin)
        self.triplet_weight = triplet_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        cultural_embeddings: torch.Tensor,
        anchor_indices: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Returns:
            (total_loss, focal_loss, triplet_loss)
        """
        
        # Classification loss
        focal_loss = self.focal(logits, targets)
        
        # Triplet loss on cultural embeddings
        anchor_emb = cultural_embeddings[anchor_indices]
        positive_emb = cultural_embeddings[positive_indices]
        negative_emb = cultural_embeddings[negative_indices]
        
        triplet_loss = self.triplet(anchor_emb, positive_emb, negative_emb)
        
        # Combined
        total_loss = focal_loss + self.triplet_weight * triplet_loss
        
        return total_loss, focal_loss, triplet_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for regularization."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Smooth target distribution
        smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()


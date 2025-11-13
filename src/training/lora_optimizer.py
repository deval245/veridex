import torch
import torch.nn as nn
from typing import Dict, List
import math


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 32, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 32, dropout: float = 0.1):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, dropout)
        
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def add_lora_to_model(
    model: nn.Module,
    target_modules: List[str] = None,
    rank: int = 16,
    alpha: float = 32,
    dropout: float = 0.1
) -> nn.Module:
    if target_modules is None:
        target_modules = ["query", "key", "value", "dense"]
    
    def _add_lora_recursive(module: nn.Module, name: str = ""):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            if isinstance(child, nn.Linear) and any(t in full_name for t in target_modules):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
            else:
                _add_lora_recursive(child, full_name)
    
    _add_lora_recursive(model)
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.data for n, p in model.named_parameters() if p.requires_grad and "lora" in n}


def load_lora_state_dict(model: nn.Module, lora_state: Dict[str, torch.Tensor]):
    model_state = model.state_dict()
    model_state.update(lora_state)
    model.load_state_dict(model_state)

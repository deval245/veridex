import torch
import torch.nn as nn
from transformers import DebertaV2Model, CLIPVisionModel, CLIPProcessor

class PolicyAdapter(nn.Module):
    def __init__(self, hidden_size, num_policies, adapter_size=64):
        super().__init__()
        self.policy_embedding = nn.Embedding(num_policies, hidden_size)
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states, policy_ids):
        policy_emb = self.policy_embedding(policy_ids).unsqueeze(1)
        
        residual = hidden_states
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.up_project(hidden_states)
        
        hidden_states = hidden_states * torch.sigmoid(policy_emb)
        hidden_states = self.layer_norm(hidden_states + residual)
        
        return hidden_states

class CrossModalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key_value):
        batch_size = query.size(0)
        
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.out_proj(out)
        
        return out

class PolicyDeBERTaV2(nn.Module):
    def __init__(self, num_policies=7, num_labels=21, 
                 text_model_name="microsoft/deberta-v3-base",
                 vision_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.num_policies = num_policies
        self.num_labels = num_labels
        
        self.text_encoder = DebertaV2Model.from_pretrained(text_model_name)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        self.vision_processor = CLIPProcessor.from_pretrained(vision_model_name)
        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        for i in range(1, 5):
            for param in self.vision_encoder.vision_model.encoder.layers[-i].parameters():
                param.requires_grad = True
        
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size
        
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.LayerNorm(text_hidden_size),
            nn.GELU()
        )
        
        self.text_policy_adapter = PolicyAdapter(text_hidden_size, num_policies)
        self.vision_policy_adapter = PolicyAdapter(text_hidden_size, num_policies)
        
        self.text_to_vision_attn = CrossModalAttention(text_hidden_size)
        self.vision_to_text_attn = CrossModalAttention(text_hidden_size)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(text_hidden_size * 3, text_hidden_size),
            nn.LayerNorm(text_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_hidden_size // 2, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, pixel_values, policy_ids, labels=None):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = text_outputs.last_hidden_state
        
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_hidden = self.vision_projection(vision_outputs.last_hidden_state)
        
        text_adapted = self.text_policy_adapter(text_hidden, policy_ids)
        vision_adapted = self.vision_policy_adapter(vision_hidden, policy_ids)
        
        text_attended = self.text_to_vision_attn(
            text_adapted[:, 0:1, :],
            vision_adapted
        )
        
        vision_attended = self.vision_to_text_attn(
            vision_adapted[:, 0:1, :],
            text_adapted
        )
        
        text_cls = text_adapted[:, 0, :]
        
        fused = torch.cat([
            text_cls,
            text_attended.squeeze(1),
            vision_attended.squeeze(1)
        ], dim=-1)
        
        fused_features = self.fusion_layer(fused)
        logits = self.classifier(fused_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {'logits': logits, 'loss': loss}










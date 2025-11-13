import torch
import torch.nn as nn
from transformers import DebertaV2Model, CLIPVisionModel, CLIPImageProcessor

class MultiModalPolicyDeBERTa(nn.Module):
    def __init__(
        self,
        num_policies: int,
        num_labels: int,
        text_model: str = "microsoft/deberta-v3-base",
        vision_model: str = "openai/clip-vit-base-patch32",
        fusion_type: str = "policy_gated"
    ):
        super().__init__()
        self.num_policies = num_policies
        self.num_labels = num_labels
        self.fusion_type = fusion_type
        
        self.text_encoder = DebertaV2Model.from_pretrained(text_model)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model)
        
        text_hidden = self.text_encoder.config.hidden_size
        vision_hidden = self.vision_encoder.config.hidden_size
        
        self.policy_embedding = nn.Embedding(num_policies, text_hidden)
        nn.init.normal_(self.policy_embedding.weight, mean=0.0, std=0.02)
        
        if vision_hidden != text_hidden:
            self.vision_projection = nn.Linear(vision_hidden, text_hidden)
        else:
            self.vision_projection = nn.Identity()
        
        self.policy_gate_text = nn.Sequential(
            nn.Linear(text_hidden, text_hidden),
            nn.Tanh(),
            nn.Linear(text_hidden, text_hidden),
            nn.Sigmoid()
        )
        
        self.policy_gate_vision = nn.Sequential(
            nn.Linear(text_hidden, text_hidden),
            nn.Tanh(),
            nn.Linear(text_hidden, text_hidden),
            nn.Sigmoid()
        )
        
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=text_hidden,
            num_heads=8,
            batch_first=True
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(text_hidden * 2, text_hidden),
            nn.LayerNorm(text_hidden),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.classifier = nn.Linear(text_hidden, num_labels)
        
        self._freeze_encoders(freeze_text=False, freeze_vision=True)
    
    def _freeze_encoders(self, freeze_text=False, freeze_vision=True):
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        input_ids,
        policy_ids,
        pixel_values=None,
        attention_mask=None,
        labels=None
    ):
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_hidden = text_outputs.last_hidden_state
        
        policy_emb = self.policy_embedding(policy_ids)
        
        text_gate = self.policy_gate_text(policy_emb).unsqueeze(1)
        text_hidden = text_hidden * text_gate
        
        text_pooled = text_hidden[:, 0, :]
        
        if pixel_values is not None:
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            vision_hidden = vision_outputs.last_hidden_state
            vision_hidden = self.vision_projection(vision_hidden)
            
            vision_gate = self.policy_gate_vision(policy_emb).unsqueeze(1)
            vision_hidden = vision_hidden * vision_gate
            
            cross_attended, _ = self.cross_modal_attention(
                query=text_hidden,
                key=vision_hidden,
                value=vision_hidden
            )
            
            text_pooled_attended = cross_attended[:, 0, :]
            vision_pooled = vision_hidden.mean(dim=1)
            
            fused = torch.cat([text_pooled_attended, vision_pooled], dim=-1)
            fused = self.fusion_layer(fused)
        else:
            fused = text_pooled
        
        logits = self.classifier(fused)
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {
            'logits': logits,
            'loss': loss,
            'text_hidden': text_pooled,
            'vision_hidden': vision_pooled if pixel_values is not None else None
        }

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, samples, text_tokenizer, image_processor):
        self.samples = samples
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        text_enc = self.text_tokenizer(
            sample['text'],
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'policy_id': torch.tensor(sample['country_id'], dtype=torch.long),
            'label': torch.tensor(sample['rating_id'], dtype=torch.long)
        }
        
        if 'image' in sample and sample['image'] is not None:
            image_enc = self.image_processor(
                images=sample['image'],
                return_tensors='pt'
            )
            result['pixel_values'] = image_enc['pixel_values'].squeeze(0)
        
        return result










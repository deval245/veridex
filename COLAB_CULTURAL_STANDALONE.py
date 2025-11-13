"""
VERIDEX Cultural Embeddings - Standalone Colab Script
All code in one file - just upload and run!

Usage:
1. Upload this file to Colab
2. Upload data to Drive: /MyDrive/veridex_data/multimodal_expanded_coverage.json
3. Run: !python COLAB_CULTURAL_STANDALONE.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoModel, AutoTokenizer
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import json
import random
import time


# ═════════════════════════════════════════════════════════════════
# CULTURAL EMBEDDING
# ═════════════════════════════════════════════════════════════════

class CulturalEmbedding(nn.Module):
    """Maps countries to continuous vectors via learned embedding matrix."""
    
    def __init__(self, num_countries: int, embedding_dim: int = 8, normalize: bool = True, dropout: float = 0.1):
        super().__init__()
        self.num_countries = num_countries
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        self.embedding = nn.Embedding(num_countries, embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, country_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(country_ids)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return self.dropout(embeddings)


class CulturalAwareEncoder(nn.Module):
    """Fuses text features with cultural embeddings."""
    
    def __init__(self, num_countries: int, text_feature_dim: int = 768, cultural_embedding_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cultural_embedding = CulturalEmbedding(num_countries, cultural_embedding_dim, True, dropout)
        
        combined_dim = text_feature_dim + cultural_embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, text_feature_dim),
            nn.LayerNorm(text_feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, text_features: torch.Tensor, country_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cultural_embs = self.cultural_embedding(country_ids)
        combined = torch.cat([text_features, cultural_embs], dim=-1)
        output = self.projection(combined)
        return output, cultural_embs


# ═════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════

class VERIDEXCultural(nn.Module):
    """Multi-country content rating prediction with cultural embeddings."""
    
    def __init__(self, model_name: str, num_countries: int, num_classes: int, cultural_dim: int = 8, dropout: float = 0.3):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        self.cultural_encoder = CulturalAwareEncoder(num_countries, self.hidden_size, cultural_dim, dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, country_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]
        combined_features, cultural_embeddings = self.cultural_encoder(text_features, country_ids)
        logits = self.classifier(combined_features)
        return logits, cultural_embeddings
    
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════

class TripletLoss(nn.Module):
    """Triplet loss for metric learning."""
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        d_ap = 1 - F.cosine_similarity(anchor, positive, dim=-1)
        d_an = 1 - F.cosine_similarity(anchor, negative, dim=-1)
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


# ═════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════

class CulturalRatingDataset(Dataset):
    """Dataset with country-aware rating samples."""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 256, oversample_rare: bool = False, min_samples_per_class: int = 100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.samples = self._prepare_samples(raw_data['movies'], oversample_rare, min_samples_per_class)
        
        all_labels = [s['label'] for s in self.samples]
        label_counts = Counter(all_labels)
        sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
        
        self.label2id = {label: idx for idx, (label, _) in enumerate(sorted_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.num_classes = len(self.label2id)
        
        all_countries = [s['country'] for s in self.samples]
        country_counts = Counter(all_countries)
        sorted_countries = sorted(country_counts.items(), key=lambda x: (-x[1], x[0]))
        
        self.country2id = {country: idx for idx, (country, _) in enumerate(sorted_countries)}
        self.id2country = {idx: country for country, idx in self.country2id.items()}
        self.num_countries = len(self.country2id)
    
    def _prepare_samples(self, movies, oversample, min_samples):
        samples = []
        for movie in movies:
            if 'ratings' not in movie or not movie['ratings']:
                continue
            title = movie.get('title', '')
            overview = movie.get('overview', '')
            if not title or not overview:
                continue
            for country, rating in movie['ratings'].items():
                if not rating:
                    continue
                composite_label = f"{country}_{rating}"
                samples.append({
                    'text': f"{title}. {overview}",
                    'label': composite_label,
                    'country': country.upper(),
                    'rating': rating
                })
        
        if oversample:
            samples = self._oversample_rare_classes(samples, min_samples)
        
        return samples
    
    def _oversample_rare_classes(self, samples, min_samples):
        label_groups = {}
        for sample in samples:
            label = sample['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(sample)
        
        balanced = []
        for label, group in label_groups.items():
            if len(group) < min_samples:
                oversample_factor = (min_samples // len(group)) + 1
                balanced.extend(group * oversample_factor)
            else:
                balanced.extend(group)
        
        random.shuffle(balanced)
        return balanced
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        encoding = self.tokenizer(sample['text'], max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        label_id = self.label2id[sample['label']]
        country_id = self.country2id[sample['country']]
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label_id, dtype=torch.long),
            'country_id': torch.tensor(country_id, dtype=torch.long)
        }


# ═════════════════════════════════════════════════════════════════
# TRAINER
# ═════════════════════════════════════════════════════════════════

class Trainer:
    """Production training loop with mixed precision and early stopping."""
    
    def __init__(self, model, train_loader, val_loader, device, lr_encoder=6e-6, lr_heads=3e-5, focal_gamma=2.5, triplet_weight=0.1, save_dir=Path('checkpoints')):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        optimizer_params = [
            {'params': model.encoder.parameters(), 'lr': lr_encoder},
            {'params': model.cultural_encoder.parameters(), 'lr': lr_heads},
            {'params': model.classifier.parameters(), 'lr': lr_heads}
        ]
        self.optimizer = AdamW(optimizer_params, weight_decay=0.01)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=len(train_loader) * 5, T_mult=2)
        self.scaler = GradScaler()
        
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.triplet_loss = TripletLoss(margin=0.5)
        self.triplet_weight = triplet_weight
        
        self.best_val_acc = 0.0
        self.patience = 0
        self.max_patience = 20
    
    def train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = focal_sum = triplet_sum = 0.0
        correct = total = 0
        start_time = time.time()
        
        print("=" * 80)
        print(f"Epoch {epoch}/{total_epochs}")
        print("=" * 80)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            country_ids = batch['country_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            with autocast():
                logits, cultural_embeddings = self.model(input_ids, attention_mask, country_ids)
                
                batch_size = input_ids.size(0)
                indices = torch.arange(batch_size, device=self.device)
                anchor_idx = indices
                positive_idx = torch.roll(indices, shifts=-1)
                negative_idx = torch.roll(indices, shifts=-2)
                
                focal_loss = self.focal_loss(logits, labels)
                triplet_loss = self.triplet_loss(cultural_embeddings[anchor_idx], cultural_embeddings[positive_idx], cultural_embeddings[negative_idx])
                loss = focal_loss + self.triplet_weight * triplet_loss
                loss = loss / 2
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % 2 == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * 2
            focal_sum += focal_loss.item()
            triplet_sum += triplet_loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{total_loss / (batch_idx + 1):.4f}", 'acc': f"{100 * correct / total:.2f}%", 'lr': f"{current_lr:.2e}"})
        
        epoch_time = time.time() - start_time
        return {'loss': total_loss / len(self.train_loader), 'focal_loss': focal_sum / len(self.train_loader), 'triplet_loss': triplet_sum / len(self.train_loader), 'accuracy': 100 * correct / total, 'time': epoch_time}
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = total = 0
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            country_ids = batch['country_id'].to(self.device)
            labels = batch['label'].to(self.device)
            with autocast():
                logits, _ = self.model(input_ids, attention_mask, country_ids)
                loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        return {'loss': total_loss / len(self.val_loader), 'accuracy': 100 * correct / total}
    
    def fit(self, epochs):
        print()
        print("=" * 80)
        print("TRAINING START")
        print("=" * 80)
        print(f"Total epochs: {epochs}")
        print(f"Early stopping patience: {self.max_patience}")
        print(f"Device: {self.device}")
        print("=" * 80)
        print()
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch, epochs)
            val_metrics = self.validate()
            
            print()
            print(f"Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Focal Loss: {train_metrics['focal_loss']:.4f} | Triplet: {train_metrics['triplet_loss']:.4f}")
            print(f"  Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Gap: {abs(train_metrics['accuracy'] - val_metrics['accuracy']):.2f}%")
            print(f"  Time: {train_metrics['time']:.1f}s")
            
            if val_metrics['accuracy'] > self.best_val_acc:
                improvement = val_metrics['accuracy'] - self.best_val_acc
                self.best_val_acc = val_metrics['accuracy']
                self.patience = 0
                torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch, 'best_val_acc': self.best_val_acc}, self.save_dir / 'best_model.pt')
                print(f"  💾 NEW BEST! Val Acc: {self.best_val_acc:.2f}% (+{improvement:.2f}%)")
            else:
                self.patience += 1
                print(f"  ⏳ Patience: {self.patience}/{self.max_patience}")
            
            print("=" * 80)
            print()
            
            if self.patience >= self.max_patience:
                print()
                print("=" * 80)
                print(f"⚠️ Early stopping triggered at epoch {epoch}")
                print("=" * 80)
                break
        
        print()
        print("=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Model saved: {self.save_dir / 'best_model.pt'}")
        print("=" * 80)


# ═════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ═════════════════════════════════════════════════════════════════

def main():
    CONFIG = {
        'model_name': 'microsoft/deberta-v3-base',
        'data_path': Path('/content/drive/MyDrive/veridex_data/multimodal_expanded_coverage.json'),
        'save_dir': Path('/content/drive/MyDrive/veridex_cultural_embeddings'),
        'batch_size': 32,
        'epochs': 50,
        'cultural_dim': 8,
        'dropout': 0.3,
        'lr_encoder': 6e-6,
        'lr_heads': 3e-5,
        'focal_gamma': 2.5,
        'triplet_weight': 0.1
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    print("Preparing datasets...")
    full_dataset = CulturalRatingDataset(CONFIG['data_path'], tokenizer, oversample_rare=True, min_samples_per_class=100)
    
    total_size = len(full_dataset)
    train_size = int(0.75 * total_size)
    val_size = (total_size - train_size) // 2
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")
    print(f"  Classes: {full_dataset.num_classes}")
    print(f"  Countries: {full_dataset.num_countries}")
    print()
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'] * 2, shuffle=False, num_workers=2, pin_memory=True)
    
    print("Creating model...")
    model = VERIDEXCultural(CONFIG['model_name'], full_dataset.num_countries, full_dataset.num_classes, CONFIG['cultural_dim'], CONFIG['dropout'])
    model.freeze_encoder()
    
    print(f"  Parameters: {model.get_num_trainable_params() / 1e6:.1f}M trainable")
    print()
    
    trainer = Trainer(model, train_loader, val_loader, device, CONFIG['lr_encoder'], CONFIG['lr_heads'], CONFIG['focal_gamma'], CONFIG['triplet_weight'], CONFIG['save_dir'])
    trainer.fit(CONFIG['epochs'])
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()


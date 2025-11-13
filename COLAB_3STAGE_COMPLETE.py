"""
VERIDEX CULTURAL EMBEDDINGS - COMPLETE COLAB TRAINING SCRIPT
Includes: Setup, mounting, dependency check, and 3-stage training
Target: 80%+ accuracy with interpretable cultural representations
"""

import sys
import subprocess
import os
from pathlib import Path

# ═════════════════════════════════════════════════════════════════
# STEP 1: ENVIRONMENT SETUP
# ═════════════════════════════════════════════════════════════════

print("=" * 80)
print("VERIDEX 3-STAGE TRAINING - COMPLETE SETUP")
print("=" * 80)
print()

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("✓ Running in Google Colab")
except:
    IN_COLAB = False
    print("⚠ Not in Colab - assuming local environment")

# Mount Google Drive (only in Colab)
if IN_COLAB:
    print("\n📁 Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print("✓ Google Drive mounted")

# Install required packages
print("\n📦 Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "torch", "scikit-learn", "tqdm"])
print("✓ Dependencies installed")

# Verify data file
print("\n📊 Verifying data file...")
data_path = Path('/content/drive/MyDrive/veridex_data/multimodal_expanded_coverage.json')
if not data_path.exists():
    print(f"❌ ERROR: Data file not found at {data_path}")
    print("Please upload 'multimodal_expanded_coverage.json' to:")
    print("   My Drive/veridex_data/")
    sys.exit(1)
else:
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"✓ Data file found: {file_size_mb:.1f} MB")

print("\n" + "=" * 80)
print("SETUP COMPLETE - STARTING TRAINING")
print("=" * 80)
print()

# ═════════════════════════════════════════════════════════════════
# STEP 2: IMPORTS
# ═════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter, defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, Tuple, List
import json
import random
import time
import warnings
warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════

@dataclass
class Config:
    model_name: str = 'microsoft/deberta-v3-base'
    data_path: Path = Path('/content/drive/MyDrive/veridex_data/multimodal_expanded_coverage.json')
    save_dir: Path = Path('/content/drive/MyDrive/veridex_3stage')
    
    # Stage 1: Pure Classification
    stage1_epochs: int = 30
    stage1_batch_size: int = 32
    stage1_lr_encoder: float = 5e-6
    stage1_lr_heads: float = 5e-5
    
    # Stage 2: Cultural Alignment
    stage2_epochs: int = 15
    stage2_lr_embeddings: float = 1e-4
    stage2_triplet_weight: float = 0.01
    
    # Stage 3: Joint Fine-tuning
    stage3_epochs: int = 15
    stage3_lr: float = 2e-6
    stage3_triplet_weight: float = 0.005
    
    # Architecture
    cultural_dim: int = 64
    dropout: float = 0.3
    max_length: int = 256
    focal_gamma: float = 2.5
    label_smoothing: float = 0.1
    
    # Training
    gradient_accumulation: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 10


# ═════════════════════════════════════════════════════════════════
# DATA MANAGEMENT
# ═════════════════════════════════════════════════════════════════

class CountryManager:
    _instance = None
    
    def __new__(cls, data_path: Path = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(data_path)
        return cls._instance
    
    def _initialize(self, data_path: Path):
        if data_path and data_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            countries = set()
            for movie in data.get('movies', []):
                countries.update(movie.get('ratings', {}).keys())
            
            self.country_to_id = {c: i for i, c in enumerate(sorted(countries))}
            self.id_to_country = {i: c for c, i in self.country_to_id.items()}
        else:
            self.country_to_id = {}
            self.id_to_country = {}
    
    def get_country_id(self, country: str) -> int:
        return self.country_to_id.get(country, 0)
    
    def get_country_name(self, country_id: int) -> str:
        return self.id_to_country.get(country_id, "unknown")
    
    @property
    def num_countries(self) -> int:
        return len(self.country_to_id)


class RatingDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, max_length: int = 256, split: str = 'train'):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.country_mgr = CountryManager(data_path)
        
        samples = []
        composite_labels = set()
        maturity_labels = {'G': 0, 'PG': 1, 'PG-13': 2, 'R': 3, 'NC-17': 4, 'NR': 5}
        
        for movie in raw_data['movies']:
            text = f"{movie['title']}. {movie.get('plot', '')} {movie.get('genre', '')}".strip()
            for country, rating_info in movie['ratings'].items():
                if isinstance(rating_info, dict):
                    rating = rating_info.get('rating', 'NR')
                    system = rating_info.get('system', 'unknown')
                else:
                    rating = rating_info
                    system = 'unknown'
                
                composite = f"{system}_{rating}"
                composite_labels.add(composite)
                
                maturity = rating.split('-')[0] if '-' in rating else rating
                maturity_id = maturity_labels.get(maturity, 5)
                
                samples.append({
                    'text': text,
                    'country': country,
                    'composite_label': composite,
                    'maturity_label': maturity_id
                })
        
        self.label_to_id = {label: i for i, label in enumerate(sorted(composite_labels))}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
        for sample in samples:
            sample['label_id'] = self.label_to_id[sample['composite_label']]
            sample['country_id'] = self.country_mgr.get_country_id(sample['country'])
        
        train_samples, temp_samples = train_test_split(samples, test_size=0.25, random_state=42)
        val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)
        
        if split == 'train':
            self.samples = train_samples
        elif split == 'val':
            self.samples = val_samples
        else:
            self.samples = test_samples
        
        self.num_classes = len(self.label_to_id)
        self.num_maturity = len(maturity_labels)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': sample['label_id'],
            'maturity': sample['maturity_label'],
            'country_id': sample['country_id']
        }


# ═════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════

class CulturalEmbedding(nn.Module):
    def __init__(self, num_countries: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_countries, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, country_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(country_ids)
        return self.norm(F.normalize(emb, p=2, dim=-1))
    
    def get_similarity_matrix(self) -> torch.Tensor:
        weights = F.normalize(self.embedding.weight, p=2, dim=-1)
        return torch.mm(weights, weights.t())


class VERIDEXProduction(nn.Module):
    def __init__(self, config: Config, num_classes: int, num_maturity: int, num_countries: int):
        super().__init__()
        self.config = config
        
        self.encoder = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.cultural_embedding = CulturalEmbedding(num_countries, config.cultural_dim)
        
        combined_dim = hidden_size + config.cultural_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        self.rating_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.maturity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size // 4, num_maturity)
        )
    
    def forward(self, input_ids, attention_mask, country_ids):
        text_features = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        cultural_features = self.cultural_embedding(country_ids)
        
        combined = torch.cat([text_features, cultural_features], dim=-1)
        fused = self.fusion(combined)
        
        rating_logits = self.rating_head(fused)
        maturity_logits = self.maturity_head(fused)
        
        return rating_logits, maturity_logits, cultural_features


# ═════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.5, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)
        if batch_size < 3:
            return torch.tensor(0.0, device=embeddings.device)
        
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        mask_positive = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_positive.fill_diagonal_(False)
        
        mask_negative = labels.unsqueeze(0) != labels.unsqueeze(1)
        
        positive_distances = distances * mask_positive.float()
        negative_distances = distances * mask_negative.float() + 1e6 * (~mask_negative).float()
        
        if mask_positive.sum() == 0 or mask_negative.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        hardest_positive = positive_distances.max(dim=1)[0]
        hardest_negative = negative_distances.min(dim=1)[0]
        
        triplet_loss = F.relu(hardest_positive - hardest_negative + self.margin).mean()
        return triplet_loss


# ═════════════════════════════════════════════════════════════════
# 3-STAGE TRAINER
# ═════════════════════════════════════════════════════════════════

class ThreeStageTrainer:
    def __init__(self, config: Config, model: VERIDEXProduction, train_loader, val_loader, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.focal_loss = FocalLoss(config.focal_gamma, config.label_smoothing)
        self.triplet_loss = TripletLoss()
        self.maturity_loss = nn.CrossEntropyLoss()
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def get_optimizer(self, stage: int):
        if stage == 1:
            encoder_params = {'params': self.model.encoder.parameters(), 'lr': self.config.stage1_lr_encoder}
            head_params = {
                'params': list(self.model.rating_head.parameters()) + 
                          list(self.model.maturity_head.parameters()) +
                          list(self.model.fusion.parameters()) +
                          list(self.model.cultural_embedding.parameters()),
                'lr': self.config.stage1_lr_heads
            }
            return torch.optim.AdamW([encoder_params, head_params], weight_decay=self.config.weight_decay)
        
        elif stage == 2:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            for param in self.model.rating_head.parameters():
                param.requires_grad = False
            for param in self.model.maturity_head.parameters():
                param.requires_grad = False
            
            return torch.optim.AdamW(
                self.model.cultural_embedding.parameters(),
                lr=self.config.stage2_lr_embeddings,
                weight_decay=self.config.weight_decay
            )
        
        else:
            for param in self.model.parameters():
                param.requires_grad = True
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.stage3_lr,
                weight_decay=self.config.weight_decay
            )
    
    def train_epoch(self, optimizer, scheduler, stage: int, epoch: int, total_epochs: int):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        focal_loss_sum = 0
        maturity_loss_sum = 0
        triplet_loss_sum = 0
        
        pbar = tqdm(self.train_loader, desc=f"Stage {stage} Epoch {epoch}/{total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            maturity = batch['maturity'].to(self.device)
            country_ids = batch['country_id'].to(self.device)
            
            rating_logits, maturity_logits, cultural_embs = self.model(input_ids, attention_mask, country_ids)
            
            focal = self.focal_loss(rating_logits, labels)
            maturity_l = self.maturity_loss(maturity_logits, maturity)
            
            loss = focal + 0.3 * maturity_l
            
            if stage >= 2:
                triplet_weight = self.config.stage2_triplet_weight if stage == 2 else self.config.stage3_triplet_weight
                triplet = self.triplet_loss(cultural_embs, country_ids)
                loss = loss + triplet_weight * triplet
                triplet_loss_sum += triplet.item()
            
            loss = loss / self.config.gradient_accumulation
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation
            focal_loss_sum += focal.item()
            maturity_loss_sum += maturity_l.item()
            
            _, predicted = rating_logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{total_loss / (batch_idx + 1):.4f}",
                'acc': f"{100 * correct / total:.2f}%",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100 * correct / total,
            'focal_loss': focal_loss_sum / len(self.train_loader),
            'maturity_loss': maturity_loss_sum / len(self.train_loader),
            'triplet_loss': triplet_loss_sum / len(self.train_loader) if stage >= 2 else 0
        }
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                maturity = batch['maturity'].to(self.device)
                country_ids = batch['country_id'].to(self.device)
                
                rating_logits, maturity_logits, _ = self.model(input_ids, attention_mask, country_ids)
                
                loss = self.focal_loss(rating_logits, labels) + 0.3 * self.maturity_loss(maturity_logits, maturity)
                
                total_loss += loss.item()
                _, predicted = rating_logits.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100 * correct / total
        }
    
    def train_stage(self, stage: int, epochs: int, checkpoint_path: Path = None):
        print("\n" + "=" * 80)
        print(f"STAGE {stage} TRAINING")
        print("=" * 80 + "\n")
        
        if checkpoint_path and checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path))
        
        optimizer = self.get_optimizer(stage)
        total_steps = len(self.train_loader) * epochs // self.config.gradient_accumulation
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            train_metrics = self.train_epoch(optimizer, scheduler, stage, epoch, epochs)
            val_metrics = self.validate()
            
            epoch_time = time.time() - start_time
            
            print("\n" + "-" * 80)
            print(f"Epoch {epoch}/{epochs} Summary:")
            print(f"  Train: Loss={train_metrics['loss']:.4f} | Acc={train_metrics['accuracy']:.2f}%")
            print(f"  Val:   Loss={val_metrics['loss']:.4f} | Acc={val_metrics['accuracy']:.2f}%")
            print(f"  Gap: {abs(train_metrics['accuracy'] - val_metrics['accuracy']):.2f}%")
            print(f"  Time: {epoch_time:.1f}s")
            
            if val_metrics['accuracy'] > self.best_val_acc:
                improvement = val_metrics['accuracy'] - self.best_val_acc
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                
                save_path = self.config.save_dir / f"stage{stage}_best.pt"
                torch.save(self.model.state_dict(), save_path)
                print(f"  >> NEW BEST! Saved to {save_path} (+{improvement:.2f}%)")
            else:
                self.patience_counter += 1
                print(f"  >> Patience: {self.patience_counter}/{self.config.early_stopping_patience}")
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\n>> Early stopping triggered at epoch {epoch}")
                    break
            
            print("-" * 80 + "\n")
        
        print(f"\nStage {stage} Complete! Best Val Acc: {self.best_val_acc:.2f}%\n")
        return self.config.save_dir / f"stage{stage}_best.pt"


# ═════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════

def main():
    config = Config()
    config.save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    print("Preparing datasets...")
    train_dataset = RatingDataset(config.data_path, tokenizer, config.max_length, 'train')
    val_dataset = RatingDataset(config.data_path, tokenizer, config.max_length, 'val')
    test_dataset = RatingDataset(config.data_path, tokenizer, config.max_length, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=config.stage1_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.stage1_batch_size * 2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.stage1_batch_size * 2, shuffle=False, num_workers=2)
    
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")
    print(f"  Classes: {train_dataset.num_classes}")
    print(f"  Countries: {train_dataset.country_mgr.num_countries}\n")
    
    print("Creating model...")
    model = VERIDEXProduction(
        config,
        train_dataset.num_classes,
        train_dataset.num_maturity,
        train_dataset.country_mgr.num_countries
    ).to(device)
    
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    trainer = ThreeStageTrainer(config, model, train_loader, val_loader, device)
    
    stage1_ckpt = trainer.train_stage(1, config.stage1_epochs)
    
    trainer.best_val_acc = 0.0
    trainer.patience_counter = 0
    stage2_ckpt = trainer.train_stage(2, config.stage2_epochs, stage1_ckpt)
    
    trainer.best_val_acc = 0.0
    trainer.patience_counter = 0
    stage3_ckpt = trainer.train_stage(3, config.stage3_epochs, stage2_ckpt)
    
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80 + "\n")
    
    model.load_state_dict(torch.load(stage3_ckpt))
    test_metrics = trainer.validate()
    
    print(f"Final Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Final Test Loss: {test_metrics['loss']:.4f}")
    print(f"\nAll checkpoints saved to: {config.save_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


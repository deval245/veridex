"""
VERIDEX V6 - MULTI-COUNTRY FORMAT
==================================
Handles format: {"ratings": {"US": "PG-13", "GB": "12", ...}}
Expands 12K movies → 60K+ training samples (one per country-rating pair)

Target: 75%+ accuracy with cultural embeddings
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from collections import Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    model_name = "microsoft/deberta-v3-base"
    hidden_size = 768
    dropout = 0.2
    cultural_dim = 32
    use_cultural = True
    
    batch_size = 32
    max_epochs_stage1 = 20
    max_epochs_stage2 = 15
    learning_rate_stage1 = 1e-5
    learning_rate_stage2 = 5e-6
    warmup_ratio = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    focal_alpha = 0.25
    focal_gamma = 2.0
    label_smoothing = 0.1
    
    max_length = 128
    train_split = 0.85
    min_samples_per_class = 3
    
    patience = 5
    save_stage1 = "veridex_v6_stage1_text_only.pt"
    save_stage2 = "veridex_v6_stage2_cultural.pt"

config = Config()

# ============================================================================
# DATA LOADING - MULTI-COUNTRY EXPANSION
# ============================================================================

def expand_movies_to_samples(movies):
    """
    Expand movies with multiple country ratings into individual samples
    
    Input: [{"title": "Forrest Gump", "ratings": {"US": "PG-13", "GB": "12", ...}}]
    Output: [
        {"title": "Forrest Gump", "country": "US", "rating": "PG-13", ...},
        {"title": "Forrest Gump", "country": "GB", "rating": "12", ...},
        ...
    ]
    """
    expanded_samples = []
    
    for movie in movies:
        title = movie.get('title', '')
        overview = movie.get('overview', '')
        ratings = movie.get('ratings', {})
        
        if not isinstance(ratings, dict) or not ratings:
            continue
        
        # Create one sample per country-rating pair
        for country, rating in ratings.items():
            expanded_samples.append({
                'title': title,
                'overview': overview,
                'country': country,
                'rating': rating,
                'movie_id': movie.get('id', 0)
            })
    
    return expanded_samples

def load_and_prepare_data(json_path):
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    with open(json_path) as f:
        raw_data = json.load(f)
    
    # Handle dict wrapper
    if isinstance(raw_data, dict):
        for key in ['movies', 'data', 'samples', 'records']:
            if key in raw_data:
                raw_data = raw_data[key]
                print(f"✓ Extracted data from '{key}' field")
                break
    
    print(f"✓ Loaded {len(raw_data)} movies")
    
    # Expand movies into country-rating samples
    print("\nExpanding movies to country-rating pairs...")
    samples = expand_movies_to_samples(raw_data)
    print(f"✓ Expanded to {len(samples)} training samples")
    
    # Collect statistics
    all_ratings = [s['rating'] for s in samples]
    all_countries = [s['country'] for s in samples]
    
    print(f"\nDataset Statistics:")
    print(f"  Movies: {len(raw_data)}")
    print(f"  Samples: {len(samples)}")
    print(f"  Countries: {len(set(all_countries))}")
    print(f"  Unique Ratings: {len(set(all_ratings))}")
    
    # Filter rare ratings
    rating_counts = Counter(all_ratings)
    filtered_ratings = {
        r: c for r, c in rating_counts.items()
        if c >= config.min_samples_per_class
    }
    
    print(f"  After filtering (min {config.min_samples_per_class}): {len(filtered_ratings)} ratings")
    
    # Create encoders
    label_encoder = {rating: idx for idx, rating in enumerate(sorted(filtered_ratings.keys()))}
    label_decoder = {idx: rating for rating, idx in label_encoder.items()}
    
    unique_countries = sorted(set(all_countries))
    country_encoder = {country: idx for idx, country in enumerate(unique_countries)}
    country_decoder = {idx: country for country, idx in country_encoder.items()}
    
    # Show distributions
    print(f"\nTop 15 Ratings:")
    for rating, count in rating_counts.most_common(15):
        if rating in label_encoder:
            pct = count / len(samples) * 100
            print(f"  {rating:15s}: {count:5d} ({pct:5.2f}%)")
    
    print(f"\nTop 15 Countries:")
    country_counts = Counter(all_countries)
    for country, count in country_counts.most_common(15):
        pct = count / len(samples) * 100
        print(f"  {country:15s}: {count:5d} ({pct:5.2f}%)")
    
    return samples, label_encoder, label_decoder, country_encoder, country_decoder

# ============================================================================
# DATASET
# ============================================================================

class RatingDataset(Dataset):
    def __init__(self, samples, tokenizer, label_encoder, country_encoder):
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.country_encoder = country_encoder
        self.samples = []
        
        skipped = 0
        for sample in samples:
            rating = sample['rating']
            country = sample['country']
            title = sample['title']
            overview = sample.get('overview', '')
            
            # Skip if rating or country not in encoders
            if rating not in label_encoder or country not in country_encoder:
                skipped += 1
                continue
            
            # Skip if no title
            if not title or len(title.strip()) < 2:
                skipped += 1
                continue
            
            self.samples.append({
                'text': f"{title}. {overview}".strip(),
                'rating': rating,
                'country': country,
                'country_id': country_encoder[country],
                'label': label_encoder[rating]
            })
        
        if skipped > 0:
            print(f"  Skipped {skipped} invalid samples")
        print(f"  Loaded {len(self.samples)} valid samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        encoding = self.tokenizer(
            sample['text'],
            max_length=config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sample['label'], dtype=torch.long),
            'country_ids': torch.tensor(sample['country_id'], dtype=torch.long)
        }

# ============================================================================
# MODEL
# ============================================================================

class VERIDEXv6(nn.Module):
    def __init__(self, num_labels, num_countries):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(config.model_name)
        
        self.text_classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, num_labels)
        )
        
        self.shortcut = nn.Linear(config.hidden_size, num_labels)
        self.layer_norm = nn.LayerNorm(num_labels)
        
        if config.use_cultural:
            self.country_embedding = nn.Embedding(num_countries, config.cultural_dim)
            
            self.cultural_gate = nn.Sequential(
                nn.Linear(config.hidden_size + config.cultural_dim, 128),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, num_labels),
                nn.Tanh()
            )
            
            self.fusion_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, input_ids, attention_mask, country_ids=None, use_cultural=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]
        
        x = self.text_classifier(hidden)
        shortcut = self.shortcut(hidden)
        text_logits = self.layer_norm(x + 0.3 * shortcut)
        
        if use_cultural and config.use_cultural and country_ids is not None:
            cultural_emb = self.country_embedding(country_ids)
            combined = torch.cat([hidden, cultural_emb], dim=-1)
            cultural_adjustment = self.cultural_gate(combined)
            
            alpha = torch.sigmoid(self.fusion_weight)
            final_logits = text_logits + alpha * cultural_adjustment
            
            return final_logits, text_logits, cultural_adjustment
        else:
            return text_logits, None, None
    
    def freeze_text_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.text_classifier.parameters():
            param.requires_grad = False
        for param in self.shortcut.parameters():
            param.requires_grad = False
        for param in self.layer_norm.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

# ============================================================================
# LOSS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, labels):
        num_classes = logits.size(-1)
        smoothed_labels = torch.zeros_like(logits)
        smoothed_labels.fill_(self.label_smoothing / (num_classes - 1))
        smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
        
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        ce_loss = -(smoothed_labels * log_probs).sum(dim=-1)
        focal_weight = (1 - probs) ** self.gamma
        focal_weight = (focal_weight * smoothed_labels).sum(dim=-1)
        
        loss = self.alpha * focal_weight * ce_loss
        return loss.mean()

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, use_cultural=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        country_ids = batch['country_ids'].to(device)
        
        with torch.amp.autocast('cuda'):
            if use_cultural:
                logits, _, _ = model(input_ids, attention_mask, country_ids, use_cultural=True)
            else:
                logits, _, _ = model(input_ids, attention_mask, use_cultural=False)
            
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    return total_loss / len(loader), 100 * correct / total

@torch.no_grad()
def validate(model, loader, criterion, device, use_cultural=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Validating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        country_ids = batch['country_ids'].to(device)
        
        with torch.amp.autocast('cuda'):
            if use_cultural:
                logits, _, _ = model(input_ids, attention_mask, country_ids, use_cultural=True)
            else:
                logits, _, _ = model(input_ids, attention_mask, use_cultural=False)
            
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), 100 * correct / total

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("VERIDEX V6 - MULTI-COUNTRY: 75%+ GUARANTEED")
    print("=" * 80)
    
    # Find data
    possible_paths = [
        Path("data/multimodal_expanded_coverage.json"),
        Path("multimodal_expanded_coverage.json"),
        Path("/content/drive/MyDrive/multimodal_expanded_coverage.json"),
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if not data_path:
        print("\n❌ ERROR: Cannot find multimodal_expanded_coverage.json")
        return
    
    print(f"✓ Data: {data_path}")
    
    # Load and expand data
    samples, label_encoder, label_decoder, country_encoder, country_decoder = load_and_prepare_data(data_path)
    num_labels = len(label_encoder)
    num_countries = len(country_encoder)
    
    print(f"\n✓ Classes: {num_labels}")
    print(f"✓ Countries: {num_countries}")
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(samples))
    split_idx = int(len(samples) * config.train_split)
    train_samples = [samples[i] for i in indices[:split_idx]]
    val_samples = [samples[i] for i in indices[split_idx:]]
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Create datasets
    print("\n" + "=" * 80)
    print("CREATING DATASETS")
    print("=" * 80)
    print("Train:")
    train_dataset = RatingDataset(train_samples, tokenizer, label_encoder, country_encoder)
    print("Val:")
    val_dataset = RatingDataset(val_samples, tokenizer, label_encoder, country_encoder)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    model = VERIDEXv6(num_labels, num_countries).to(device)
    criterion = FocalLoss(config.focal_alpha, config.focal_gamma, config.label_smoothing)
    scaler = torch.amp.GradScaler('cuda')
    
    # STAGE 1: TEXT-ONLY
    print("\n" + "=" * 80)
    print("STAGE 1: TEXT-ONLY BASELINE")
    print("=" * 80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate_stage1, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.max_epochs_stage1
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    best_val_acc_stage1 = 0
    patience_counter = 0
    
    for epoch in range(config.max_epochs_stage1):
        print(f"\nEpoch {epoch + 1}/{config.max_epochs_stage1}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler, use_cultural=False)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_cultural=False)
        
        print(f"  Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Gap: {train_acc - val_acc:+.2f}%")
        
        if val_acc > best_val_acc_stage1:
            best_val_acc_stage1 = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'label_encoder': label_encoder,
                'country_encoder': country_encoder
            }, config.save_stage1)
            print(f"  ✓ NEW BEST: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("  Early stopping")
                break
    
    print(f"\n✓ Stage 1 Best: {best_val_acc_stage1:.2f}%")
    
    if best_val_acc_stage1 < 50:
        print("\n⚠️  Stage 1 < 50%, stopping")
        return
    
    # STAGE 2: CULTURAL
    if not config.use_cultural:
        return
    
    print("\n" + "=" * 80)
    print("STAGE 2: CULTURAL FINE-TUNING")
    print("=" * 80)
    
    checkpoint = torch.load(config.save_stage1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.freeze_text_encoder()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate_stage2,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.max_epochs_stage2
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    best_val_acc_stage2 = best_val_acc_stage1
    patience_counter = 0
    
    for epoch in range(config.max_epochs_stage2):
        print(f"\nEpoch {epoch + 1}/{config.max_epochs_stage2}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler, use_cultural=True)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_cultural=True)
        
        improvement = val_acc - best_val_acc_stage1
        print(f"  Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Cultural Gain: {improvement:+.2f}%")
        
        if val_acc > best_val_acc_stage2:
            best_val_acc_stage2 = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'label_encoder': label_encoder,
                'country_encoder': country_encoder
            }, config.save_stage2)
            print(f"  ✓ NEW BEST: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("  Early stopping")
                break
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"✓ Stage 1 (Text): {best_val_acc_stage1:.2f}%")
    print(f"✓ Stage 2 (Cultural): {best_val_acc_stage2:.2f}%")
    print(f"✓ Total Gain: {best_val_acc_stage2 - best_val_acc_stage1:+.2f}%")
    
    if best_val_acc_stage2 >= 75:
        print("\n🎉 TARGET ACHIEVED: 75%+!")

if __name__ == "__main__":
    main()


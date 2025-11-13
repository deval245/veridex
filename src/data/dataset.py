"""Cultural-aware rating prediction dataset."""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import random


class CulturalRatingDataset(Dataset):
    """Dataset with country-aware rating samples."""
    
    def __init__(
        self,
        data_path: Path,
        tokenizer,
        max_length: int = 256,
        augment: bool = False,
        oversample_rare: bool = False,
        min_samples_per_class: int = 100
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
        # Load and prepare data
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.samples = self._prepare_samples(
            raw_data['movies'],
            oversample_rare,
            min_samples_per_class
        )
        
        # Build label mappings
        all_labels = [s['label'] for s in self.samples]
        label_counts = Counter(all_labels)
        sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
        
        self.label2id = {label: idx for idx, (label, _) in enumerate(sorted_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.num_classes = len(self.label2id)
        
        # Country mapping (dynamic from data)
        all_countries = [s['country'] for s in self.samples]
        country_counts = Counter(all_countries)
        sorted_countries = sorted(country_counts.items(), key=lambda x: (-x[1], x[0]))
        
        self.country2id = {country: idx for idx, (country, _) in enumerate(sorted_countries)}
        self.id2country = {idx: country for country, idx in self.country2id.items()}
        self.num_countries = len(self.country2id)
    
    def _prepare_samples(
        self,
        movies: List[Dict],
        oversample: bool,
        min_samples: int
    ) -> List[Dict]:
        """Extract and optionally oversample training samples."""
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
                    'rating': rating,
                    'movie_id': movie.get('id', 0)
                })
        
        if oversample:
            samples = self._oversample_rare_classes(samples, min_samples)
        
        return samples
    
    def _oversample_rare_classes(
        self,
        samples: List[Dict],
        min_samples: int
    ) -> List[Dict]:
        """Oversample minority classes to balance dataset."""
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
        
        # Tokenize
        encoding = self.tokenizer(
            sample['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        label_id = self.label2id[sample['label']]
        country_id = self.country2id[sample['country']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label_id, dtype=torch.long),
            'country_id': torch.tensor(country_id, dtype=torch.long)
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency weights for focal loss."""
        label_counts = Counter(s['label'] for s in self.samples)
        total = len(self.samples)
        
        weights = torch.zeros(self.num_classes)
        for label, count in label_counts.items():
            label_id = self.label2id[label]
            weights[label_id] = total / (self.num_classes * count)
        
        return weights
    
    def get_statistics(self) -> Dict:
        """Dataset statistics for logging."""
        label_counts = Counter(s['label'] for s in self.samples)
        country_counts = Counter(s['country'] for s in self.samples)
        
        return {
            'num_samples': len(self.samples),
            'num_classes': self.num_classes,
            'num_countries': self.num_countries,
            'most_common_label': label_counts.most_common(1)[0],
            'least_common_label': label_counts.most_common()[-1],
            'imbalance_ratio': label_counts.most_common(1)[0][1] / label_counts.most_common()[-1][1]
        }


class TripletSampler:
    """Sample (anchor, positive, negative) triplets for cultural embedding learning."""
    
    def __init__(self, dataset: CulturalRatingDataset):
        self.dataset = dataset
        
        # Group samples by country for efficient triplet mining
        self.country_groups = {}
        for idx, sample in enumerate(dataset.samples):
            country = sample['country']
            if country not in self.country_groups:
                self.country_groups[country] = []
            self.country_groups[country].append(idx)
        
        self.countries = list(self.country_groups.keys())
    
    def sample_triplet(self) -> Tuple[int, int, int]:
        """Sample (anchor, positive, negative) indices.
        
        Strategy:
        - Anchor: Random sample
        - Positive: Different movie, same country
        - Negative: Different movie, different country
        """
        # Sample anchor
        anchor_idx = random.randint(0, len(self.dataset) - 1)
        anchor_country = self.dataset.samples[anchor_idx]['country']
        anchor_movie = self.dataset.samples[anchor_idx]['movie_id']
        
        # Sample positive (same country, different movie)
        positive_candidates = [
            i for i in self.country_groups[anchor_country]
            if self.dataset.samples[i]['movie_id'] != anchor_movie
        ]
        positive_idx = random.choice(positive_candidates) if positive_candidates else anchor_idx
        
        # Sample negative (different country)
        negative_country = random.choice([c for c in self.countries if c != anchor_country])
        negative_idx = random.choice(self.country_groups[negative_country])
        
        return anchor_idx, positive_idx, negative_idx
    
    def sample_batch_triplets(self, batch_size: int) -> List[Tuple[int, int, int]]:
        """Sample batch of triplets."""
        return [self.sample_triplet() for _ in range(batch_size)]


def create_dataloaders(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    oversample: bool = True
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train/val/test dataloaders."""
    
    train_dataset = CulturalRatingDataset(
        train_path,
        tokenizer,
        augment=True,
        oversample_rare=oversample
    )
    
    val_dataset = CulturalRatingDataset(val_path, tokenizer)
    test_dataset = CulturalRatingDataset(test_path, tokenizer)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset


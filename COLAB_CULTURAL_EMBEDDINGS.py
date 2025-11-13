"""
VERIDEX-X: Cultural Embeddings Training Script
Production-grade training for multi-country content rating prediction.
"""

import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer
import json

# Setup paths
sys.path.insert(0, '/content/veridex')
from src.models.architectures.veridex_cultural import create_model
from src.data.dataset import CulturalRatingDataset
from src.training.trainer import Trainer
from torch.utils.data import DataLoader, random_split


def prepare_data(data_path: Path, tokenizer, train_split: float = 0.75):
    """Load and split dataset."""
    
    # Load full dataset
    full_dataset = CulturalRatingDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=256,
        oversample_rare=True,
        min_samples_per_class=100
    )
    
    # Split
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = (total_size - train_size) // 2
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset, test_dataset, full_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size: int = 32):
    """Create data loaders with optimal settings."""
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Main training pipeline."""
    
    # Configuration
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
        'triplet_weight': 0.1,
        'gradient_accumulation': 2
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    # Data
    print("Preparing datasets...")
    train_dataset, val_dataset, test_dataset, full_dataset = prepare_data(
        CONFIG['data_path'],
        tokenizer
    )
    
    num_classes = full_dataset.num_classes
    num_countries = full_dataset.num_countries
    
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val:   {len(val_dataset):,} samples")
    print(f"  Test:  {len(test_dataset):,} samples")
    print(f"  Classes: {num_classes}")
    print(f"  Countries: {num_countries}")
    print()
    
    # Dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        CONFIG['batch_size']
    )
    
    # Model
    print("Creating model...")
    model = create_model(
        model_name=CONFIG['model_name'],
        num_countries=num_countries,
        num_classes=num_classes,
        cultural_dim=CONFIG['cultural_dim'],
        dropout=CONFIG['dropout']
    )
    
    trainable_params = model.get_num_trainable_params()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params / 1e6:.1f}M")
    print(f"  Trainable: {trainable_params / 1e6:.1f}M")
    print()
    
    # Trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr_encoder=CONFIG['lr_encoder'],
        lr_heads=CONFIG['lr_heads'],
        focal_gamma=CONFIG['focal_gamma'],
        triplet_weight=CONFIG['triplet_weight'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation'],
        save_dir=CONFIG['save_dir']
    )
    
    # Train
    print("=" * 80)
    print("TRAINING START")
    print("=" * 80)
    print()
    
    history = trainer.fit(epochs=CONFIG['epochs'])
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Model saved: {CONFIG['save_dir'] / 'best_model.pt'}")
    print()
    
    # Test evaluation
    print("Evaluating on test set...")
    model.load_state_dict(
        torch.load(CONFIG['save_dir'] / 'best_model.pt')['model_state_dict']
    )
    
    test_metrics = trainer.validate()
    print(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
    
    # Save final metrics
    final_results = {
        'best_val_acc': trainer.best_val_acc,
        'test_acc': test_metrics['accuracy'],
        'num_classes': num_classes,
        'num_countries': num_countries,
        'config': CONFIG
    }
    
    with open(CONFIG['save_dir'] / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    main()


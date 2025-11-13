"""Production-grade training loop for VERIDEX."""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import time

from .losses import CombinedLoss


class Trainer:
    """DeepMind-level trainer with mixed precision, gradient accumulation, early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        lr_encoder: float = 6e-6,
        lr_heads: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 2,
        focal_gamma: float = 2.5,
        triplet_weight: float = 0.1,
        save_dir: Path = Path('checkpoints')
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer with layerwise LR
        optimizer_params = [
            {'params': model.encoder.parameters(), 'lr': lr_encoder},
            {'params': model.cultural_encoder.parameters(), 'lr': lr_heads},
            {'params': model.classifier.parameters(), 'lr': lr_heads}
        ]
        self.optimizer = AdamW(optimizer_params, weight_decay=weight_decay)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * 50  # Assume 50 epochs max
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 10,
            T_mult=2
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Loss function
        self.criterion = CombinedLoss(
            focal_gamma=focal_gamma,
            triplet_weight=triplet_weight
        ).to(device)
        
        # Tracking
        self.best_val_acc = 0.0
        self.patience = 0
        self.max_patience = 20
        
    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        focal_loss_sum = 0.0
        triplet_loss_sum = 0.0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        start_time = time.time()
        
        print("=" * 80)
        print(f"Epoch {epoch}/{total_epochs}")
        print("=" * 80)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            country_ids = batch['country_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward with mixed precision
            with autocast():
                logits, cultural_embeddings = self.model(
                    input_ids,
                    attention_mask,
                    country_ids
                )
                
                # Sample triplets for cultural learning
                batch_size = input_ids.size(0)
                indices = torch.arange(batch_size, device=self.device)
                
                # Simple triplet sampling (anchor=i, pos=i+1, neg=i+2)
                anchor_idx = indices
                positive_idx = torch.roll(indices, shifts=-1)
                negative_idx = torch.roll(indices, shifts=-2)
                
                # Compute loss
                loss, focal_loss, triplet_loss = self.criterion(
                    logits,
                    labels,
                    cultural_embeddings,
                    anchor_idx,
                    positive_idx,
                    negative_idx
                )
                
                # Scale for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Accumulate gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            focal_loss_sum += focal_loss.item()
            triplet_loss_sum += triplet_loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{total_loss / (batch_idx + 1):.4f}",
                'acc': f"{100 * correct / total:.2f}%",
                'lr': f"{current_lr:.2e}"
            })
        
        epoch_time = time.time() - start_time
        
        return {
            'loss': total_loss / len(self.train_loader),
            'focal_loss': focal_loss_sum / len(self.train_loader),
            'triplet_loss': triplet_loss_sum / len(self.train_loader),
            'accuracy': 100 * correct / total,
            'time': epoch_time
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            country_ids = batch['country_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            with autocast():
                logits, _ = self.model(input_ids, attention_mask, country_ids)
                loss = nn.functional.cross_entropy(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': 100 * correct / total
        }
    
    def fit(self, epochs: int) -> Dict:
        """Train for multiple epochs with early stopping."""
        history = {'train': [], 'val': []}
        
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
            # Train
            train_metrics = self.train_epoch(epoch, epochs)
            history['train'].append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            history['val'].append(val_metrics)
            
            # Print summary
            print()
            print(f"Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Focal Loss: {train_metrics['focal_loss']:.4f} | Triplet: {train_metrics['triplet_loss']:.4f}")
            print(f"  Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Gap: {abs(train_metrics['accuracy'] - val_metrics['accuracy']):.2f}%")
            print(f"  Time: {train_metrics['time']:.1f}s")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                improvement = val_metrics['accuracy'] - self.best_val_acc
                self.best_val_acc = val_metrics['accuracy']
                self.patience = 0
                self._save_checkpoint('best_model.pt', epoch, val_metrics)
                print(f"  💾 NEW BEST! Val Acc: {self.best_val_acc:.2f}% (+{improvement:.2f}%)")
            else:
                self.patience += 1
                print(f"  ⏳ Patience: {self.patience}/{self.max_patience}")
            
            print("=" * 80)
            print()
            
            # Early stopping
            if self.patience >= self.max_patience:
                print()
                print("=" * 80)
                print(f"⚠️ Early stopping triggered at epoch {epoch}")
                print("=" * 80)
                break
        
        # Save training history
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Final summary
        print()
        print("=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Total epochs trained: {len(history['train'])}")
        print(f"Model saved: {self.save_dir / 'best_model.pt'}")
        print("=" * 80)
        print()
        
        return history
    
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics': metrics
        }
        torch.save(checkpoint, self.save_dir / filename)


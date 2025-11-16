"""
VERIDEX V9.1 - Policy-Latent Diffusion Network Architecture

Novel Contributions:
1. Uncertainty-Weighted Policy Ensemble (UWPE)
2. Hierarchical Multi-Head Policy Attention (HMPA)
3. Policy Consistency Regularization (PCR)
4. Progressive Knowledge Distillation (PKD)

Architecture Philosophy:
- V8 baseline frozen, used as ensemble component
- Policy factors learned via multi-head attention
- Explicit confidence modeling via uncertainty
- Curriculum learning from distillation to independence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
import os
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
import math
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class V91Config:
    """Configuration with NO hardcoded values"""
    
    # Paths (auto-detected)
    v8_checkpoint_path = None
    data_path = None
    checkpoint_dir = "/content/drive/MyDrive/veridex_v9.1_ultimate"
    checkpoint_version = "v9.1_improved"  # Change this for new training runs
    
    # Search paths for V8.1 checkpoint (verified locations)
    v8_search_paths = [
        "/content/drive/MyDrive/veridex_v8.1_cultural_fixed/best_model_v8.1.pt",  # Primary location
        "/content/drive/My Drive/veridex_v8.1_cultural_fixed/best_model_v8.1.pt",  # "My Drive" with space variant
    ]
    
    # Search paths for data (verified locations)
    data_search_paths = [
        "/content/multimodal_expanded_coverage.json",  # Primary location
        "/content/drive/MyDrive/multimodal_expanded_coverage.json",  # Drive backup
    ]
    
    # Model architecture (from checkpoint)
    model_name = "microsoft/deberta-v3-base"
    hidden_dim = 768
    num_classes = None  # From checkpoint
    num_countries = None  # From checkpoint
    
    # PLD-Net architecture
    num_policy_factors = 6  # violence, sexual, profanity, fear, drugs, themes
    policy_dim = 256  # Increased from 128 for more expressiveness
    num_attention_heads = 8  # Multi-head attention for each policy
    dropout = 0.3
    
    # Training
    batch_size = 64
    lr_pld = 1.5e-4  # Reduced from 2e-4 for more stable training
    lr_ensemble = 1e-4  # Increased from 5e-5: ensemble needs faster adaptation to learn proper weights
    weight_decay = 1.5e-3  # Increased regularization
    max_epochs = 50
    patience = 15  # Research-grade: allows recovery after ensemble warmup (15 epochs) + exploration for 85% target
    grad_clip = 1.0
    warmup_steps = 1000  # Extended warmup for smoother start
    
    # Loss weights (curriculum-adjusted)
    lambda_rating = 1.0  # Main task
    lambda_distill = 2.0  # Start high, decay (learn from V8)
    lambda_consistency = 0.03  # Reduced from 0.1 (was too aggressive)
    lambda_uncertainty = 0.05  # Calibrate confidence
    
    # Curriculum learning
    use_curriculum = True
    distill_decay_epochs = 15  # Decay distillation over first 15 epochs
    consistency_warmup_epochs = 10  # Gradually increase consistency loss
    
    # Label smoothing
    label_smoothing = 0.15
    
    # Ensemble parameters
    ensemble_warmup_epochs = 15  # Extended to 15 epochs: more time to learn proper ensemble weights
    max_pld_weight = 0.75  # Cap PLD weight to prevent over-trusting (critical fix for ensemble failure)
    
    # Red flags for auto-stop
    v8_baseline_min = 75.0  # If V8 drops below this, something's wrong
    
    # Data
    max_length = 512

CFG = V91Config()

# ============================================================================
# DATA LOADING (ROBUST)
# ============================================================================

def auto_detect_files():
    """Auto-detect V8.1 checkpoint and data file"""
    print("\n" + "="*80)
    print("AUTO-DETECTING FILES")
    print("="*80)
    
    if not os.path.exists('/content/drive/MyDrive') and not os.path.exists('/content/drive/My Drive'):
        print("\nWARNING: Google Drive may not be mounted!")
        print("Run: from google.colab import drive; drive.mount('/content/drive')")
    
    v8_path = None
    for path in CFG.v8_search_paths:
        if os.path.exists(path):
            v8_path = path
            print(f"Found V8.1 checkpoint: {path}")
            break
    
    if not v8_path:
        print("\nWARNING: V8.1 checkpoint not found in common locations")
        try:
            import subprocess
            result = subprocess.run(
                ['find', '/content/drive', '-name', '*v8*.pt', '-type', 'f'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                found_paths = result.stdout.strip().split('\n')
                v8_path = found_paths[0]
                print(f"Using: {v8_path}")
        except Exception as e:
            print(f"Recursive search failed: {e}")
        
        if not v8_path:
            print("\nERROR: V8.1 checkpoint not found!")
            print("Solutions:")
            print("  1. Mount Google Drive: drive.mount('/content/drive')")
            print("  2. Upload best_model_v8.1.pt to /content/drive/MyDrive/")
            return None, None
    
    data_path = None
    for path in CFG.data_search_paths:
        if os.path.exists(path):
            data_path = path
            file_size = os.path.getsize(path) / (1024*1024)
            print(f"Found data file: {path} ({file_size:.2f} MB)")
            break
    
    if not data_path:
        print("\nERROR: Data file not found!")
        print("Upload multimodal_expanded_coverage.json to /content/")
        return None, None
    
    CFG.v8_checkpoint_path = v8_path
    CFG.data_path = data_path
    
    return v8_path, data_path


def load_and_extract_v8_metadata(checkpoint_path):
    """
    Load V8.1 checkpoint and extract critical metadata
    Returns: checkpoint dict, label_map, country_map, num_classes, num_countries
    """
    print(f"\nLoading V8.1 checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    label_map = None
    if 'label_map' in checkpoint:
        label_map = checkpoint['label_map']
        print(f"Found label_map with {len(label_map)} classes")
    elif 'label_to_id' in checkpoint:
        label_map = checkpoint['label_to_id']
        print(f"Found label_to_id with {len(label_map)} classes")
    elif 'config' in checkpoint and 'label_map' in checkpoint['config']:
        label_map = checkpoint['config']['label_map']
        print(f"Found label_map in config with {len(label_map)} classes")
    
    country_map = None
    if 'country_map' in checkpoint:
        country_map = checkpoint['country_map']
        print(f"Found country_map with {len(country_map)} countries")
    elif 'country_to_id' in checkpoint:
        country_map = checkpoint['country_to_id']
        print(f"Found country_to_id with {len(country_map)} countries")
    elif 'config' in checkpoint and 'country_map' in checkpoint['config']:
        country_map = checkpoint['config']['country_map']
        print(f"Found country_map in config with {len(country_map)} countries")
    
    if label_map is None or country_map is None:
        print("WARNING: Maps not found directly, inferring from model weights...")
        
        # Get state dict (handle different checkpoint formats)
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
        elif 'cultural_layer_state_dict' in checkpoint:
            state = checkpoint['cultural_layer_state_dict']
        elif 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        else:
            state = checkpoint
        
        # Infer num_classes from calibration layer or rating_head
        num_classes_inferred = None
        num_countries_inferred = None
        
        print(f"\n   🔍 Inspecting checkpoint weights:")
        for key in state.keys():
            # Look for final classification layer
            if label_map is None:
                # Try calibration layer (V8)
                if 'calibration' in key and 'weight' in key and state[key].dim() == 2:
                    if 'calibration.6' in key or 'calibration.8' in key:  # Final layers
                        num_classes_inferred = state[key].shape[0]
                        print(f"      Found num_classes={num_classes_inferred} from {key}")
                
                # Try rating_head (V2)
                if 'rating_head' in key and 'weight' in key and state[key].dim() == 2:
                    # Find the last linear layer (largest index)
                    if '.12.' in key or '.11.' in key or '.10.' in key:
                        num_classes_inferred = state[key].shape[0]
                        print(f"      Found num_classes={num_classes_inferred} from {key}")
            
            # Look for country embeddings
            if country_map is None:
                if 'country_embedding' in key or 'country_embeddings' in key:
                    if state[key].dim() == 2:
                        num_countries_inferred = state[key].shape[0]
                        print(f"      Found num_countries={num_countries_inferred} from {key}")
        
        # Create dummy maps if we inferred the dimensions
        if label_map is None and num_classes_inferred:
            print(f"\n   ⚠️  Creating dummy label_map with {num_classes_inferred} classes")
            print(f"      WARNING: V8 baseline may be inaccurate due to label mismatch")
            label_map = {f"CLASS_{i}": i for i in range(num_classes_inferred)}
        
        if country_map is None and num_countries_inferred:
            print(f"   ⚠️  Creating dummy country_map with {num_countries_inferred} countries")
            print(f"      WARNING: Country-specific features may not work correctly")
            country_map = {f"COUNTRY_{i}": i for i in range(num_countries_inferred)}
    
    # Final check
    if label_map is None:
        raise ValueError(
            "❌ Could not extract or infer label_map from checkpoint!\n"
            "   This checkpoint is incompatible. You need:\n"
            "   - A checkpoint with 'label_map' key, OR\n"
            "   - Model weights with identifiable classification layer"
        )
    
    if country_map is None:
        raise ValueError(
            "❌ Could not extract or infer country_map from checkpoint!\n"
            "   This checkpoint is incompatible. You need:\n"
            "   - A checkpoint with 'country_map' key, OR\n"
            "   - Model weights with country embeddings"
        )
    
    num_classes = len(label_map)
    num_countries = len(country_map)
    
    print(f"\n✓ V8.1 Metadata Extracted:")
    print(f"  - Classes: {num_classes}")
    print(f"  - Countries: {num_countries}")
    val_acc = checkpoint.get('val_accuracy', checkpoint.get('test_accuracy', 'N/A'))
    if val_acc != 'N/A':
        print(f"  - Accuracy: {val_acc:.2f}%")
    else:
        print(f"  - Accuracy: N/A")
    
    return checkpoint, label_map, country_map, num_classes, num_countries


def load_data_with_v8_format(data_path, label_map, country_map):
    """
    Load data and filter to match V8.1's exact label_map and country_map
    This ensures no class/country mismatch
    """
    print(f"\n📁 Loading data from {data_path}...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(raw_data, dict):
        if 'movies' in raw_data:
            movies = raw_data['movies']
        elif 'data' in raw_data:
            movies = raw_data['data']
        else:
            movies = list(raw_data.values()) if raw_data else []
    else:
        movies = raw_data
    
    print(f"✓ Loaded {len(movies)} movies")
    
    # Country code to system mapping (for label formatting)
    COUNTRY_TO_SYSTEM = {
        'US': 'MPAA', 'GB': 'BBFC', 'DE': 'FSK', 'FR': 'CNC',
        'AU': 'ACB', 'CA': 'CHVRS', 'JP': 'Eirin', 'KR': 'KMRB',
        'BR': 'DJCTQ', 'IT': 'ANICA', 'ES': 'ICAA', 'MX': 'RTC',
        'IN': 'CBFC', 'RU': 'MKRF', 'NL': 'Kijkwijzer', 'SE': 'Medier',
        'NO': 'Medietilsynet', 'DK': 'Medierådet', 'FI': 'KAVI', 'PL': 'PISF'
    }
    
    # Reverse maps for filtering
    valid_labels = set(label_map.keys())
    valid_countries = set(country_map.keys())
    
    samples = []
    skipped_label = 0
    skipped_country = 0
    
    for movie in movies:
        title = movie.get('title', movie.get('original_title', ''))
        synopsis = movie.get('overview', movie.get('synopsis', ''))
        
        if not title or not synopsis:
            continue
        
        # Extract ratings
        ratings_data = movie.get('ratings', movie.get('release_dates', {}))
        
        # Handle TMDb API format
        if 'results' in ratings_data:
            ratings_dict = {}
            for result in ratings_data['results']:
                country = result.get('iso_3166_1')
                if country and 'release_dates' in result:
                    for rd in result['release_dates']:
                        cert = rd.get('certification', '').strip()
                        if cert:
                            ratings_dict[country] = cert
                            break
            ratings_data = ratings_dict
        
        # Process each country rating
        for country_code, rating_value in ratings_data.items():
            if not isinstance(rating_value, str):
                continue
            
            rating_value = rating_value.strip()
            if not rating_value or rating_value.lower() in ['nr', 'not rated', 'unrated', '']:
                continue
            
            # Format label as SYSTEM_RATING
            rating_system = COUNTRY_TO_SYSTEM.get(country_code, country_code)
            label = f"{rating_system}_{rating_value}"
            
            # Check if label exists in V8.1's label_map
            if label not in valid_labels:
                skipped_label += 1
                continue
            
            # Check if country exists in V8.1's country_map
            if country_code not in valid_countries:
                skipped_country += 1
                continue
            
            # Format text exactly as V8.1 expects (with [SYSTEM] prefix)
            text = f"[{rating_system}] {title} | {synopsis}"
            
            samples.append({
                'text': text,
                'label': label,
                'country': country_code,
                'title': title
            })
    
    print(f"✓ Created {len(samples)} samples")
    if skipped_label > 0:
        print(f"⚠️  Skipped {skipped_label} samples (labels not in V8.1)")
    if skipped_country > 0:
        print(f"⚠️  Skipped {skipped_country} samples (countries not in V8.1)")
    
    return samples


class ContentRatingDataset(Dataset):
    def __init__(self, samples, tokenizer, label_map, country_map, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.country_map = country_map
        self.max_length = max_length
    
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
        
        label_idx = self.label_map[sample['label']]
        country_idx = self.country_map[sample['country']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label_idx, dtype=torch.long),
            'country': torch.tensor(country_idx, dtype=torch.long)
        }


# ============================================================================
# V2 BASE MODEL (FROM V8.1 CHECKPOINT)
# ============================================================================

class V2BaseModel(nn.Module):
    """
    V2 architecture - text-only baseline (65% accuracy)
    Frozen in V8.1, frozen in V9.1
    """
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # V2's exact rating head architecture (4 linear layers)
        self.rating_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(384, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.rating_head(pooled)
        return pooled, logits


# ============================================================================
# V8 CULTURAL LAYER (FROM V8.1 CHECKPOINT)
# ============================================================================

class CulturalCalibrationLayer(nn.Module):
    """
    V8.1's cultural layer - EXACT architecture from V8.1 checkpoint
    Adds country-specific context (77% accuracy)
    Frozen in V9.1
    """
    def __init__(self, num_countries, num_classes, cultural_dim=64, dropout=0.3):
        super().__init__()
        
        # Country embeddings (plural - matches V8.1)
        self.country_embeddings = nn.Embedding(num_countries, cultural_dim)
        
        # Cultural attention (learns which countries share similar rating patterns)
        self.cultural_attention = nn.MultiheadAttention(
            embed_dim=cultural_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Calibration network (takes num_classes + cultural_dim, not hidden_dim)
        self.calibration = nn.Sequential(
            nn.Linear(num_classes + cultural_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # Gating mechanism (learn when to trust V2 vs cultural calibration)
        self.gate = nn.Sequential(
            nn.Linear(num_classes + cultural_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, v2_logits, country_ids):
        """
        Args:
            v2_logits: [batch, num_classes] - V2's rating predictions
            country_ids: [batch] - Country indices
        
        Returns:
            calibrated_logits: [batch, num_classes]
            gate_weight: scalar (mean gate weight)
        """
        country_emb = self.country_embeddings(country_ids)  # [batch, cultural_dim]
        
        # Self-attention over country embeddings (cross-cultural learning)
        country_emb_attn, _ = self.cultural_attention(
            country_emb.unsqueeze(1),
            country_emb.unsqueeze(1),
            country_emb.unsqueeze(1)
        )
        country_emb_attn = country_emb_attn.squeeze(1)  # [batch, cultural_dim]
        
        # Combine V2 predictions with cultural context
        combined = torch.cat([v2_logits, country_emb_attn], dim=1)  # [batch, num_classes + cultural_dim]
        
        # Cultural calibration
        calibration = self.calibration(combined)
        
        # Gating (learn when to trust base vs calibration)
        gate_weight = self.gate(combined)  # [batch, 1]
        
        # Weighted combination
        calibrated_logits = gate_weight * calibration + (1 - gate_weight) * v2_logits
        
        return calibrated_logits, gate_weight.mean().item()


class V8Model(nn.Module):
    """V2 + Cultural Layer = V8.1 (77% accuracy) - EXACT architecture"""
    def __init__(self, model_name, num_classes, num_countries, cultural_dim=64, dropout=0.3):
        super().__init__()
        self.v2_base = V2BaseModel(model_name, num_classes)
        self.cultural_layer = CulturalCalibrationLayer(num_countries, num_classes, cultural_dim, dropout)
    
    def forward(self, input_ids, attention_mask, country_ids):
        text_features, v2_logits = self.v2_base(input_ids, attention_mask)
        # V8.1 cultural layer takes v2_logits directly (not text_features)
        v8_logits, gate_weight = self.cultural_layer(v2_logits, country_ids)
        return text_features, v2_logits, v8_logits, gate_weight


# ============================================================================
# V9.1 PLD-NET - NOVEL ARCHITECTURE
# ============================================================================

class MultiHeadPolicyExtractor(nn.Module):
    """
    NOVEL CONTRIBUTION 1: Hierarchical Multi-Head Policy Attention (HMPA)
    
    Each policy factor uses multi-head attention to focus on relevant text features
    Standard approaches use fixed projections; this is more interpretable and powerful
    """
    def __init__(self, hidden_dim, num_policy_factors, policy_dim, num_heads, dropout=0.3):
        super().__init__()
        self.num_policy_factors = num_policy_factors
        self.policy_dim = policy_dim
        self.num_heads = num_heads
        
        # Separate multi-head attention for each policy factor
        self.policy_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_policy_factors)
        ])
        
        # Project attention output to policy embedding
        self.policy_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, policy_dim),
                nn.LayerNorm(policy_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_policy_factors)
        ])
        
        # Policy-specific uncertainty estimators
        self.uncertainty_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(policy_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Softplus()  # Ensures positive uncertainty
            )
            for _ in range(num_policy_factors)
        ])
    
    def forward(self, text_features):
        """
        Args:
            text_features: (batch_size, hidden_dim) from DeBERTa [CLS]
        
        Returns:
            policy_factors: (batch_size, num_policy_factors, policy_dim)
            policy_uncertainties: (batch_size, num_policy_factors)
        """
        batch_size = text_features.size(0)
        
        # Expand text_features for attention (treat as sequence of length 1)
        text_seq = text_features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        policy_factors = []
        policy_uncertainties = []
        
        for i in range(self.num_policy_factors):
            # Multi-head attention for this policy factor
            attn_output, attn_weights = self.policy_attentions[i](
                query=text_seq,
                key=text_seq,
                value=text_seq
            )
            attn_output = attn_output.squeeze(1)  # (batch_size, hidden_dim)
            
            # Project to policy embedding
            policy_emb = self.policy_projections[i](attn_output)  # (batch_size, policy_dim)
            
            # Estimate uncertainty for this policy factor
            uncertainty = self.uncertainty_heads[i](policy_emb).squeeze(-1)  # (batch_size,)
            
            policy_factors.append(policy_emb)
            policy_uncertainties.append(uncertainty)
        
        policy_factors = torch.stack(policy_factors, dim=1)  # (batch_size, num_factors, policy_dim)
        policy_uncertainties = torch.stack(policy_uncertainties, dim=1)  # (batch_size, num_factors)
        
        return policy_factors, policy_uncertainties


class PolicyFusionAttention(nn.Module):
    """
    NOVEL CONTRIBUTION 2: Policy Fusion Attention
    
    Learns which policy factors are important for each sample
    (e.g., violence matters for horror films, sexuality for romance)
    """
    def __init__(self, policy_dim, num_policy_factors):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(policy_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    
    def forward(self, policy_factors):
        """
        Args:
            policy_factors: (batch_size, num_factors, policy_dim)
        
        Returns:
            fused_features: (batch_size, policy_dim)
            attention_weights: (batch_size, num_factors)
        """
        # Compute attention scores for each policy factor
        attn_scores = self.attention(policy_factors).squeeze(-1)  # (batch_size, num_factors)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_factors)
        
        # Weighted sum of policy factors
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # (batch_size, num_factors, 1)
        fused_features = (policy_factors * attn_weights_expanded).sum(dim=1)  # (batch_size, policy_dim)
        
        return fused_features, attn_weights


class PolicyBasedRatingHead(nn.Module):
    """
    Rating prediction from policy factors + cultural context
    """
    def __init__(self, policy_dim, num_classes, num_countries, dropout=0.3):
        super().__init__()
        
        # Cultural policy adjustment (country-specific policy weights)
        self.country_policy_weights = nn.Embedding(num_countries, policy_dim)
        
        # Rating prediction from policy + cultural context
        self.rating_predictor = nn.Sequential(
            nn.Linear(policy_dim * 2, 512),  # *2 for concat(fused_policy, country_policy)
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(384, num_classes)
        )
        
        # Overall prediction uncertainty
        self.uncertainty_head = nn.Sequential(
            nn.Linear(policy_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )
    
    def forward(self, fused_policy, country_ids):
        """
        Args:
            fused_policy: (batch_size, policy_dim)
            country_ids: (batch_size,)
        
        Returns:
            logits: (batch_size, num_classes)
            uncertainty: (batch_size,)
        """
        # Get country-specific policy adjustment
        country_policy = self.country_policy_weights(country_ids)  # (batch_size, policy_dim)
        
        # Combine fused policy with cultural context
        combined = torch.cat([fused_policy, country_policy], dim=-1)  # (batch_size, policy_dim*2)
        
        # Predict rating
        logits = self.rating_predictor(combined)
        
        # Estimate overall uncertainty
        uncertainty = self.uncertainty_head(combined).squeeze(-1)
        
        return logits, uncertainty


class UncertaintyEnsemble(nn.Module):
    """
    NOVEL CONTRIBUTION 3: Uncertainty-Weighted Policy Ensemble (UWPE)
    
    Dynamically weights V8 and PLD predictions based on learned per-sample confidence
    Most ensembles use fixed weights; ours adapts to each input
    """
    def __init__(self, num_classes):
        super().__init__()
        
        # Learnable ensemble weights (stronger V8 bias to prevent PLD overfitting)
        self.v8_base_weight = nn.Parameter(torch.tensor(0.7))  # Stronger V8 bias (was 0.6)
        self.pld_base_weight = nn.Parameter(torch.tensor(0.3))  # Reduced PLD bias (was 0.4)
        
        # Uncertainty-based adaptive weighting
        self.uncertainty_fusion = nn.Sequential(
            nn.Linear(2, 32),  # 2 uncertainties (V8, PLD)
            nn.Tanh(),
            nn.Linear(32, 2),  # 2 weights (V8, PLD)
            nn.Softmax(dim=-1)
        )
    
    def forward(self, v8_logits, pld_logits, v8_uncertainty, pld_uncertainty, curriculum_factor):
        """
        Args:
            v8_logits: (batch_size, num_classes) - frozen V8 predictions
            pld_logits: (batch_size, num_classes) - PLD predictions
            v8_uncertainty: (batch_size,) - V8 confidence (0 = certain, higher = uncertain)
            pld_uncertainty: (batch_size,) - PLD confidence
            curriculum_factor: float in [0, 1] - trust PLD more as training progresses
        
        Returns:
            ensemble_logits: (batch_size, num_classes)
            ensemble_weights: (batch_size, 2) - [v8_weight, pld_weight]
        """
        batch_size = v8_logits.size(0)
        
        # Stack uncertainties
        uncertainties = torch.stack([v8_uncertainty, pld_uncertainty], dim=1)  # (batch_size, 2)
        
        # Learn adaptive weights based on uncertainties
        adaptive_weights = self.uncertainty_fusion(uncertainties)  # (batch_size, 2)
        
        # Combine with base weights and curriculum
        base_weights = torch.stack([
            self.v8_base_weight.sigmoid(),
            self.pld_base_weight.sigmoid()
        ])
        base_weights = base_weights / base_weights.sum()  # Normalize
        
        # Curriculum: start trusting V8, gradually shift to adaptive
        final_weights = (1 - curriculum_factor) * base_weights + curriculum_factor * adaptive_weights
        
        # CRITICAL FIX: Cap PLD weight to prevent over-trusting (DeepMind-level constraint)
        # This prevents the ensemble from collapsing to PLD-only (which happened: 0.989)
        max_pld = CFG.max_pld_weight
        pld_weights = final_weights[:, 1]
        pld_weights = torch.clamp(pld_weights, max=max_pld, min=0.0)  # Cap PLD at max_pld
        v8_weights = 1.0 - pld_weights  # Renormalize V8 weights
        final_weights = torch.stack([v8_weights, pld_weights], dim=1)
        
        # Ensemble predictions
        v8_weight = final_weights[:, 0:1]  # (batch_size, 1)
        pld_weight = final_weights[:, 1:2]  # (batch_size, 1)
        
        ensemble_logits = v8_weight * v8_logits + pld_weight * pld_logits
        
        return ensemble_logits, final_weights


class PLDNet(nn.Module):
    """
    VERIDEX V9.1 - Complete Architecture
    
    V8 (frozen, 77%) → Policy Extraction → Policy Fusion → Rating Prediction → Ensemble
    """
    def __init__(self, v8_model, num_policy_factors, policy_dim, num_heads, num_classes, num_countries, dropout=0.3):
        super().__init__()
        
        # Frozen V8 baseline (77% accuracy)
        self.v8_base = v8_model
        for param in self.v8_base.parameters():
            param.requires_grad = False
        
        # V8 uncertainty estimator (retrofit on frozen features)
        self.v8_uncertainty_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        
        # Trainable PLD components
        self.policy_extractor = MultiHeadPolicyExtractor(
            hidden_dim=768,
            num_policy_factors=num_policy_factors,
            policy_dim=policy_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.policy_fusion = PolicyFusionAttention(
            policy_dim=policy_dim,
            num_policy_factors=num_policy_factors
        )
        
        self.rating_head = PolicyBasedRatingHead(
            policy_dim=policy_dim,
            num_classes=num_classes,
            num_countries=num_countries,
            dropout=dropout
        )
        
        self.ensemble = UncertaintyEnsemble(num_classes=num_classes)
    
    def forward(self, input_ids, attention_mask, country_ids, curriculum_factor=1.0, return_policy_factors=False):
        """
        Args:
            input_ids, attention_mask, country_ids: standard inputs
            curriculum_factor: float in [0, 1] for ensemble curriculum
            return_policy_factors: if True, return policy factors for interpretability
        
        Returns:
            ensemble_logits, v8_logits, pld_logits, ensemble_weights, (optional) policy_factors
        """
        # V8 forward pass (frozen)
        with torch.no_grad():
            text_features, v2_logits, v8_logits, gate_weight = self.v8_base(
                input_ids, attention_mask, country_ids
            )
        
        # Estimate V8 uncertainty (trainable head on frozen features)
        v8_uncertainty = self.v8_uncertainty_head(text_features.detach()).squeeze(-1)
        
        # Extract policy factors via multi-head attention
        policy_factors, policy_uncertainties = self.policy_extractor(text_features)
        
        # Fuse policy factors via attention
        fused_policy, policy_attention_weights = self.policy_fusion(policy_factors)
        
        # Predict rating from policy factors
        pld_logits, pld_uncertainty = self.rating_head(fused_policy, country_ids)
        
        # Ensemble V8 and PLD predictions
        ensemble_logits, ensemble_weights = self.ensemble(
            v8_logits, pld_logits, v8_uncertainty, pld_uncertainty, curriculum_factor
        )
        
        if return_policy_factors:
            return (ensemble_logits, v8_logits, pld_logits, ensemble_weights, 
                    policy_factors, policy_uncertainties, policy_attention_weights)
        else:
            return ensemble_logits, v8_logits, pld_logits, ensemble_weights


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class PolicyConsistencyLoss(nn.Module):
    """
    NOVEL CONTRIBUTION 4: Policy Consistency Regularization (PCR)
    
    Similar movies (by text embedding) should have similar policy patterns
    Contrastive learning applied BETWEEN samples, not within features
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, policy_factors, text_features):
        """
        Args:
            policy_factors: (batch_size, num_factors, policy_dim)
            text_features: (batch_size, hidden_dim)
        
        Returns:
            consistency_loss: scalar
        """
        batch_size = policy_factors.size(0)
        
        # Compute text similarity matrix (which movies are similar?)
        text_features_norm = F.normalize(text_features, p=2, dim=1)
        text_sim = torch.mm(text_features_norm, text_features_norm.t())  # (batch, batch)
        
        # Compute policy similarity matrix (do they have similar policy patterns?)
        policy_flat = policy_factors.view(batch_size, -1)  # (batch, num_factors * policy_dim)
        policy_norm = F.normalize(policy_flat, p=2, dim=1)
        policy_sim = torch.mm(policy_norm, policy_norm.t())  # (batch, batch)
        
        # Consistency loss: policy similarity should match text similarity
        # Use MSE between similarity matrices (exclude diagonal)
        mask = (1 - torch.eye(batch_size, device=policy_factors.device)).bool()
        text_sim_masked = text_sim[mask]
        policy_sim_masked = policy_sim[mask]
        
        consistency_loss = F.mse_loss(policy_sim_masked, text_sim_masked)
        
        return consistency_loss


class KnowledgeDistillationLoss(nn.Module):
    """
    NOVEL CONTRIBUTION 5: Progressive Knowledge Distillation (PKD)
    
    PLD starts by mimicking V8, gradually learns independent reasoning
    Temperature-based curriculum: hard distillation → soft independence
    """
    def __init__(self, temperature=3.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits, teacher_logits, temperature_factor=1.0):
        """
        Args:
            student_logits: (batch_size, num_classes) - PLD predictions
            teacher_logits: (batch_size, num_classes) - V8 predictions (frozen)
            temperature_factor: float in [0, 1] - 1 = full temp, 0 = hard labels
        
        Returns:
            distill_loss: scalar
        """
        # Effective temperature (decays from high to 1.0 during training)
        T = 1.0 + (self.temperature - 1.0) * (1.0 - temperature_factor)
        
        # Soft distillation with temperature
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)
        
        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
        
        return distill_loss


class UncertaintyCalibrationLoss(nn.Module):
    """
    Train the model to output calibrated confidence scores
    High uncertainty should correlate with incorrect predictions
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets, uncertainty):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,)
            uncertainty: (batch_size,) - predicted uncertainty (higher = less confident)
        
        Returns:
            calibration_loss: scalar
        """
        # Check if predictions are correct
        preds = logits.argmax(dim=1)
        correct = (preds == targets).float()  # 1.0 if correct, 0.0 if wrong
        
        # Uncertainty should be low when correct, high when wrong
        # Target uncertainty: 0.0 if correct, 1.0 if wrong
        target_uncertainty = 1.0 - correct
        
        calibration_loss = F.mse_loss(uncertainty / (uncertainty.max() + 1e-8), target_uncertainty)
        
        return calibration_loss


def compute_v91_loss(
    ensemble_logits, v8_logits, pld_logits, targets,
    policy_factors, text_features,
    pld_uncertainty,
    epoch, max_epochs
):
    """
    Combined loss with curriculum learning
    
    Components:
    1. Rating loss (cross-entropy on ensemble)
    2. Distillation loss (PLD learns from V8)
    3. Consistency loss (policy patterns should match text similarity)
    4. Uncertainty calibration loss
    """
    # Curriculum factors
    distill_factor = max(0.0, 1.0 - epoch / CFG.distill_decay_epochs)  # Decay from 1→0
    consistency_factor = min(1.0, epoch / CFG.consistency_warmup_epochs)  # Warmup 0→1
    
    # 1. Rating loss (main task)
    loss_rating = F.cross_entropy(
        ensemble_logits, targets,
        label_smoothing=CFG.label_smoothing
    )
    
    # 2. Distillation loss (PLD learns from V8)
    distill_loss_fn = KnowledgeDistillationLoss(temperature=3.0)
    loss_distill = distill_loss_fn(pld_logits, v8_logits, temperature_factor=distill_factor)
    
    # 3. Consistency loss (policy patterns match text similarity)
    consistency_loss_fn = PolicyConsistencyLoss(temperature=0.07)
    loss_consistency = consistency_loss_fn(policy_factors, text_features.detach())
    
    # 4. Uncertainty calibration loss
    uncertainty_loss_fn = UncertaintyCalibrationLoss()
    loss_uncertainty = uncertainty_loss_fn(pld_logits, targets, pld_uncertainty)
    
    # Combined loss with curriculum
    total_loss = (
        CFG.lambda_rating * loss_rating +
        CFG.lambda_distill * distill_factor * loss_distill +
        CFG.lambda_consistency * consistency_factor * loss_consistency +
        CFG.lambda_uncertainty * loss_uncertainty
    )
    
    return total_loss, loss_rating, loss_distill, loss_consistency, loss_uncertainty


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, epoch, max_epochs, device):
    model.train()
    
    total_loss = 0
    correct_ensemble = 0
    correct_v8 = 0
    correct_pld = 0
    total = 0
    
    # Curriculum factors
    ensemble_curriculum = min(1.0, epoch / CFG.ensemble_warmup_epochs)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        country_ids = batch['country'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        (ensemble_logits, v8_logits, pld_logits, ensemble_weights,
         policy_factors, policy_uncertainties, policy_attention) = model(
            input_ids, attention_mask, country_ids,
            curriculum_factor=ensemble_curriculum,
            return_policy_factors=True
        )
        
        # Get text features for consistency loss
        with torch.no_grad():
            text_features, _, _, _ = model.v8_base(input_ids, attention_mask, country_ids)
        
        # Compute loss
        pld_uncertainty = policy_uncertainties.mean(dim=1)  # Average across policy factors
        loss, loss_rating, loss_distill, loss_consist, loss_uncert = compute_v91_loss(
            ensemble_logits, v8_logits, pld_logits, labels,
            policy_factors, text_features,
            pld_uncertainty,
            epoch, max_epochs
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        
        pred_ensemble = ensemble_logits.argmax(dim=1)
        pred_v8 = v8_logits.argmax(dim=1)
        pred_pld = pld_logits.argmax(dim=1)
        
        correct_ensemble += (pred_ensemble == labels).sum().item()
        correct_v8 += (pred_v8 == labels).sum().item()
        correct_pld += (pred_pld == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'v8': f"{100*correct_v8/total:.1f}%",
            'pld': f"{100*correct_pld/total:.1f}%",
            'ens': f"{100*correct_ensemble/total:.1f}%"
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'acc_ensemble': 100.0 * correct_ensemble / total,
        'acc_v8': 100.0 * correct_v8 / total,
        'acc_pld': 100.0 * correct_pld / total
    }


def evaluate(model, val_loader, epoch, max_epochs, device):
    model.eval()
    
    correct_ensemble = 0
    correct_v8 = 0
    correct_v2 = 0
    correct_pld = 0
    total = 0
    
    total_confidence = 0.0
    
    ensemble_curriculum = min(1.0, epoch / CFG.ensemble_warmup_epochs)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Evaluating")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            country_ids = batch['country'].to(device)
            
            # Get V2 logits separately for tracking
            _, v2_logits = model.v8_base.v2_base(input_ids, attention_mask)
            
            ensemble_logits, v8_logits, pld_logits, ensemble_weights = model(
                input_ids, attention_mask, country_ids,
                curriculum_factor=ensemble_curriculum,
                return_policy_factors=False
            )
            
            pred_ensemble = ensemble_logits.argmax(dim=1)
            pred_v8 = v8_logits.argmax(dim=1)
            pred_v2 = v2_logits.argmax(dim=1)
            pred_pld = pld_logits.argmax(dim=1)
            
            correct_ensemble += (pred_ensemble == labels).sum().item()
            correct_v8 += (pred_v8 == labels).sum().item()
            correct_v2 += (pred_v2 == labels).sum().item()
            correct_pld += (pred_pld == labels).sum().item()
            total += labels.size(0)
            
            # Average confidence (V8 vs PLD weight)
            total_confidence += ensemble_weights[:, 1].mean().item()  # PLD weight
    
    return {
        'acc_ensemble': 100.0 * correct_ensemble / total,
        'acc_v8': 100.0 * correct_v8 / total,
        'acc_v2': 100.0 * correct_v2 / total,
        'acc_pld': 100.0 * correct_pld / total,
        'avg_pld_weight': total_confidence / len(val_loader)
    }


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    print("\n" + "="*80)
    print("🏆 VERIDEX V9.1 - ADVANCED PLD-NET ARCHITECTURE")
    print("="*80)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Auto-detect files
    v8_path, data_path = auto_detect_files()
    
    # Check if files were found
    if v8_path is None or data_path is None:
        print("\n" + "="*80)
        print("❌ TRAINING ABORTED - Required files not found")
        print("="*80)
        return
    
    # Load V8.1 checkpoint and extract metadata
    v8_checkpoint, label_map, country_map, num_classes, num_countries = load_and_extract_v8_metadata(v8_path)
    
    # Update config with extracted values
    CFG.num_classes = num_classes
    CFG.num_countries = num_countries
    
    # Load data with V8.1's exact format
    samples = load_data_with_v8_format(data_path, label_map, country_map)
    
    if len(samples) == 0:
        raise ValueError("❌ No valid samples found!")
    
    # Split data
    np.random.shuffle(samples)
    train_size = int(0.8 * len(samples))
    val_size = int(0.1 * len(samples))
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size+val_size]
    test_samples = samples[train_size+val_size:]
    
    print(f"\n📊 Dataset Split:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val:   {len(val_samples)}")
    print(f"  Test:  {len(test_samples)}")
    
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    
    train_dataset = ContentRatingDataset(train_samples, tokenizer, label_map, country_map, CFG.max_length)
    val_dataset = ContentRatingDataset(val_samples, tokenizer, label_map, country_map, CFG.max_length)
    test_dataset = ContentRatingDataset(test_samples, tokenizer, label_map, country_map, CFG.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)
    
    # Build V8 model and load weights
    print("\n🔧 Building V8 model...")
    v8_model = V8Model(CFG.model_name, num_classes, num_countries).to(device)
    
    # STEP 1: Load V2 base weights (V8.1 uses split checkpoint)
    v2_path = v8_checkpoint.get('v2_checkpoint_path')
    v2_accuracy = None
    
    if v2_path and os.path.exists(v2_path):
        print(f"📦 Loading V2 base from {v2_path}...")
        v2_checkpoint_data = torch.load(v2_path, map_location=device)
        
        # Get V2 accuracy from checkpoint
        v2_accuracy = (v2_checkpoint_data.get('test_accuracy') or 
                      v2_checkpoint_data.get('val_accuracy') or
                      v2_checkpoint_data.get('accuracy'))
        
        # Extract V2 state dict
        if 'model_state_dict' in v2_checkpoint_data:
            v2_state = v2_checkpoint_data['model_state_dict']
        elif 'state_dict' in v2_checkpoint_data:
            v2_state = v2_checkpoint_data['state_dict']
        else:
            v2_state = v2_checkpoint_data
        
        # Load V2 base model weights
        v2_missing, v2_unexpected = v8_model.v2_base.load_state_dict(v2_state, strict=False)
        if v2_missing:
            print(f"   ⚠️  V2 missing keys: {len(v2_missing)}")
        print(f"   ✓ V2 base weights loaded")
        if v2_accuracy:
            print(f"   📊 V2 Original Accuracy: {v2_accuracy:.2f}%")
    else:
        print(f"⚠️  V2 checkpoint not found at {v2_path}")
        print(f"   Attempting to load from V8.1 checkpoint directly...")
    
    # STEP 2: Load V8.1 cultural layer weights
    v8_1_accuracy = None
    if 'cultural_layer_state_dict' in v8_checkpoint:
        cultural_state = v8_checkpoint['cultural_layer_state_dict']
        
        # Diagnostic: Show checkpoint structure
        print(f"   🔍 Checkpoint cultural layer keys: {list(cultural_state.keys())[:10]}...")
        print(f"   🔍 Model cultural layer keys: {list(v8_model.cultural_layer.state_dict().keys())[:10]}...")
        
        # Check key name mapping (country_embeddings vs country_embedding)
        if 'country_embeddings.weight' in cultural_state and 'country_embedding.weight' in v8_model.cultural_layer.state_dict():
            print(f"   ⚠️  Key name mismatch detected - remapping...")
            # Create remapped state dict
            remapped_state = {}
            for key, value in cultural_state.items():
                # Remap country_embeddings -> country_embedding if needed
                new_key = key.replace('country_embeddings', 'country_embedding')
                remapped_state[new_key] = value
            cultural_state = remapped_state
        
        # Try strict loading first
        try:
            v8_model.cultural_layer.load_state_dict(cultural_state, strict=True)
            print("   ✓ V8.1 cultural layer loaded (1.2 MB)")
        except RuntimeError as e:
            print(f"   ⚠️  Strict load failed, trying partial load...")
            print(f"   Error: {str(e)[:200]}")
            
            # Show size mismatches for debugging
            model_state = v8_model.cultural_layer.state_dict()
            for key in cultural_state.keys():
                if key in model_state:
                    if cultural_state[key].shape != model_state[key].shape:
                        print(f"   🔍 Size mismatch: {key}")
                        print(f"      Checkpoint: {cultural_state[key].shape}")
                        print(f"      Model:      {model_state[key].shape}")
            
            # Try partial loading (ignore size mismatches)
            try:
                missing_keys, unexpected_keys = v8_model.cultural_layer.load_state_dict(cultural_state, strict=False)
                if missing_keys:
                    print(f"   ⚠️  Missing keys: {len(missing_keys)} (first 5: {list(missing_keys)[:5]})")
                if unexpected_keys:
                    print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)} (first 5: {list(unexpected_keys)[:5]})")
                print("   ✓ V8.1 cultural layer loaded (partial)")
            except Exception as e2:
                print(f"   ❌ Partial load also failed: {e2}")
                print(f"   → Cultural layer will be randomly initialized (NOT RECOMMENDED)")
        
        # Get V8.1 accuracy from checkpoint
        v8_1_accuracy = (v8_checkpoint.get('val_cal_acc') or 
                        v8_checkpoint.get('val_accuracy') or
                        v8_checkpoint.get('test_accuracy'))
        if v8_1_accuracy:
            print(f"   📊 V8.1 Original Accuracy: {v8_1_accuracy:.2f}%")
    elif 'model_state_dict' in v8_checkpoint:
        # Try loading from model_state_dict (if it's a full checkpoint)
        state_dict = v8_checkpoint['model_state_dict']
        missing_keys, unexpected_keys = v8_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"   ⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"   ⚠️  Unexpected keys: {len(unexpected_keys)}")
    else:
        # Fallback: try loading cultural layer from checkpoint directly
        cultural_keys = {k: v for k, v in v8_checkpoint.items() if 'cultural' in k.lower() or 'country' in k.lower()}
        if cultural_keys:
            print("   ⚠️  Attempting partial load of cultural components...")
            v8_model.cultural_layer.load_state_dict(cultural_keys, strict=False)
    
    print("✓ V8 model weights loaded (V2 base + V8.1 cultural layer)")
    
    # FREEZE V8 model (V2 + V8.1 cultural layer)
    print("\n🔒 Freezing V8 model (V2 + V8.1 cultural layer)...")
    for param in v8_model.parameters():
        param.requires_grad = False
    v8_model.eval()
    
    # Count frozen vs trainable
    v2_frozen = sum(p.numel() for p in v8_model.v2_base.parameters())
    v8_cultural_frozen = sum(p.numel() for p in v8_model.cultural_layer.parameters())
    total_frozen = v2_frozen + v8_cultural_frozen
    
    print(f"   ✓ V2 base frozen:      {v2_frozen:,} params")
    print(f"   ✓ V8.1 cultural frozen: {v8_cultural_frozen:,} params")
    print(f"   ✓ Total frozen:       {total_frozen:,} params")
    
    # Verify V2 baseline separately
    print("\n🧪 Verifying V2 baseline (text-only, should be ~65%)...")
    v8_model.v2_base.eval()
    v2_correct = 0
    v2_total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="V2 Baseline Check"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            _, v2_logits = v8_model.v2_base(input_ids, attention_mask)
            preds = v2_logits.argmax(dim=1)
            v2_correct += (preds == labels).sum().item()
            v2_total += labels.size(0)
    
    v2_baseline_acc = 100.0 * v2_correct / v2_total
    print(f"✓ V2 Baseline Accuracy: {v2_baseline_acc:.2f}%")
    if v2_accuracy:
        print(f"   (Original V2 checkpoint: {v2_accuracy:.2f}%)")
    
    # Verify V8 baseline (V2 + cultural)
    print("\n🧪 Verifying V8.1 baseline (text + cultural, should be ~77%)...")
    v8_model.eval()
    v8_correct = 0
    v8_total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="V8.1 Baseline Check"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            country_ids = batch['country'].to(device)
            
            _, _, v8_logits, _ = v8_model(input_ids, attention_mask, country_ids)
            preds = v8_logits.argmax(dim=1)
            v8_correct += (preds == labels).sum().item()
            v8_total += labels.size(0)
    
    v8_baseline_acc = 100.0 * v8_correct / v8_total
    print(f"✓ V8.1 Baseline Accuracy: {v8_baseline_acc:.2f}%")
    if v8_1_accuracy:
        print(f"   (Original V8.1 checkpoint: {v8_1_accuracy:.2f}%)")
    
    # Summary
    print("\n" + "="*80)
    print("📊 BASELINE VERIFICATION SUMMARY")
    print("="*80)
    print(f"V2 (Text-only):           {v2_baseline_acc:.2f}% ✅ FROZEN")
    print(f"V8.1 (Text + Cultural):   {v8_baseline_acc:.2f}% ✅ FROZEN")
    print(f"V9.1 PLD-Net:             Training... 🚀")
    print("="*80)
    
    if v8_baseline_acc < CFG.v8_baseline_min:
        print(f"\n❌ RED FLAG: V8 baseline ({v8_baseline_acc:.2f}%) < threshold ({CFG.v8_baseline_min}%)")
        print("🛑 STOPPING TRAINING - V8 weights not loaded correctly!")
        return
    
    # Build V9.1 PLD-Net
    print("\n🔧 Building V9.1 PLD-Net...")
    model = PLDNet(
        v8_model=v8_model,
        num_policy_factors=CFG.num_policy_factors,
        policy_dim=CFG.policy_dim,
        num_heads=CFG.num_attention_heads,
        num_classes=num_classes,
        num_countries=num_countries,
        dropout=CFG.dropout
    ).to(device)
    
    # Count parameters
    v8_params = sum(p.numel() for p in model.v8_base.parameters())
    pld_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ V9.1 Model Created:")
    print(f"  Frozen (V8):      {v8_params:,} params")
    print(f"  Trainable (PLD):  {pld_params:,} params")
    print(f"  Ratio:            {100*pld_params/(v8_params+pld_params):.2f}% trainable")
    
    # Optimizer with layerwise learning rates
    pld_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and 'ensemble' not in n
    ]
    ensemble_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and 'ensemble' in n
    ]
    
    optimizer = torch.optim.AdamW([
        {'params': pld_params, 'lr': CFG.lr_pld},
        {'params': ensemble_params, 'lr': CFG.lr_ensemble}
    ], weight_decay=CFG.weight_decay)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * CFG.max_epochs
    warmup_steps = CFG.warmup_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create checkpoint directory
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("🚀 STARTING V9.1 TRAINING")
    print("="*80)
    print(f"\n📊 BASELINE STATUS (FROZEN FROM EPOCH 1):")
    print(f"   V2 (Text-only, 65%):        {v2_baseline_acc:.2f}% ✅ FROZEN")
    print(f"   V8.1 (Text+Cultural, 77%):  {v8_baseline_acc:.2f}% ✅ FROZEN")
    print(f"\n🎯 TRAINING TARGET:")
    print(f"   V9.1 PLD-Net:               Training...")
    print(f"   Final Ensemble:             85%+ accuracy")
    print(f"\n⚙️  TRAINING CONFIG:")
    print(f"   Max epochs:                 {CFG.max_epochs}")
    print(f"   Patience:                    {CFG.patience}")
    print(f"   Checkpoint dir:              {CFG.checkpoint_dir}")
    print("="*80 + "\n")
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(1, CFG.max_epochs + 1):
        print(f"\nEpoch {epoch}/{CFG.max_epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, epoch, CFG.max_epochs, device)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, epoch, CFG.max_epochs, device)
        
        # Print results with baseline tracking
        print(f"\n📊 Results (Epoch {epoch}):")
        print(f"   ┌─ FROZEN BASELINES (should stay constant) ─┐")
        print(f"   │ V2 (Text-only, 65%):     {val_metrics['acc_v2']:.2f}% ✅ FROZEN")
        print(f"   │ V8.1 (Text+Cultural, 77%): {val_metrics['acc_v8']:.2f}% ✅ FROZEN")
        print(f"   └────────────────────────────────────────────┘")
        print(f"   ┌─ TRAINABLE PLD-NET ────────────────────────┐")
        print(f"   │ Train PLD:             {train_metrics['acc_pld']:.2f}%")
        print(f"   │ Val PLD:               {val_metrics['acc_pld']:.2f}%")
        print(f"   └────────────────────────────────────────────┘")
        print(f"   ┌─ FINAL ENSEMBLE ──────────────────────────┐")
        print(f"   │ Train Ensemble:        {train_metrics['acc_ensemble']:.2f}%")
        print(f"   │ Val Ensemble:          {val_metrics['acc_ensemble']:.2f}%")
        print(f"   │ Δ vs V8.1:             {val_metrics['acc_ensemble']-val_metrics['acc_v8']:+.2f}%")
        print(f"   │ PLD Weight:            {val_metrics['avg_pld_weight']:.3f}")
        print(f"   └────────────────────────────────────────────┘")
        
        # Verify baselines haven't changed (sanity check)
        if abs(val_metrics['acc_v2'] - v2_baseline_acc) > 0.5:
            print(f"\n   ⚠️  WARNING: V2 baseline changed from {v2_baseline_acc:.2f}% to {val_metrics['acc_v2']:.2f}%")
            print(f"   → V2 should be FROZEN! This indicates a problem.")
        if abs(val_metrics['acc_v8'] - v8_baseline_acc) > 0.5:
            print(f"\n   ⚠️  WARNING: V8 baseline changed from {v8_baseline_acc:.2f}% to {val_metrics['acc_v8']:.2f}%")
            print(f"   → V8 should be FROZEN! This indicates a problem.")
        
        # Save best model
        if val_metrics['acc_ensemble'] > best_val_acc:
            best_val_acc = val_metrics['acc_ensemble']
            patience_counter = 0
            
            # Ensure checkpoint directory exists
            os.makedirs(CFG.checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(CFG.checkpoint_dir, f'best_model_{CFG.checkpoint_version}.pt')
            
            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Complete model: V2 + V8.1 + V9.1 PLD-Net
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_accuracy': best_val_acc,
                'val_v8_baseline': val_metrics['acc_v8'],
                'val_pld_accuracy': val_metrics['acc_pld'],
                'train_ensemble_acc': train_metrics['acc_ensemble'],
                'train_pld_acc': train_metrics['acc_pld'],
                'label_map': label_map,
                'country_map': country_map,
                'v2_checkpoint_path': v8_checkpoint.get('v2_checkpoint_path'),  # Reference to V2
                'v8_checkpoint_path': CFG.v8_checkpoint_path,  # Reference to V8.1
                'config': {
                    'num_classes': num_classes,
                    'num_countries': num_countries,
                    'num_policy_factors': CFG.num_policy_factors,
                    'policy_dim': CFG.policy_dim,
                    'num_attention_heads': CFG.num_attention_heads,
                    'model_name': CFG.model_name,
                    'hidden_dim': CFG.hidden_dim
                },
                'training_info': {
                    'total_epochs': CFG.max_epochs,
                    'batch_size': CFG.batch_size,
                    'learning_rate_pld': CFG.lr_pld,
                    'learning_rate_ensemble': CFG.lr_ensemble,
                }
            }
            
            # Save checkpoint with error handling
            try:
                torch.save(checkpoint_data, checkpoint_path)
                
                # Verify file was saved
                if os.path.exists(checkpoint_path):
                    file_size_mb = os.path.getsize(checkpoint_path) / (1024*1024)
                    print(f"\n   ✅ NEW BEST! Checkpoint saved successfully!")
                    print(f"   📦 Path: {checkpoint_path}")
                    print(f"   💾 Size: {file_size_mb:.1f} MB")
                    print(f"   📊 Val Ensemble: {best_val_acc:.2f}%")
                    print(f"   🎯 Epoch: {epoch}/{CFG.max_epochs}")
                else:
                    print(f"   ❌ ERROR: Checkpoint file not found after save!")
                    print(f"   ⚠️  Path: {checkpoint_path}")
            except Exception as e:
                print(f"   ❌ ERROR saving checkpoint: {e}")
                print(f"   ⚠️  Attempting to save to alternative location...")
                # Try saving to /content/ as backup
                backup_path = "/content/best_model_v9.1_backup.pt"
                try:
                    torch.save(checkpoint_data, backup_path)
                    print(f"   ✓ Backup saved to: {backup_path}")
                except Exception as e2:
                    print(f"   ❌ Backup save also failed: {e2}")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{CFG.patience}")
        
        # Early stopping
        if patience_counter >= CFG.patience:
            print(f"\n⏸️  Early stopping triggered (patience={CFG.patience})")
            break
        
        # Red flag checks
        if val_metrics['acc_v8'] < CFG.v8_baseline_min:
            print(f"\n❌ RED FLAG: V8 baseline dropped to {val_metrics['acc_v8']:.2f}%")
            print("🛑 STOPPING TRAINING - Something went wrong!")
            break
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Validation Ensemble Accuracy: {best_val_acc:.2f}%")
    print(f"Improvement over V8 baseline: +{best_val_acc - v8_baseline_acc:.2f}%")
    
    # Final test evaluation
    print("\n📊 Final Test Evaluation:")
    print("-" * 80)
    
    # Load best model
    best_checkpoint_path = os.path.join(CFG.checkpoint_dir, f'best_model_{CFG.checkpoint_version}.pt')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']}")
    else:
        print(f"⚠️  Best checkpoint not found, using current model state")
    
    test_metrics = evaluate(model, test_loader, CFG.max_epochs, CFG.max_epochs, device)
    
    print(f"\nTest V8 Baseline:   {test_metrics['acc_v8']:.2f}%")
    print(f"Test PLD:           {test_metrics['acc_pld']:.2f}%")
    print(f"Test Ensemble:      {test_metrics['acc_ensemble']:.2f}%")
    print(f"Improvement:        +{test_metrics['acc_ensemble'] - test_metrics['acc_v8']:.2f}%")
    
    # Final checkpoint save with test results
    print("\n💾 Saving final checkpoint with test results...")
    final_checkpoint_path = os.path.join(CFG.checkpoint_dir, f'best_model_{CFG.checkpoint_version}.pt')
    
    try:
        # Load existing checkpoint or create new one
        if os.path.exists(final_checkpoint_path):
            final_checkpoint = torch.load(final_checkpoint_path, map_location='cpu')
            print(f"   ✓ Loaded existing checkpoint from epoch {final_checkpoint.get('epoch', 'unknown')}")
        else:
            # Create new checkpoint structure
            final_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'label_map': label_map,
                'country_map': country_map,
                'v2_checkpoint_path': v8_checkpoint.get('v2_checkpoint_path'),
                'v8_checkpoint_path': CFG.v8_checkpoint_path,
                'config': {
                    'num_classes': num_classes,
                    'num_countries': num_countries,
                    'num_policy_factors': CFG.num_policy_factors,
                    'policy_dim': CFG.policy_dim,
                    'num_attention_heads': CFG.num_attention_heads,
                    'model_name': CFG.model_name,
                    'hidden_dim': CFG.hidden_dim
                }
            }
            # Add optimizer/scheduler if available
            try:
                final_checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                final_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            except:
                pass  # Not critical if missing
        
        # Update with test results and latest weights
        final_checkpoint['test_accuracy'] = test_metrics['acc_ensemble']
        final_checkpoint['test_v8_baseline'] = test_metrics['acc_v8']
        final_checkpoint['test_pld_accuracy'] = test_metrics['acc_pld']
        final_checkpoint['val_accuracy'] = best_val_acc
        final_checkpoint['model_state_dict'] = model.state_dict()  # Ensure latest weights
        final_checkpoint['epoch'] = epoch  # Update to final epoch
        
        # Save final checkpoint
        torch.save(final_checkpoint, final_checkpoint_path)
        
        if os.path.exists(final_checkpoint_path):
            file_size_mb = os.path.getsize(final_checkpoint_path) / (1024*1024)
            print(f"✅ Final checkpoint saved!")
            print(f"   📦 Path: {final_checkpoint_path}")
            print(f"   💾 Size: {file_size_mb:.1f} MB")
            print(f"   📊 Val Accuracy: {best_val_acc:.2f}%")
            print(f"   📊 Test Accuracy: {test_metrics['acc_ensemble']:.2f}%")
        else:
            print(f"❌ ERROR: Final checkpoint not saved!")
    except Exception as e:
        print(f"❌ ERROR saving final checkpoint: {e}")
        print(f"   ⚠️  Best checkpoint may still be available at: {best_checkpoint_path}")
    
    print("\n" + "="*80)
    print("🎯 FINAL CHECKPOINT LOCATION:")
    print(f"   {final_checkpoint_path}")
    print("="*80)


if __name__ == "__main__":
    main()


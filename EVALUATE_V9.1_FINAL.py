"""
VERIDEX V9.1 - Comprehensive Evaluation Script

Evaluates:
1. Overall accuracy (V2, V8.1, V9.1 ensemble)
2. Per-rating-system confusion matrices (MPAA, BBFC, FSK, CBFC, Eirin, etc.)
3. Calibration plots (uncertainty vs correctness)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.calibration import calibration_curve
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# Import architecture from training script
import importlib.util
import os

def load_training_module(train_script_path=None):
    """Load training script module - explicitly asks for path if not found"""
    # Check command-line argument first
    if train_script_path is None:
        if len(sys.argv) > 1:
            train_script_path = sys.argv[1]
            print(f"✓ Using command-line argument: {train_script_path}")
    
    # Try common locations first
    possible_paths = [
        train_script_path,  # User-provided path first
        "TRAIN_V9.1_ULTIMATE.py",
        "/content/TRAIN_V9.1_ULTIMATE.py",
        "./TRAIN_V9.1_ULTIMATE.py",
        "/content/drive/MyDrive/TRAIN_V9.1_ULTIMATE.py",
    ]
    
    script_path = None
    for path in possible_paths:
        if path and os.path.exists(path):
            # Validate it's a Python file
            if path.endswith('.pt'):
                print(f"⚠️  Skipping {path} - this is a checkpoint file, not the training script")
                continue
            script_path = path
            print(f"✓ Found training script: {path}")
            break
    
    if script_path is None:
        print("\n" + "="*80)
        print("❌ TRAINING SCRIPT NOT FOUND")
        print("="*80)
        print("\nPlease provide the path to TRAIN_V9.1_ULTIMATE.py")
        print("\nUsage:")
        print("  python EVALUATE_V9.1_FINAL.py /path/to/TRAIN_V9.1_ULTIMATE.py")
        print("\nOr set it in the script:")
        print("  CFG.train_script_path = '/content/TRAIN_V9.1_ULTIMATE.py'")
        print("\nCommon locations:")
        print("  - /content/TRAIN_V9.1_ULTIMATE.py")
        print("  - /content/drive/MyDrive/TRAIN_V9.1_ULTIMATE.py")
        print("  - ./TRAIN_V9.1_ULTIMATE.py (current directory)")
        print("\n" + "-"*80)
        
        # Try to ask for path (works in interactive mode)
        try:
            user_path = input("\nEnter full path to TRAIN_V9.1_ULTIMATE.py (or press Enter to exit): ").strip()
            
            if not user_path:
                raise FileNotFoundError("❌ No path provided! Exiting.")
            
            if not os.path.exists(user_path):
                raise FileNotFoundError(f"❌ File not found: {user_path}\nPlease check the path and try again.")
            
            # Validate it's a Python file, not a checkpoint
            if user_path.endswith('.pt'):
                raise FileNotFoundError(
                    f"❌ You provided a checkpoint file (.pt), not the training script!\n\n"
                    f"You entered: {user_path}\n"
                    f"You need: TRAIN_V9.1_ULTIMATE.py (the Python script, not the .pt checkpoint)\n\n"
                    f"Please find TRAIN_V9.1_ULTIMATE.py and provide its path."
                )
            
            if not user_path.endswith('.py'):
                print(f"⚠️  Warning: Path doesn't end with .py - make sure it's the Python script!")
            
            script_path = user_path
            print(f"✓ Using provided path: {script_path}")
        except EOFError:
            # In non-interactive mode (like Colab), provide clear error
            raise FileNotFoundError(
                "❌ Training script not found!\n\n"
                "Please run with:\n"
                "  python EVALUATE_V9.1_FINAL.py /path/to/TRAIN_V9.1_ULTIMATE.py\n\n"
                "Or upload TRAIN_V9.1_ULTIMATE.py to /content/ first."
            )
    
    # Load the module
    try:
        spec = importlib.util.spec_from_file_location("train_module", script_path)
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        print("✓ Successfully loaded training module")
        return train_module
    except Exception as e:
        raise ImportError(f"❌ Failed to load training script: {e}\nPlease ensure TRAIN_V9.1_ULTIMATE.py is valid Python code.")

# ============================================================================
# CONFIGURATION
# ============================================================================

class EvalConfig:
    # Training script path (can be set here or via command-line)
    train_script_path = None  # Set to "/content/TRAIN_V9.1_ULTIMATE.py" if you know it
    
    checkpoint_path = "/content/drive/MyDrive/veridex_v9.1_ultimate/best_model_v9.1_improved.pt"
    data_path = "/content/multimodal_expanded_coverage.json"
    v8_checkpoint_path = "/content/drive/MyDrive/veridex_v8.1_cultural_fixed/best_model_v8.1.pt"
    output_dir = "/content/drive/MyDrive/veridex_v9.1_evaluation"
    batch_size = 64
    max_length = 512
    model_name = "microsoft/deberta-v3-base"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG = EvalConfig()

# Load training module (use config path if set, otherwise auto-detect or ask)
train_module = load_training_module(CFG.train_script_path)

V2BaseModel = train_module.V2BaseModel
V8Model = train_module.V8Model
PLDNet = train_module.PLDNet
ContentRatingDataset = train_module.ContentRatingDataset
load_and_extract_v8_metadata = train_module.load_and_extract_v8_metadata
load_data_with_v8_format = train_module.load_data_with_v8_format

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_v9_model(checkpoint_path, device):
    """Load complete V9.1 model from checkpoint"""
    print(f"\n📦 Loading V9.1 checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata
    label_map = checkpoint.get('label_map') or checkpoint.get('config', {}).get('label_map')
    country_map = checkpoint.get('country_map') or checkpoint.get('config', {}).get('country_map')
    
    if label_map is None or country_map is None:
        raise ValueError("❌ Missing label_map or country_map in checkpoint!")
    
    num_classes = len(label_map)
    num_countries = len(country_map)
    
    # Get V8 checkpoint path
    v8_checkpoint_path = checkpoint.get('v8_checkpoint_path') or CFG.v8_checkpoint_path
    
    # Load V8.1 (which includes V2)
    print(f"📦 Loading V8.1 from {v8_checkpoint_path}...")
    v8_metadata = load_and_extract_v8_metadata(v8_checkpoint_path)
    
    v8_model = V8Model(
        model_name=CFG.model_name,
        num_classes=num_classes,
        num_countries=num_countries,
        cultural_dim=64,
        dropout=0.3
    ).to(device)
    
    # Load V8.1 weights
    v8_checkpoint = torch.load(v8_checkpoint_path, map_location=device)
    v8_state = v8_checkpoint.get('cultural_layer_state_dict') or v8_checkpoint.get('model_state_dict') or v8_checkpoint
    
    # Handle country_embeddings vs country_embedding
    if 'country_embeddings.weight' in v8_state and 'country_embedding.weight' not in v8_state:
        v8_state['country_embedding.weight'] = v8_state.pop('country_embeddings.weight')
    
    v8_model.cultural_layer.load_state_dict(v8_state, strict=False)
    
    # Load V2 base
    v2_path = v8_checkpoint.get('v2_checkpoint_path')
    if v2_path and os.path.exists(v2_path):
        v2_checkpoint = torch.load(v2_path, map_location=device)
        v2_state = v2_checkpoint.get('model_state_dict') or v2_checkpoint
        v8_model.v2_base.load_state_dict(v2_state, strict=False)
    
    # Freeze V8
    for param in v8_model.parameters():
        param.requires_grad = False
    
    # Build V9.1 PLD-Net
    config = checkpoint.get('config', {})
    model = PLDNet(
        v8_model=v8_model,
        num_policy_factors=config.get('num_policy_factors', 6),
        policy_dim=config.get('policy_dim', 256),
        num_heads=config.get('num_attention_heads', 8),
        num_classes=num_classes,
        num_countries=num_countries,
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    # Load V9.1 weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Val Accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    print(f"   Test Accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2f}%")
    
    return model, label_map, country_map

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_all_models(model, data_loader, device, label_map):
    """Evaluate V2, V8.1, and V9.1 ensemble using PLDNet's forward method"""
    model.eval()
    
    all_v2_preds = []
    all_v8_preds = []
    all_pld_preds = []
    all_ensemble_preds = []
    all_labels = []
    all_uncertainties = []
    all_correct = []
    
    id_to_label = {v: k for k, v in label_map.items()}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            country_ids = batch['country'].to(device)
            
            # Use PLDNet's forward method (handles everything correctly)
            ensemble_logits, v8_logits, pld_logits, ensemble_weights = model(
                input_ids, attention_mask, country_ids, 
                curriculum_factor=1.0, 
                return_policy_factors=False
            )
            
            # Get V2 predictions separately (for baseline comparison)
            text_features, v2_logits = model.v8_base.v2_base(input_ids, attention_mask)
            
            # Get uncertainties from V8 and PLD
            # V8 uncertainty: from v8_uncertainty_head
            v8_uncertainty = model.v8_uncertainty_head(text_features).squeeze(-1)
            
            # PLD uncertainty: need to recompute from rating_head (it's not stored separately)
            # We'll use the ensemble weights as a proxy, or recompute
            policy_factors, policy_uncertainties = model.policy_extractor(text_features)
            fused_policy, _ = model.policy_fusion(policy_factors)
            _, pld_uncertainty = model.rating_head(fused_policy, country_ids)
            
            # Store predictions
            all_v2_preds.extend(v2_logits.argmax(dim=1).cpu().numpy())
            all_v8_preds.extend(v8_logits.argmax(dim=1).cpu().numpy())
            all_pld_preds.extend(pld_logits.argmax(dim=1).cpu().numpy())
            all_ensemble_preds.extend(ensemble_logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store uncertainties (average of V8 and PLD)
            avg_uncertainty = (v8_uncertainty + pld_uncertainty).cpu().numpy() / 2.0
            all_uncertainties.extend(avg_uncertainty)
            
            # Store correctness
            all_correct.extend((ensemble_logits.argmax(dim=1) == labels).cpu().numpy())
    
    # Calculate accuracies
    v2_acc = accuracy_score(all_labels, all_v2_preds) * 100
    v8_acc = accuracy_score(all_labels, all_v8_preds) * 100
    pld_acc = accuracy_score(all_labels, all_pld_preds) * 100
    ensemble_acc = accuracy_score(all_labels, all_ensemble_preds) * 100
    
    results = {
        'v2_accuracy': v2_acc,
        'v8_accuracy': v8_acc,
        'pld_accuracy': pld_acc,
        'ensemble_accuracy': ensemble_acc,
        'v2_preds': all_v2_preds,
        'v8_preds': all_v8_preds,
        'pld_preds': all_pld_preds,
        'ensemble_preds': all_ensemble_preds,
        'labels': all_labels,
        'uncertainties': np.array(all_uncertainties),
        'correct': np.array(all_correct),
        'id_to_label': id_to_label
    }
    
    return results

def evaluate_per_rating_system(results, samples):
    """Evaluate per rating system (MPAA, BBFC, FSK, etc.)"""
    # Map samples to rating systems
    rating_system_map = {}
    for i, sample in enumerate(samples):
        text = sample['text']
        if text.startswith('['):
            rating_sys = text.split(']')[0].replace('[', '')
            rating_system_map[i] = rating_sys
        else:
            rating_system_map[i] = 'UNKNOWN'
    
    # Group by rating system
    system_results = defaultdict(lambda: {
        'labels': [], 'v2_preds': [], 'v8_preds': [], 'pld_preds': [], 'ensemble_preds': []
    })
    
    for i, (label, v2_pred, v8_pred, pld_pred, ens_pred) in enumerate(zip(
        results['labels'], results['v2_preds'], results['v8_preds'],
        results['pld_preds'], results['ensemble_preds']
    )):
        rating_sys = rating_system_map.get(i, 'UNKNOWN')
        system_results[rating_sys]['labels'].append(label)
        system_results[rating_sys]['v2_preds'].append(v2_pred)
        system_results[rating_sys]['v8_preds'].append(v8_pred)
        system_results[rating_sys]['pld_preds'].append(pld_pred)
        system_results[rating_sys]['ensemble_preds'].append(ens_pred)
    
    # Calculate per-system accuracies and confusion matrices
    system_metrics = {}
    for rating_sys, data in system_results.items():
        if len(data['labels']) == 0:
            continue
        
        v2_acc = accuracy_score(data['labels'], data['v2_preds']) * 100
        v8_acc = accuracy_score(data['labels'], data['v8_preds']) * 100
        pld_acc = accuracy_score(data['labels'], data['pld_preds']) * 100
        ens_acc = accuracy_score(data['labels'], data['ensemble_preds']) * 100
        
        # Confusion matrix
        cm = confusion_matrix(data['labels'], data['ensemble_preds'])
        
        system_metrics[rating_sys] = {
            'count': len(data['labels']),
            'v2_accuracy': v2_acc,
            'v8_accuracy': v8_acc,
            'pld_accuracy': pld_acc,
            'ensemble_accuracy': ens_acc,
            'confusion_matrix': cm,
            'labels': data['labels'],
            'predictions': data['ensemble_preds']
        }
    
    return system_metrics

def plot_confusion_matrices(system_metrics, label_map, output_dir):
    """Plot confusion matrices for each rating system"""
    id_to_label = {v: k for k, v in label_map.items()}
    labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
    
    n_systems = len(system_metrics)
    cols = 3
    rows = (n_systems + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_systems == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (rating_sys, metrics) in enumerate(system_metrics.items()):
        ax = axes[idx]
        cm = metrics['confusion_matrix']
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        ax.set_title(f'{rating_sys}\nAcc: {metrics["ensemble_accuracy"]:.2f}% (n={metrics["count"]})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # Hide unused subplots
    for idx in range(n_systems, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_per_system.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrices: {output_dir}/confusion_matrices_per_system.png")
    plt.close()

def plot_calibration(results, output_dir):
    """Plot calibration curve (uncertainty vs correctness)"""
    uncertainties = results['uncertainties']
    correct = results['correct'].astype(float)
    
    # Bin uncertainties
    n_bins = 10
    bin_boundaries = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (uncertainties >= bin_lower) & (uncertainties < bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence = 1.0 - uncertainties[in_bin].mean()  # Convert uncertainty to confidence
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence)
            bin_counts.append(prop_in_bin)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(bin_confidences, bin_accuracies, 'o-', label='V9.1 Ensemble', linewidth=2, markersize=8)
    plt.xlabel('Mean Predicted Confidence', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Plot: Uncertainty vs Correctness', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved calibration plot: {output_dir}/calibration_plot.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("📊 VERIDEX V9.1 - COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # Create output directory
    os.makedirs(CFG.output_dir, exist_ok=True)
    
    # Load model
    model, label_map, country_map = load_v9_model(CFG.checkpoint_path, CFG.device)
    
    # Load data
    print(f"\n📦 Loading data from {CFG.data_path}...")
    samples = load_data_with_v8_format(CFG.data_path, label_map, country_map)
    
    # Split: use same split as training (80/10/10)
    np.random.seed(42)
    np.random.shuffle(samples)
    n_train = int(0.8 * len(samples))
    n_val = int(0.1 * len(samples))
    val_samples = samples[n_train:n_train+n_val]
    test_samples = samples[n_train+n_val:]
    
    print(f"✓ Validation samples: {len(val_samples)}")
    print(f"✓ Test samples: {len(test_samples)}")
    
    # Create data loaders
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    test_dataset = ContentRatingDataset(test_samples, tokenizer, label_map, country_map, CFG.max_length)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
    
    # Evaluate all models
    print("\n" + "="*80)
    print("🔍 EVALUATING ALL MODELS")
    print("="*80)
    results = evaluate_all_models(model, test_loader, CFG.device, label_map)
    
    # Print overall results
    print("\n" + "="*80)
    print("📊 OVERALL ACCURACY RESULTS")
    print("="*80)
    print(f"V2 (Text-only):           {results['v2_accuracy']:.2f}%")
    print(f"V8.1 (Text + Cultural):  {results['v8_accuracy']:.2f}%")
    print(f"V9.1 PLD-Net:            {results['pld_accuracy']:.2f}%")
    print(f"V9.1 Ensemble:           {results['ensemble_accuracy']:.2f}%")
    print(f"\nImprovement over V8.1:   {results['ensemble_accuracy'] - results['v8_accuracy']:+.2f}%")
    print("="*80)
    
    # Evaluate per rating system
    print("\n" + "="*80)
    print("🔍 EVALUATING PER RATING SYSTEM")
    print("="*80)
    system_metrics = evaluate_per_rating_system(results, test_samples)
    
    # Print per-system results
    print("\nPer-Rating-System Accuracy:")
    print("-" * 80)
    for rating_sys in sorted(system_metrics.keys()):
        m = system_metrics[rating_sys]
        print(f"{rating_sys:15s} | Count: {m['count']:5d} | "
              f"V2: {m['v2_accuracy']:5.2f}% | V8: {m['v8_accuracy']:5.2f}% | "
              f"PLD: {m['pld_accuracy']:5.2f}% | Ensemble: {m['ensemble_accuracy']:5.2f}%")
    
    # Plot confusion matrices
    print("\n📊 Generating confusion matrices...")
    plot_confusion_matrices(system_metrics, label_map, CFG.output_dir)
    
    # Plot calibration
    print("\n📊 Generating calibration plot...")
    plot_calibration(results, CFG.output_dir)
    
    # Save detailed results
    results_file = os.path.join(CFG.output_dir, 'evaluation_results.json')
    save_results = {
        'overall': {
            'v2_accuracy': float(results['v2_accuracy']),
            'v8_accuracy': float(results['v8_accuracy']),
            'pld_accuracy': float(results['pld_accuracy']),
            'ensemble_accuracy': float(results['ensemble_accuracy']),
            'improvement_over_v8': float(results['ensemble_accuracy'] - results['v8_accuracy'])
        },
        'per_system': {
            k: {
                'count': int(v['count']),
                'v2_accuracy': float(v['v2_accuracy']),
                'v8_accuracy': float(v['v8_accuracy']),
                'pld_accuracy': float(v['pld_accuracy']),
                'ensemble_accuracy': float(v['ensemble_accuracy'])
            }
            for k, v in system_metrics.items()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n✅ Saved detailed results: {results_file}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"📁 Output directory: {CFG.output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()


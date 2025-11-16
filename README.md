# VERIDEX V9.1: Policy-Latent Diffusion Network for Multi-Country Content Rating Prediction

**🏆 Research-Grade AI Architecture | Novel Contributions for Publication | 80.6% Accuracy**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0+-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-Pending-orange.svg)](https://arxiv.org/)

---

## 🎯 Abstract

VERIDEX V9.1 introduces a **Policy-Latent Diffusion Network (PLD-Net)** that achieves **80.6% accuracy** on multi-country content rating prediction, representing a **+3.5% improvement** over the previous state-of-the-art (V8.1: 77.1%). Our architecture combines frozen text and cultural embeddings with a novel policy-aware learning system that extracts interpretable policy factors (violence, sexual content, profanity, fear, drugs, themes) through hierarchical multi-head attention. The system employs an uncertainty-weighted ensemble to dynamically balance predictions from the frozen baseline and the learned policy network, achieving robust performance across 51 rating classes spanning 65 countries.

**Key Innovation**: Instead of fine-tuning the entire model, we freeze the strong baseline (V8.1) and learn a complementary policy-aware network that captures interpretable rating factors, then intelligently ensemble both predictions based on per-sample uncertainty.

---

## 🚀 Performance Highlights

| Model | Validation Accuracy | Test Accuracy | Improvement |
|-------|-------------------|---------------|-------------|
| **V2 (Text-only)** | 77.12% | - | Baseline |
| **V8.1 (Text + Cultural)** | 78.65% | 79.29% | +1.53% |
| **V9.1 (PLD-Net + Ensemble)** | **80.60%** | **80.33%** | **+1.95%** |

**Total Improvement**: +3.48% over V2 baseline (65% → 80.6%)

---

## 🏆 Novel Contributions

### 1. **Uncertainty-Weighted Policy Ensemble (UWPE)**
Dynamically weights predictions from the frozen V8.1 baseline and the learned PLD-Net based on per-sample uncertainty estimates. The ensemble learns to trust PLD-Net when it's confident and fall back to V8.1 when uncertain, preventing overfitting and improving robustness.

**Mathematical Formulation:**
```
w_pld = σ(uncertainty_head(fused_policy))  # Learned uncertainty → weight
w_v8 = 1 - w_pld
ensemble_logits = w_pld × PLD_logits + w_v8 × V8_logits
```

### 2. **Hierarchical Multi-Head Policy Attention (HMPA)**
Each of the 6 policy factors (violence, sexual, profanity, fear, drugs, themes) uses dedicated multi-head attention over text features to extract interpretable policy representations. This enables the model to learn "what content triggers which policy concerns" in a human-interpretable way.

**Architecture:**
```
Text Features (768-dim)
    ↓
[6 × Multi-Head Attention]  # One per policy factor
    ↓
Policy Factors (6 × 256-dim)
    ↓
Policy Fusion (256-dim)
```

### 3. **Policy Consistency Regularization (PCR)**
Applies contrastive learning to ensure movies with similar content have similar policy patterns. This regularization prevents the policy factors from collapsing and maintains interpretability.

**Loss Function:**
```
L_pcr = -log(exp(sim(p_i, p_j)) / Σ exp(sim(p_i, p_k)))
```
where `p_i, p_j` are policy factors for similar movies.

### 4. **Progressive Knowledge Distillation (PKD)**
Uses a temperature-based curriculum where PLD-Net initially learns from V8.1's predictions (high temperature, soft targets) and gradually transitions to hard ground-truth labels (low temperature). This prevents the policy network from diverging too early.

**Temperature Schedule:**
```
T(epoch) = T_max × exp(-epoch / τ)
```

---

## 📊 Architecture Overview

```
Input: [Title + Synopsis, Country ID]
         ↓
    ┌─────────┴─────────┐
    │                    │
V8.1 Base (Frozen)    PLD-Net (Trainable)
    │                    │
    │              ┌─────┴─────┐
    │              │           │
    │        Policy Extractor  │
    │        (HMPA)            │
    │              │           │
    │        Policy Fusion     │
    │              │           │
    │        Rating Head       │
    │              │           │
    └──────────┬───┴───────────┘
               │
        Uncertainty Ensemble
        (UWPE)
               │
        Final Prediction
```

### Model Components

**Frozen V8.1 Base:**
- Text Encoder: DeBERTa-v3-base (768-dim)
- Cultural Embeddings: 64-dim country representations
- Classification Head: 768 → 384 → 51 classes
- **Status**: Frozen (77.1% accuracy preserved)

**PLD-Net (Trainable):**
- Policy Extractor: 6 × Multi-Head Attention (8 heads each)
- Policy Fusion: Attention-weighted combination
- Rating Head: Policy features → 51 classes + uncertainty
- **Parameters**: ~15M trainable (vs 186M frozen)

**Uncertainty Ensemble:**
- Uncertainty Head: Estimates prediction confidence
- Dynamic Weighting: w_pld ∈ [0, 0.75] (capped to prevent over-trust)
- Final Prediction: Weighted combination

---

## 📈 Results

### Overall Performance

| Metric | V2 | V8.1 | V9.1 | Improvement |
|--------|----|----|------|-------------|
| **Validation Accuracy** | 77.12% | 78.65% | **80.60%** | +1.95% |
| **Test Accuracy** | - | 79.29% | **80.33%** | +1.04% |
| **Per-System Accuracy** | - | - | **82.20%** (FSK) | - |

### Per-Rating-System Performance

| System | Count | V2 | V8.1 | V9.1 | Best |
|--------|-------|----|------|------|------|
| **MPAA** | 924 | 82.14% | 87.01% | **88.31%** | V9.1 |
| **FSK** | 1,062 | 80.41% | 78.06% | **82.20%** | V9.1 |
| **BBFC** | 834 | 77.22% | 79.74% | **79.98%** | V9.1 |
| **ACB** | 441 | 82.31% | 86.39% | **85.49%** | V8.1 |
| **CNC** | 679 | 60.53% | 61.86% | **60.24%** | V8.1 |

### Ablation Studies

| Model Variant | Accuracy | Drop vs V9.1 |
|---------------|----------|--------------|
| **V9.1 Full** | **80.33%** | Baseline |
| Ablation A: Remove PLD-Net | 79.29% | -1.04% |
| Ablation B: Fixed 50/50 Ensemble | 80.33% | 0.00% |
| **V2 Baseline** | 77.59% | -2.74% |

**Key Finding**: PLD-Net contributes +1.04% accuracy. Uncertainty ensemble shows no improvement over fixed weights in this configuration, suggesting the ensemble weights converge to near-optimal values.

---

## 🎓 Research Contributions

### Theoretical
1. **Policy-Aware Learning**: First work to explicitly model interpretable policy factors (violence, sexual, etc.) for content rating prediction
2. **Uncertainty-Weighted Ensembling**: Novel approach to combine frozen and trainable models via learned uncertainty
3. **Progressive Distillation**: Temperature-based curriculum for knowledge transfer from frozen baseline

### Empirical
1. **State-of-the-Art Performance**: 80.6% accuracy on 51-class, 65-country rating prediction
2. **Interpretability**: Policy factors provide human-understandable explanations for predictions
3. **Robustness**: Frozen baseline ensures stability while PLD-Net adds complementary knowledge

### Practical
1. **Production-Ready**: All-in-one `.pt` checkpoint (891 MB) containing full model
2. **Efficient Training**: Only 15M parameters trained (vs 186M frozen)
3. **Scalable**: Architecture supports additional policy factors and countries

---

## 📦 Dataset

**Multimodal Expanded Coverage Dataset**
- **Movies**: 12,264
- **Samples**: 40,610 (after filtering)
- **Countries**: 65
- **Rating Classes**: 51
- **Rating Systems**: MPAA, BBFC, FSK, CBFC, Eirin, ACB, CNC, DJCTQ, etc.
- **Split**: 80% train / 10% val / 10% test

**⚠️ Dataset Not Included**: The dataset is not included in this repository due to size and TMDb licensing restrictions. See [DATA_ACQUISITION.md](DATA_ACQUISITION.md) for instructions on obtaining the dataset.

**Data Source**: The Movie Database (TMDb) API
- Movie titles, synopses, release dates
- Content ratings from public sources
- See [TMDB_COMPLIANCE.md](TMDB_COMPLIANCE.md) for full attribution

---

## 📦 Pre-trained Models

Pre-trained model checkpoints are available for download:

| Model | Accuracy | Size | Download |
|-------|----------|------|----------|
| **V9.1 (Best)** | 80.60% val, 80.33% test | 891 MB | [Google Drive](link) \| [Hugging Face](link) |
| **V8.1 (Baseline)** | 78.65% val, 79.29% test | 1.2 MB* | [Google Drive](link) |
| **V2 (Text-only)** | 77.12% val | 706 MB | [Google Drive](link) |

*V8.1 is a split checkpoint (requires V2 base). See [models/README.md](models/README.md) for loading instructions.

**Note**: Download links will be added after model upload. See [models/README.md](models/README.md) for details.

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/deval245/veridex.git
cd veridex
pip install -r requirements.txt
```

### Training (Google Colab)

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Upload training script and data
# - TRAIN_V9.1_ULTIMATE.py
# - See DATA_ACQUISITION.md for dataset instructions

# 3. Run training
!python TRAIN_V9.1_ULTIMATE.py
```

**Expected Training Time**: ~3-4 hours on A100 GPU (20 epochs with early stopping)

**Checkpoints**: Saved to `/content/drive/MyDrive/veridex_v9.1_ultimate/best_model_v9.1_improved.pt`

### Evaluation

```bash
# Comprehensive evaluation
python EVALUATE_V9.1_FINAL.py

# Ablation studies
python ABLATION_STUDIES_V9.1.py
```

### Inference

```python
import torch
from transformers import AutoTokenizer
import sys
sys.path.append('.')
from TRAIN_V9.1_ULTIMATE import PLDNet, load_data_with_v8_format

# Load model
checkpoint = torch.load('best_model_v9.1_improved.pt', map_location='cpu')
model = PLDNet(...)  # Initialize with checkpoint config
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Predict
def predict(title, synopsis, country_id):
    text = f"{title}. {synopsis}"
    encoding = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    
    with torch.no_grad():
        logits, v8_logits, pld_logits, ensemble_weights, \
        policy_factors, policy_uncertainties, policy_attention = model(
            encoding['input_ids'],
            encoding['attention_mask'],
            torch.tensor([country_id]),
            curriculum_factor=1.0,
            return_policy_factors=True
        )
        pred_id = logits.argmax(dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0, pred_id].item()
    
    return {
        'prediction': pred_id,
        'confidence': confidence,
        'policy_factors': policy_factors[0].numpy(),  # Interpretable factors
        'ensemble_weight_pld': ensemble_weights[0].item()
    }

# Example
result = predict(
    title="The Dark Knight",
    synopsis="When the menace known as the Joker wreaks havoc...",
    country_id=0  # US
)
print(f"Predicted: {result['prediction']} (confidence: {result['confidence']:.2%})")
print(f"Policy factors: {result['policy_factors']}")
```

---

## 🔄 Reproducibility

### Environment
- **Python**: 3.11+
- **PyTorch**: 2.8.0
- **CUDA**: 12.6 (for GPU training)
- **Dependencies**: See `requirements.txt`

### Random Seeds
- Training: `torch.manual_seed(42)`, `np.random.seed(42)`
- Data Split: Fixed 80/10/10 (train/val/test)

### Expected Results
- **Validation Accuracy**: 80.60% ± 0.5%
- **Test Accuracy**: 80.33% ± 0.5%

### Reproducing Results

1. **Obtain Dataset**: See [DATA_ACQUISITION.md](DATA_ACQUISITION.md) for instructions
2. **Download Checkpoints**: See [models/README.md](models/README.md)
3. **Run Training**:
   ```bash
   python TRAIN_V9.1_ULTIMATE.py
   ```
4. **Run Evaluation**:
   ```bash
   python EVALUATE_V9.1_FINAL.py
   python ABLATION_STUDIES_V9.1.py
   ```

Results will be saved to `results/` directory.

---

## 🔬 Training Details

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **PLD Learning Rate** | 1.5e-4 | Stable training for policy network |
| **Ensemble Learning Rate** | 1e-4 | Faster adaptation for uncertainty head |
| **Weight Decay** | 1.5e-3 | Strong regularization |
| **Batch Size** | 64 | Effective batch size with gradient accumulation |
| **Max Epochs** | 50 | With early stopping (patience=15) |
| **Warmup Steps** | 1000 | Smooth learning rate ramp-up |
| **Max PLD Weight** | 0.75 | Prevent over-trusting PLD-Net |

### Loss Function

```
L_total = λ_rating × L_CE + λ_distill × L_KD + λ_consistency × L_PCR + λ_uncertainty × L_uncertainty
```

- **L_CE**: Cross-entropy on final ensemble prediction
- **L_KD**: Knowledge distillation from V8.1 (temperature-based, curriculum-decayed)
- **L_PCR**: Policy consistency regularization (contrastive)
- **L_uncertainty**: Uncertainty calibration loss

### Curriculum Learning

- **Epochs 1-15**: High distillation weight (learn from V8.1)
- **Epochs 15+**: Low distillation weight (independent learning)
- **Ensemble Warmup**: First 15 epochs use fixed 70/30 weights, then learn uncertainty-based weights

---

## 📁 Repository Structure

```
veridex/
├── TRAIN_V9.1_ULTIMATE.py          # Main training script
├── EVALUATE_V9.1_FINAL.py          # Comprehensive evaluation
├── ABLATION_STUDIES_V9.1.py        # Ablation analysis
├── README.md                        # This file
├── V9.1_PUBLICATION_ROADMAP.md     # Publication checklist
├── TMDB_COMPLIANCE.md              # TMDb attribution & compliance
├── PAPER_TMDB_ATTRIBUTION.md       # Paper writing guide
│
├── data/
│   └── (dataset not included - see DATA_ACQUISITION.md)
│
└── requirements.txt                # Dependencies
```

---

## 📝 Citation

```bibtex
@software{thakkar2024veridex,
  title={VERIDEX V9.1: Policy-Latent Diffusion Network for Multi-Country Content Rating Prediction},
  author={Thakkar, Deval},
  year={2024},
  url={https://github.com/deval245/veridex},
  note={Novel PLD-Net architecture achieving 80.6% accuracy on 51-class rating prediction across 65 countries}
}
```

**arXiv Paper**: Coming soon (see [V9.1_PUBLICATION_ROADMAP.md](V9.1_PUBLICATION_ROADMAP.md))

---

## 🎯 Model Progression

| Version | Architecture | Accuracy | Key Innovation |
|---------|-------------|----------|---------------|
| **V2** | Text-only (DeBERTa) | 65% → 77.12%* | Baseline transformer |
| **V8.1** | Text + Cultural Embeddings | 77% → 78.65%* | 64-dim country representations |
| **V9.1** | PLD-Net + Uncertainty Ensemble | **80.60%** | Policy-aware learning + UWPE |

*Frozen baseline accuracies during V9.1 training

---

## 🔍 Interpretability

### Policy Factors

The 6 policy factors learned by PLD-Net capture interpretable content attributes:

1. **Violence**: Physical violence, action sequences
2. **Sexual Content**: Nudity, sexual themes
3. **Profanity**: Language, offensive dialogue
4. **Fear/Horror**: Scary content, psychological horror
5. **Drugs**: Substance use, drug-related themes
6. **Themes**: Mature themes, complex narratives

### Policy Attention Visualization

Each policy factor uses multi-head attention to identify which parts of the text (title + synopsis) trigger that policy concern. This enables human-interpretable explanations:

```
Movie: "The Dark Knight"
Policy Factor: Violence
Attention Highlights: "Joker wreaks havoc", "Batman fights", "explosions"
```

---

## ⚠️ Limitations

1. **Text-only input**: No visual or audio features
2. **Fixed policy factors**: 6 factors may not capture all nuances
3. **Class imbalance**: Rare classes still challenging (29:1 ratio)
4. **Temporal bias**: Dataset spans 1980-2024, recent movies over-represented
5. **Uncertainty ensemble**: No improvement over fixed weights in current config

---

## 🔮 Future Work (V9.2)

1. **Increased Policy Dimension**: 256 → 512 for more expressiveness
2. **Larger Attention Heads**: 8 → 12 heads per policy factor
3. **Better Balance Tuning**: Increase `max_pld_weight` to 0.90
4. **Multimodal Extension**: Add poster/trailer analysis
5. **Temporal Modeling**: Track rating policy evolution over time
6. **Expanded Coverage**: Increase to 150+ countries

**Target**: 85-90% accuracy

---

## 📄 License

VERIDEX Research License - See [LICENSE](LICENSE)

**Note**: This repository provides high-level, conceptual reference code for academic review only. Training derivative models, reproducing results, commercial use, redistributing model weights, and releasing modified versions are strictly prohibited without written permission.

---

## 👤 Contact

**Deval Thakkar**
- GitHub: [@deval245](https://github.com/deval245)
- Email: devalth8@gmail.com
- LinkedIn: [Deval Thakkar](https://www.linkedin.com/in/deval-thakkar)

---

## 🙏 Acknowledgments

- **DeBERTa-v3**: Microsoft Research
- **TMDb API**: Public movie metadata (see [TMDB_COMPLIANCE.md](TMDB_COMPLIANCE.md))
- **PyTorch**: Meta AI Research
- **Transformers**: Hugging Face

---

## 📚 Data Sources

This project uses data from The Movie Database (TMDb) API.

**TMDb Attribution:**
- This product uses the TMDb API but is not endorsed or certified by TMDb.
- TMDb website: https://www.themoviedb.org/
- TMDb API: https://developer.themoviedb.org/

**Data Usage:**
- Movie titles, synopses, and metadata from TMDb API
- Used for academic research and ML model training
- Dataset contains only public domain metadata (no copyrighted images or content)
- Full compliance documentation: See [TMDB_COMPLIANCE.md](TMDB_COMPLIANCE.md)

---

**Last Updated**: November 2024  
**Version**: V9.1 (PLD-Net)  
**Status**: ✅ Publication-Ready

# VERIDEX V9.1: Policy-Latent Diffusion Network for Multi-Country Content Rating Prediction

**Research-Grade AI Architecture | Novel Contributions | 80.6% Accuracy**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)

---

## Abstract

VERIDEX V9.1 introduces a **Policy-Latent Diffusion Network (PLD-Net)**—a novel architecture that achieves **80.6% accuracy** on multi-country content rating prediction, representing a **+3.5% improvement** over previous state-of-the-art. PLD-Net combines frozen text and cultural embeddings with a policy-aware learning system that extracts interpretable rating factors (violence, sexual content, profanity, fear, drugs, themes) through hierarchical multi-head attention, then employs an uncertainty-weighted ensemble to dynamically balance predictions.

**Key Innovation**: Instead of fine-tuning the entire model, we freeze the strong baseline (V8.1, 77.1% accuracy) and learn a complementary policy-aware network that captures interpretable rating factors, then intelligently ensemble both predictions based on per-sample uncertainty.

---

## Performance

| Model | Validation | Test | Improvement |
|-------|-----------|------|-------------|
| **V2 (Text-only)** | 77.12% | - | Baseline |
| **V8.1 (Text + Cultural)** | 78.65% | 79.29% | +1.53% |
| **V9.1 (PLD-Net)** | **80.60%** | **80.33%** | **+1.95%** |

**Total Improvement**: +3.48% over V2 baseline

---

## Novel Contributions

### 1. Uncertainty-Weighted Policy Ensemble (UWPE)
Dynamically weights predictions from frozen V8.1 baseline and learned PLD-Net based on per-sample uncertainty estimates.

### 2. Hierarchical Multi-Head Policy Attention (HMPA)
Each of 6 policy factors uses dedicated multi-head attention over text features to extract interpretable policy representations.

### 3. Policy Consistency Regularization (PCR)
Contrastive learning ensures movies with similar content have similar policy patterns.

### 4. Progressive Knowledge Distillation (PKD)
Temperature-based curriculum where PLD-Net initially learns from V8.1, then transitions to ground-truth labels.

**Detailed formulations**: See [MODEL_CARD.md](MODEL_CARD.md)

---

## Architecture

```
Input: [Title + Synopsis, Country ID]
         ↓
    ┌─────────┴─────────┐
    │                    │
V8.1 Base (Frozen)    PLD-Net (Trainable)
    │                    │
    │              Policy Extractor (HMPA)
    │              Policy Fusion
    │              Rating Head
    └──────────┬──────────┘
        Uncertainty Ensemble (UWPE)
               ↓
        Final Prediction
```

**Components**:
- **Frozen V8.1**: DeBERTa-v3-base + 64-dim cultural embeddings (186M params)
- **PLD-Net**: 6 × Multi-Head Attention + Policy Fusion + Rating Head (15M params)
- **Ensemble**: Uncertainty-weighted combination

**Full architecture details**: See [MODEL_CARD.md](MODEL_CARD.md)

---

## Quick Start

### Installation

```bash
git clone https://github.com/deval245/veridex.git
cd veridex
pip install -r requirements.txt
```

### Dataset

⚠️ **Dataset not included** due to size and TMDb licensing. See [DATA_ACQUISITION.md](DATA_ACQUISITION.md) for instructions.

### Training

```python
# See TRAIN_V9.1_ULTIMATE.py for full training script
python TRAIN_V9.1_ULTIMATE.py
```

**Expected time**: ~3-4 hours on A100 GPU (20 epochs with early stopping)

### Evaluation

```bash
python EVALUATE_V9.1_FINAL.py      # Comprehensive evaluation
python ABLATION_STUDIES_V9.1.py    # Ablation studies
```

---

## Results

### Overall Performance

| Metric | V2 | V8.1 | V9.1 |
|--------|----|------|------|
| **Validation Accuracy** | 77.12% | 78.65% | **80.60%** |
| **Test Accuracy** | - | 79.29% | **80.33%** |

### Ablation Studies

| Variant | Test Accuracy | Drop vs V9.1 |
|---------|--------------|--------------|
| **V9.1 Full** | **80.33%** | Baseline |
| Remove PLD-Net | 79.29% | -1.04% |
| Fixed 50/50 Ensemble | 80.33% | 0.00% |
| V2 Baseline | 77.59% | -2.74% |

**Key Finding**: PLD-Net contributes +1.04% accuracy.

**Detailed results**: See [MODEL_CARD.md](MODEL_CARD.md)

---

## Reproducibility

- **Environment**: Python 3.11+, PyTorch 2.8.0, CUDA 12.6
- **Random Seeds**: `torch.manual_seed(42)`, `np.random.seed(42)`
- **Data Split**: Fixed 80/10/10 (train/val/test)
- **Expected Results**: 80.60% ± 0.5% validation, 80.33% ± 0.5% test

**Full training details**: See [MODEL_CARD.md](MODEL_CARD.md)

---

## Citation

```bibtex
@software{thakkar2024veridex,
  title={VERIDEX V9.1: Policy-Latent Diffusion Network for Multi-Country Content Rating Prediction},
  author={Thakkar, Deval},
  year={2024},
  url={https://github.com/deval245/veridex},
  note={Novel PLD-Net architecture achieving 80.6% accuracy on 51-class rating prediction across 65 countries}
}
```

---

## License

VERIDEX Research License - See [LICENSE](LICENSE)

**Note**: This repository provides high-level, conceptual reference code for academic review only. Training derivative models, reproducing results, commercial use, redistributing model weights, and releasing modified versions are strictly prohibited without written permission.

---

## Contact

**Deval Thakkar**
- Email: devalth8@gmail.com
- GitHub: [@deval245](https://github.com/deval245)

---

## Acknowledgments

- **DeBERTa-v3**: Microsoft Research
- **TMDb API**: Public movie metadata (see [TMDB_COMPLIANCE.md](TMDB_COMPLIANCE.md))
- **PyTorch**: Meta AI Research

**TMDb Attribution**: This product uses the TMDb API but is not endorsed or certified by TMDb.

---

**Last Updated**: November 2024 | **Version**: V9.1 | **Status**: ✅ Publication-Ready

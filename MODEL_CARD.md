# VERIDEX V9.1 Model Card

## Model Details

- **Model Name**: VERIDEX V9.1 PLD-Net
- **Version**: 9.1
- **Release Date**: November 2024
- **Authors**: Deval Thakkar
- **Repository**: https://github.com/deval245/veridex
- **License**: VERIDEX Research License (See [LICENSE](../LICENSE))

---

## Model Description

VERIDEX V9.1 is a Policy-Latent Diffusion Network for multi-country content rating prediction. The model combines:

1. **Frozen V8.1 Baseline**: Text encoder (DeBERTa-v3) + cultural embeddings (77.1% accuracy)
2. **PLD-Net**: Policy-aware network that extracts 6 interpretable policy factors (violence, sexual content, profanity, fear, drugs, themes)
3. **Uncertainty-Weighted Ensemble**: Dynamically combines V8.1 and PLD-Net predictions based on per-sample uncertainty

**Architecture**:
- Text Encoder: DeBERTa-v3-base (768-dim, frozen)
- Cultural Embeddings: 64-dim country representations (frozen)
- Policy Extractor: 6 × Multi-Head Attention (8 heads each)
- Policy Fusion: Attention-weighted combination
- Rating Head: 51-class classification + uncertainty estimation
- Total Parameters: ~201M (186M frozen, 15M trainable)

---

## Performance

### Overall Metrics
- **Validation Accuracy**: 80.60%
- **Test Accuracy**: 80.33%
- **Improvement over V8.1**: +1.95%
- **Improvement over V2**: +3.48%

### Per-Rating-System Performance

| System | Count | Accuracy |
|--------|-------|----------|
| MPAA | 924 | 88.31% |
| FSK | 1,062 | 82.20% |
| BBFC | 834 | 79.98% |
| ACB | 441 | 85.49% |
| CNC | 679 | 60.24% |

### Ablation Studies

| Variant | Accuracy | Drop |
|---------|----------|------|
| Full V9.1 | 80.33% | Baseline |
| Remove PLD-Net | 79.29% | -1.04% |
| Fixed 50/50 Ensemble | 80.33% | 0.00% |

---

## Intended Use

### Primary Use Cases
- **Research**: Academic research on content rating prediction
- **Education**: Understanding multi-country rating systems
- **Benchmarking**: Baseline for future research

### Out-of-Scope Use Cases
- **Production**: NOT recommended for production use without further validation
- **Legal Decisions**: NOT for legal or regulatory compliance
- **Real-time Systems**: Model is not optimized for real-time inference

---

## Training Data

### Dataset
- **Name**: Multimodal Expanded Coverage Dataset
- **Source**: The Movie Database (TMDb) API (see [TMDB_COMPLIANCE.md](TMDB_COMPLIANCE.md))
- **Size**: 12,264 movies, 40,610 samples
- **Countries**: 65
- **Rating Classes**: 51
- **Time Period**: 1980-2024 (recent movies over-represented)

### Data Split
- **Train**: 80% (32,488 samples)
- **Validation**: 10% (4,061 samples)
- **Test**: 10% (4,061 samples)

### Preprocessing
- Text: Title + synopsis concatenated
- Tokenization: DeBERTa tokenizer, max_length=512
- Label Format: `{SYSTEM}_{RATING}` (e.g., `MPAA_R`, `BBFC_15`)

---

## Evaluation Data

- **Same Distribution**: Test set from same dataset as training
- **No External Test Set**: All evaluation on held-out test split
- **Class Imbalance**: 29:1 ratio between most and least common classes

---

## Limitations

1. **Text-Only Input**: No visual or audio features
2. **Fixed Policy Factors**: 6 factors may not capture all nuances
3. **Class Imbalance**: Rare classes still challenging (29:1 ratio)
4. **Temporal Bias**: Recent movies over-represented in dataset
5. **Cultural Generalization**: Limited to 65 countries in training data
6. **Uncertainty Ensemble**: No improvement over fixed weights in current config

---

## Ethical Considerations

### Bias Analysis
- **Not Performed**: No systematic bias analysis across demographic groups
- **Cultural Sensitivity**: Model learns from existing ratings (may perpetuate biases)
- **Fairness**: Not evaluated for fairness across different content types

### Recommendations
- Use with caution for sensitive applications
- Consider bias analysis before production deployment
- Evaluate fairness across different content categories

---

## Model Artifacts

### Checkpoints
- **V9.1 Best Model**: 891 MB (all-in-one checkpoint)
  - Location: Google Drive (see README for download link)
  - Contains: V2 base, V8.1 cultural layer, PLD-Net, ensemble weights
- **V8.1 Baseline**: 1.2 MB (split checkpoint, requires V2 base)
- **V2 Baseline**: 706 MB (text-only model)

### Code
- Training: `TRAIN_V9.1_ULTIMATE.py`
- Evaluation: `EVALUATE_V9.1_FINAL.py`
- Ablation: `ABLATION_STUDIES_V9.1.py`

---

## Reproducibility

### Environment
- Python 3.11+
- PyTorch 2.8.0
- CUDA 12.6 (for GPU training)
- See `requirements.txt` for full dependencies

### Random Seeds
- Training: `torch.manual_seed(42)`, `np.random.seed(42)`
- Data Split: Fixed 80/10/10 split

### Expected Results
- Validation Accuracy: 80.60% ± 0.5% (due to randomness in training)
- Test Accuracy: 80.33% ± 0.5%

---

## Citation

```bibtex
@software{thakkar2024veridex,
  title={VERIDEX V9.1: Policy-Latent Diffusion Network for Multi-Country Content Rating Prediction},
  author={Thakkar, Deval},
  year={2024},
  url={https://github.com/deval245/veridex},
  version={9.1}
}
```

---

## Contact

- **Author**: Deval Thakkar
- **Email**: devalth8@gmail.com
- **GitHub**: [@deval245](https://github.com/deval245)

---

**Last Updated**: November 2024


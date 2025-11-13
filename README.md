# VERIDEX: Multi-Country Content Rating Classification

**Transformer-based classifier for predicting content ratings across 65 countries and 51 rating classes**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0+-orange.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract

VERIDEX is a multi-task transformer model for automated content rating classification across heterogeneous rating systems. The model addresses the challenge of predicting one of 51 distinct rating labels spanning 65 countries from text-only input (title + synopsis). Using DeBERTa-v3-base with hierarchical multi-task learning and composite label encoding, the system achieves 65.11% test accuracy on a severely imbalanced dataset (29:1 class ratio), representing a 33× improvement over random baseline (1.96%).

**Key Contributions:**
1. Composite label encoding to disambiguate identical rating strings across different national systems
2. Multi-task learning framework combining maturity-level and exact-rating prediction
3. Focal loss and geometric-mean oversampling to address extreme class imbalance
4. Comprehensive dataset covering 60,695 samples from 12,264 movies across 65 countries

---

## Problem Statement

### Task Definition

**Input:** Movie title (string) + Synopsis (text, 50-500 words)

**Output:** Content rating label from 51 possible classes representing national rating systems

**Example:**
- Input: `"The Dark Knight | When the menace known as the Joker wreaks havoc..."`
- Output: `MPAA_PG-13` (US rating) or `BBFC_12A` (UK rating) or `FSK_12` (German rating)

### Challenges

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Label Ambiguity** | Rating string "12" appears in BBFC, FSK, CNC, KMRB with different meanings | Model must learn system-specific context |
| **Class Imbalance** | Max: 5,413 samples (MPAA_R), Min: 184 samples (CNC_10), Ratio: 29:1 | Standard cross-entropy fails |
| **Cultural Context** | Same content receives different ratings based on cultural norms | Requires country-aware encoding |
| **Text-Only Input** | No visual, audio, or behavioral features | Fundamental ceiling on accuracy |

### Real-World Motivation

Content platforms operating globally must validate ratings for each country's system. Manual validation does not scale:
- 10,000 movies × 65 countries = 650,000 validations
- Manual review time: ~1 hour per validation
- Annual workload for new content: prohibitive

An automated classifier reduces manual review burden by auto-approving high-confidence predictions and flagging ambiguous cases.

---

## Approach

### 1. Composite Label Encoding

**Problem:** Rating string "12" is ambiguous across systems.

**Solution:** Encode as `{SYSTEM}_{RATING}` (e.g., `BBFC_12`, `FSK_12`, `CNC_12`, `KMRB_12`)

**Benefit:** Forces model to learn system-specific patterns rather than conflating identical strings.

### 2. Multi-Task Learning

**Architecture:** Two prediction heads with shared DeBERTa-v3-base encoder

- **Auxiliary Task:** Predict maturity level (5 classes: Family, Young Teen, Teen, Older Teen, Mature)
- **Main Task:** Predict exact rating (51 classes: `MPAA_R`, `BBFC_15`, etc.)

**Benefit:** Auxiliary task provides hierarchical signal, helping model learn coarse-grained patterns before fine-grained distinctions.

**Loss Function:**
```
L_total = 0.25 × L_maturity + 0.75 × L_rating
```

### 3. Handling Class Imbalance

**Techniques:**
1. **Geometric Mean Oversampling:** Raise rare classes to √(max × min) ≈ 997 samples
2. **Focal Loss:** Down-weight easy examples, up-weight hard ones (γ=2.5)
3. **Label Smoothing:** Prevent overconfidence (ε=0.12)
4. **Class Weights:** √(1/frequency) weighting in loss

### 4. System-Aware Input Formatting

**Format:** `[SYSTEM] Title | Synopsis`

**Example:**
```
[MPAA] The Dark Knight | When the menace known as the Joker...
[BBFC] The Dark Knight | When the menace known as the Joker...
[FSK] The Dark Knight | When the menace known as the Joker...
```

**Benefit:** Explicit system markers help model contextualize predictions.

---

## Architecture

### Model Structure

```
Input Text (title + synopsis)
    ↓
DeBERTa-v3-base Tokenizer (max 256 tokens)
    ↓
DeBERTa-v3-base Encoder (184M parameters)
    ↓
[CLS] Token Representation (768-dim)
    ↓
┌─────────────────────────┬──────────────────────────┐
│   Maturity Head         │    Rating Head           │
│   (Auxiliary Task)      │    (Main Task)           │
│                         │                          │
│   Linear(768 → 384)     │    Linear(768 → 512)     │
│   → ReLU + Dropout(0.45)│    → ReLU + Dropout(0.45)│
│   → Linear(384 → 192)   │    → Linear(512 → 384)   │
│   → ReLU + Dropout(0.45)│    → ReLU + Dropout(0.45)│
│   → Linear(192 → 5)     │    → Linear(384 → 51)    │
│                         │                          │
│   Output: Maturity      │    Output: Rating        │
└─────────────────────────┴──────────────────────────┘
```

**Total Parameters:** 186.0M (184M encoder + 2M heads)

### Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **Encoder LR** | 6e-6 | Conservative for pretrained weights |
| **Heads LR** | 3e-5 | Higher for randomly initialized layers |
| **Batch Size** | 32 × 2 (grad accum) = 64 | Optimal for A100 memory |
| **Max Epochs** | 50 | With early stopping (patience=20) |
| **Dropout** | 0.45 | Aggressive regularization |
| **Focal Gamma** | 2.5 | Strong focus on hard examples |
| **Label Smoothing** | 0.12 | Prevent overconfidence |
| **Weight Decay** | 0.01 | L2 regularization |
| **Grad Clip** | 0.5 | Prevent exploding gradients |
| **Scheduler** | Cosine with 15% warmup | Smooth LR annealing |
| **Mixed Precision** | FP16 | 2× speedup, 50% memory reduction |

---

## Results

### Dataset

- **Source:** TMDb API (public movie metadata)
- **Size:** 60,695 samples from 12,264 unique movies
- **Countries:** 65 (MPAA/US, BBFC/UK, FSK/Germany, CNC/France, ACB/Australia, EIRIN/Japan, DJCTQ/Brazil, + 58 others)
- **Classes:** 51 unique rating labels
- **Split:** 75% train / 12.5% validation / 12.5% test (stratified)

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test Accuracy** | **65.11%** | Primary metric |
| **Validation Accuracy** | 64.68% | Consistent with test |
| **Training Accuracy** | 76.03% | Moderate overfitting (11% gap) |
| **Mean Per-Class Accuracy** | 67.89% | Balanced across classes |
| **Std Dev (Per-Class)** | 30.51% | High variance due to imbalance |
| **Random Baseline** | 1.96% | Uniform guess over 51 classes |
| **Improvement Factor** | **33.2×** | Significant over baseline |

### Per-Class Analysis

**Best Performing (≥85% accuracy):**

| Class | Accuracy | Samples | System |
|-------|----------|---------|--------|
| ANICA_T | 100.00% | 95 | Italy |
| DJCTQ_12 | 100.00% | 95 | Brazil |
| DJCTQ_14 | 98.95% | 95 | Brazil |
| DJCTQ_16 | 100.00% | 95 | Brazil |
| CBOS_16 | 100.00% | 96 | Poland |
| ACB_R18+ | 88.54% | 96 | Australia |
| CNC_10 | 89.47% | 95 | France |
| ACB_G | 85.26% | 95 | Australia |

**Lowest Performing (<20% accuracy):**

| Class | Accuracy | Samples | Primary Confusion |
|-------|----------|---------|-------------------|
| EIRIN_R15+ | 4.21% | 95 | EIRIN_R18+ (adjacent maturity) |
| BBFC_12 | 16.54% | 133 | BBFC_12A (UK split rating) |
| EIRIN_PG12 | 15.38% | 104 | EIRIN_G (semantic overlap) |
| CNC_16 | 15.62% | 96 | CNC_18 (fine-grained boundary) |

**Analysis:** Low-performing classes exhibit inherent ambiguity. EIRIN_R15+ vs R18+ and BBFC_12 vs 12A are difficult even for human annotators without visual content.

### Training Dynamics

- **Best Epoch:** 27
- **Training Time:** ~3 hours (NVIDIA A100 80GB)
- **Early Stopping:** Triggered at epoch 47 (patience=20)
- **Train-Val Gap:** 11.35 percentage points at best epoch
- **Convergence:** Validation accuracy plateaued after epoch 27 despite continued training

---

## Usage

### Installation

```bash
# Clone repository
git clone https://github.com/deval245/veridex.git
cd veridex

# Install dependencies
pip install -r requirements.txt
```

### Training

**Option 1: Google Colab (Free A100)**

1. Upload `COLAB_V2_PRODUCTION_90_PERCENT.py` to Colab
2. Upload `data/multimodal_expanded_coverage.json` to Google Drive
3. Run the script (takes ~3 hours)
4. Model checkpoint saved to: `/content/drive/MyDrive/veridex_v2_production/best_model_v2.pt`

**Keep Alive Script** (paste in browser console, F12):
```javascript
setInterval(() => {
    console.log("🟢 VERIDEX Training Active");
    document.querySelector('colab-toolbar-button#connect')?.click();
}, 60000);
```

**Option 2: Local Training**

```bash
# Requires CUDA GPU with 16GB+ VRAM
python COLAB_V2_PRODUCTION_90_PERCENT.py
```

### Inference

```python
import torch
from transformers import AutoTokenizer

# Load model
checkpoint = torch.load('best_model_v2.pt', map_location='cpu')
model = checkpoint['model']  # Or rebuild from architecture
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

id_to_label = checkpoint['id_to_label']

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Predict function
def predict_rating(title, synopsis, system="MPAA"):
    text = f"[{system}] {title} | {synopsis}"
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding=True
    )
    
    with torch.no_grad():
        rating_logits, maturity_logits = model(
            encoding['input_ids'],
            encoding['attention_mask']
        )
        
        # Get prediction
        pred_id = rating_logits.argmax(dim=1).item()
        confidence = torch.softmax(rating_logits, dim=1)[0, pred_id].item()
        predicted_label = id_to_label[pred_id]
        
    return {
        'rating': predicted_label.split('_')[1],  # e.g., "PG-13"
        'system': predicted_label.split('_')[0],  # e.g., "MPAA"
        'confidence': confidence,
        'auto_approve': confidence > 0.90  # High confidence threshold
    }

# Example
result = predict_rating(
    title="The Dark Knight",
    synopsis="When the menace known as the Joker wreaks havoc and chaos...",
    system="MPAA"
)
print(result)
# {'rating': 'PG-13', 'system': 'MPAA', 'confidence': 0.87, 'auto_approve': False}
```

### Batch Inference

```python
import pandas as pd

# Load dataset
df = pd.read_csv('movies.csv')  # Columns: title, synopsis

# Predict for all movies
predictions = []
for _, row in df.iterrows():
    result = predict_rating(row['title'], row['synopsis'], system="MPAA")
    predictions.append(result)

# Save results
df['predicted_rating'] = [p['rating'] for p in predictions]
df['confidence'] = [p['confidence'] for p in predictions]
df.to_csv('predictions.csv', index=False)
```

---

## Methodology

### Data Collection

1. **Source:** TMDb API (The Movie Database, public)
2. **Coverage Strategy:**
   - Started with 7 major markets (US, UK, Germany, France, Australia, Japan, Brazil)
   - Expanded to 65 countries based on global OTT platform coverage needs
   - Fetched movies with ≥100 votes and release dates 1980-2024
3. **Deduplication:** Removed duplicate movies across countries
4. **Filtering:** Kept rating classes with ≥100 samples

### Preprocessing

1. **Text Cleaning:**
   - Remove HTML tags
   - Normalize whitespace
   - Truncate synopsis to 256 tokens (DeBERTa limit)

2. **Label Processing:**
   - Map raw ratings to `{SYSTEM}_{RATING}` format
   - Assign maturity level (0-4) based on rating hierarchy
   - Filter out classes with <100 samples

3. **Data Augmentation (for rare classes):**
   - Word dropout (p=0.13)
   - Synonym replacement using WordNet
   - Applied only to classes with <500 samples

### Training Procedure

1. **Initialization:** Load pretrained DeBERTa-v3-base from Hugging Face
2. **Oversampling:** Balance dataset using geometric mean strategy
3. **Optimization:** AdamW with layerwise learning rates
4. **Regularization:** High dropout (0.45), label smoothing (0.12), gradient clipping (0.5)
5. **Early Stopping:** Monitor validation accuracy, patience=20 epochs
6. **Checkpointing:** Save best model based on validation accuracy

---

## Limitations

1. **Text-Only:** No visual (poster, frames) or audio features analyzed
2. **Static Model:** Does not adapt to rating system policy changes over time
3. **Class Imbalance:** Some rare classes (<200 samples) still underperform
4. **Cultural Nuance:** Text alone cannot capture all culture-specific context
5. **Temporal Bias:** Dataset skewed toward recent movies (1980-2024)

---

## Future Work

1. **Multimodal Extension:**
   - Add movie poster analysis (CNN encoder)
   - Add video frame sampling (3D CNN or ViT)
   - Expected improvement: 10-15 percentage points

2. **Active Learning:**
   - Prioritize data collection for struggling classes (EIRIN_R15+, BBFC_12)
   - Human-in-the-loop annotation for ambiguous cases

3. **Explainability:**
   - Attention visualization to identify content triggers
   - Generate natural language explanations (e.g., "Rated R due to violence")

4. **Online Learning:**
   - Detect rating policy drift
   - Update model incrementally without full retraining

5. **Expanded Coverage:**
   - Add remaining 85 countries (total 150+ rating systems worldwide)
   - Include TV-specific rating systems (TV-Y, TV-MA, etc.)

---

## Repository Structure

```
veridex/
├── COLAB_V2_PRODUCTION_90_PERCENT.py  # Main training script (802 lines)
├── COLAB_INSTRUCTIONS.txt             # Colab setup guide
├── COLAB_KEEP_ALIVE.js                # Browser keep-alive script
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── README.md                           # This file
│
├── data/
│   ├── multimodal_expanded_coverage.json  # Training dataset (60K samples)
│   └── benchmarks/
│       └── POLICYBENCH_SPEC.md            # Benchmark specifications
│
├── src/                                # Source code (modular)
│   ├── config.py                       # Configuration management
│   ├── adapters/
│   │   ├── tmdb.py                     # TMDb API adapter
│   │   └── base.py                     # Base adapter interface
│   ├── models/
│   │   ├── policy_deberta.py           # Base model architecture
│   │   ├── policy_deberta_v2.py        # Enhanced model
│   │   └── multimodal_policy_deberta.py # Multimodal (future)
│   ├── rating_systems/
│   │   ├── countries.json              # Country-system mappings
│   │   └── manager.py                  # Rating system manager
│   └── training/
│       └── lora_optimizer.py           # LoRA fine-tuning (future)
│
├── scripts/
│   ├── expand_for_disney_coverage.py  # Dataset expansion
│   ├── expand_disney_async_monitored.py # Monitored expansion
│   └── fetch_multimodal_dataset.py    # Data fetcher
│
├── examples/
│   └── demo_production.py              # Inference demo
│
└── tests/
    ├── unit/                           # Unit tests
    └── integration/                    # Integration tests
```

---

## Dependencies

**Core:**
- Python 3.11+
- PyTorch 2.8+
- Transformers 4.0+ (Hugging Face)
- NumPy, Pandas

**Optional:**
- CUDA 12.6+ (for GPU training)
- Google Colab (for free A100 access)

See `requirements.txt` for complete list.

---

## Citation

If you use VERIDEX in your research, please cite:

```bibtex
@software{thakkar2024veridex,
  title={VERIDEX: Multi-Country Content Rating Classification},
  author={Thakkar, Deval},
  year={2024},
  url={https://github.com/deval245/veridex},
  note={Transformer-based classifier achieving 65.11\% accuracy on 51-class content rating prediction}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Contact

**Deval Thakkar**
- Email: devalth8@gmail.com
- GitHub: [@deval245](https://github.com/deval245)
- LinkedIn: [Deval Thakkar](https://www.linkedin.com/in/deval-thakkar)

---

## Acknowledgments

- **DeBERTa-v3:** Microsoft Research
- **TMDb API:** The Movie Database (public data source)
- **PyTorch:** Meta AI Research
- **Hugging Face:** Transformers library
- **Google Colab:** Free GPU resources

---

**Last Updated:** November 13, 2024

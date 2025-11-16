# Figures

This directory contains figures and visualizations for the VERIDEX V9.1 paper.

## Structure

```
figures/
├── architecture_diagram.png      # Model architecture
├── confusion_matrix.png          # Overall confusion matrix
├── calibration_plot.png          # Uncertainty calibration
├── ablation_results.png          # Ablation study visualization
└── README.md                     # This file
```

## Generating Figures

### From Evaluation Scripts

The evaluation scripts automatically generate figures:

```bash
# Main evaluation (generates confusion matrices and calibration plot)
python EVALUATE_V9.1_FINAL.py

# Figures saved to: /content/drive/MyDrive/veridex_v9.1_evaluation/
```

### Manual Generation

You can regenerate figures from saved results:

```python
import json
import matplotlib.pyplot as plt

# Load evaluation results
with open('results/v9.1_evaluation/evaluation_results.json') as f:
    results = json.load(f)

# Generate custom visualizations
# ... your plotting code ...
```

## Figure Requirements for Paper

### Resolution
- **Minimum**: 300 DPI
- **Recommended**: 600 DPI for print
- **Format**: PNG (for arXiv) or PDF (for IEEE)

### Sizes
- **Single column**: 3.5 inches width
- **Double column**: 7 inches width
- **Height**: Adjust to maintain aspect ratio

### Fonts
- **Labels**: 10-12pt
- **Title**: 12-14pt
- **Use**: Sans-serif fonts (Arial, Helvetica)

## Figure Descriptions

### `architecture_diagram.png`
- Shows V9.1 architecture
- Includes: V8.1 base, PLD-Net, ensemble
- Flow diagram format

### `confusion_matrix.png`
- Overall confusion matrix (51 classes)
- Or per-system confusion matrices
- Color-coded for readability

### `calibration_plot.png`
- Uncertainty vs correctness
- Shows model calibration quality
- Scatter plot or reliability diagram

### `ablation_results.png`
- Bar chart comparing variants
- Shows contribution of each component
- Error bars if available

## Note

Figures are generated during evaluation. For the paper, use high-resolution versions. These placeholder files are not committed to the repo (see `.gitignore`).


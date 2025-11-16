# Evaluation Results

This directory contains evaluation outputs from VERIDEX V9.1.

## Structure

```
results/
├── v9.1_evaluation/          # Main evaluation outputs
│   ├── evaluation_results.json
│   ├── confusion_matrices_per_system.png
│   └── calibration_plot.png
│
└── ablation_studies/          # Ablation study results
    └── ablation_results.json
```

## Files

### `v9.1_evaluation/evaluation_results.json`
Comprehensive evaluation metrics:
- Overall accuracy (V2, V8.1, V9.1)
- Per-rating-system accuracy
- Per-class metrics
- Confidence scores

### `v9.1_evaluation/confusion_matrices_per_system.png`
Confusion matrices for each rating system (MPAA, BBFC, FSK, etc.)

### `v9.1_evaluation/calibration_plot.png`
Calibration plot showing uncertainty vs correctness

### `ablation_studies/ablation_results.json`
Ablation study results:
- Full V9.1
- Ablation A: Remove PLD-Net
- Ablation B: Fixed 50/50 ensemble
- V2 baseline

## Generating Results

Run the evaluation scripts:

```bash
# Main evaluation
python EVALUATE_V9.1_FINAL.py

# Ablation studies
python ABLATION_STUDIES_V9.1.py
```

Results are automatically saved to:
- `/content/drive/MyDrive/veridex_v9.1_evaluation/` (on Colab)
- `results/` (local)

## Note

These files are generated during evaluation. For reproducibility, commit the JSON files but not the PNG files (too large). See `.gitignore` for details.


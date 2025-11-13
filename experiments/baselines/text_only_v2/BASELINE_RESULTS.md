# VERIDEX v2 Baseline Results

## Model Configuration
- **Architecture**: DeBERTa-v3-base (184M parameters)
- **Approach**: Text-only, Multi-task learning
- **Tasks**: Rating prediction (51 classes) + Maturity prediction (5 classes)

## Dataset
- **Samples**: 60,695 (train: 50,525, val: 8,421, test: 8,421)
- **Countries**: 65
- **Rating Systems**: 51 unique classes
- **Class Imbalance**: 29:1 (max: 5,413, min: 184)

## Training Configuration
- **Encoder LR**: 6e-6
- **Heads LR**: 3e-5
- **Batch Size**: 32 (effective 64 with gradient accumulation)
- **Epochs**: 50 (early stopping patience=20)
- **Dropout**: 0.45
- **Focal Gamma**: 2.5
- **Label Smoothing**: 0.12

## Results
- **Test Accuracy**: 65.11%
- **Validation Accuracy**: 64.68%
- **Training Accuracy**: 76.03%
- **Mean Per-Class Accuracy**: 67.89%
- **Random Baseline**: 1.96% (1/51)
- **Improvement**: 33.2× over random

## Best Performing Classes (≥85% accuracy)
- ANICA_T: 100.00%
- DJCTQ_12: 100.00%
- DJCTQ_14: 98.95%
- DJCTQ_16: 100.00%
- CBOS_16: 100.00%
- ACB_R18+: 88.54%
- CNC_10: 89.47%
- ACB_G: 85.26%

## Worst Performing Classes (<20% accuracy)
- EIRIN_R15+: 4.21% (confused with R18+)
- BBFC_12: 16.54% (confused with 12A)
- EIRIN_PG12: 15.38% (confused with G)
- CNC_16: 15.62% (confused with 18)

## Training Dynamics
- **Best Epoch**: 27
- **Training Time**: ~3 hours (NVIDIA A100)
- **Train-Val Gap**: 11.35 percentage points

## Backup Date
`date +"%Y-%m-%d %H:%M:%S"`

## Next Steps
- Implement cultural embeddings to improve accuracy → target 72-75%
- Address struggling classes through better cultural context
- Enable zero-shot prediction for new countries

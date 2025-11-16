# Pre-trained Models

This directory contains information about VERIDEX model architectures.

## Important Notes

- **Checkpoints are not distributed** due to licensing restrictions and file size limitations.
- **Users must train their own model** using the provided training script `TRAIN_V9.1_ULTIMATE.py`.
- **Architecture, documentation, and evaluation code** are included for academic reference only.
- **Model weights are not publicly available** and cannot be redistributed.

---

## Model Architectures

### V9.1 (Best Model)

**Architecture**: Policy-Latent Diffusion Network (PLD-Net)

- **Components**: V2 base (frozen), V8.1 cultural layer (frozen), PLD-Net (trainable), uncertainty-weighted ensemble
- **Performance**: 80.60% validation accuracy, 80.33% test accuracy
- **Format**: All-in-one checkpoint containing complete model state
- **Key Innovation**: Combines frozen baseline with trainable policy-aware network

### V8.1 (Baseline)

**Architecture**: Text encoder + Cultural embeddings

- **Components**: DeBERTa-v3-base text encoder, 64-dim cultural embeddings, calibration layer
- **Performance**: 78.65% validation accuracy, 79.29% test accuracy
- **Format**: Split checkpoint (cultural layer only, requires V2 base)
- **Key Innovation**: Country-specific cultural embeddings for rating prediction

**Note**: V8.1 uses a "split checkpoint" architecture where the cultural layer is stored separately from the V2 base model. This allows incremental training of cultural components.

### V2 (Text-only Baseline)

**Architecture**: Text-only transformer

- **Components**: DeBERTa-v3-base encoder, multi-task classification heads
- **Performance**: 77.12% validation accuracy
- **Format**: Full model checkpoint
- **Purpose**: Baseline for text-only rating prediction without cultural context

---

## Model Loading (Conceptual)

The following code examples demonstrate how models would be loaded **if you have trained your own checkpoint** using the provided training scripts. These checkpoints are **not included** in this repository.

### V9.1 (All-in-One)

```python
import torch
from TRAIN_V9.1_ULTIMATE import PLDNet

# Load your private checkpoint (not provided with repo)
# checkpoint = torch.load('path/to/your/private_checkpoint.pt', map_location='cpu')

# Initialize model architecture (use config from checkpoint)
# model = PLDNet(
#     model_name=checkpoint['config']['model_name'],
#     num_classes=checkpoint['config']['num_classes'],
#     num_countries=checkpoint['config']['num_countries'],
#     # ... other config from checkpoint
# )

# Load weights
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
```

### V8.1 (Split Checkpoint)

```python
# Conceptual example - requires your own trained checkpoints
# First load V2 base (from your private checkpoint)
# v2_checkpoint = torch.load('path/to/your/v2_checkpoint.pt')
# v2_model = V2BaseModel(...)
# v2_model.load_state_dict(v2_checkpoint['model_state_dict'])

# Then load V8.1 cultural layer (from your private checkpoint)
# v8_checkpoint = torch.load('path/to/your/v8_cultural_checkpoint.pt')
# cultural_layer = CulturalCalibrationLayer(...)
# cultural_layer.load_state_dict(v8_checkpoint['cultural_layer_state_dict'])

# Combine
# v8_model = V8Model(v2_model, cultural_layer)
```

**Note**: All checkpoint loading code is commented out to emphasize that checkpoints are not provided. Users must train their own models.

---

## File Structure

```
models/
├── README.md                    # This file
└── (checkpoints not included - users must train their own)
```

---

## Training Your Own Model

To obtain model weights, you must train the model yourself:

1. **Obtain Dataset**: See [DATA_ACQUISITION.md](../DATA_ACQUISITION.md) for instructions
2. **Run Training**: Execute `TRAIN_V9.1_ULTIMATE.py` with your dataset
3. **Checkpoints**: Training will save checkpoints to your specified directory
4. **Evaluation**: Use `EVALUATE_V9.1_FINAL.py` to evaluate your trained model

**Expected Training Time**: ~3-4 hours on A100 GPU (20 epochs with early stopping)

---

## License

Models are governed by the VERIDEX Research License. See [LICENSE](../LICENSE) for details.

**Restrictions**:
- Redistribution of model weights is strictly prohibited
- Commercial use requires written permission
- Training derivative models requires written permission
- Model weights are for academic review only

---

**Last Updated**: November 16, 2025

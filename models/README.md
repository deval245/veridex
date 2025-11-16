# Pre-trained Models

This directory contains information about pre-trained VERIDEX models.

## Model Checkpoints

Due to file size limitations, model checkpoints are hosted on cloud storage.

### V9.1 (Best Model) - Recommended

- **Accuracy**: 80.60% validation, 80.33% test
- **Size**: 891 MB
- **Format**: All-in-one `.pt` checkpoint
- **Contains**: V2 base, V8.1 cultural layer, PLD-Net, ensemble weights
- **Download**: 
  - [Google Drive](https://drive.google.com/...) - *Add your link here*
  - [Hugging Face Hub](https://huggingface.co/...) - *Optional: Upload to HF*

**SHA256 Checksum**: `abc123...` - *Add checksum after upload*

### V8.1 (Baseline)

- **Accuracy**: 78.65% validation, 79.29% test
- **Size**: 1.2 MB (split checkpoint)
- **Format**: Cultural layer only (requires V2 base)
- **Download**: [Google Drive](https://drive.google.com/...) - *Add your link here*

**Note**: V8.1 is a "split checkpoint" - it only contains the cultural layer. You need the V2 base model to use it.

### V2 (Text-only Baseline)

- **Accuracy**: 77.12% validation
- **Size**: 706 MB
- **Format**: Full model checkpoint
- **Download**: [Google Drive](https://drive.google.com/...) - *Add your link here*

## Loading Models

### V9.1 (All-in-One)

```python
import torch
from TRAIN_V9.1_ULTIMATE import PLDNet

# Load checkpoint
checkpoint = torch.load('best_model_v9.1_improved.pt', map_location='cpu')

# Initialize model (use config from checkpoint)
model = PLDNet(
    model_name=checkpoint['config']['model_name'],
    num_classes=checkpoint['config']['num_classes'],
    num_countries=checkpoint['config']['num_countries'],
    # ... other config from checkpoint
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### V8.1 (Split Checkpoint)

```python
# First load V2 base
v2_checkpoint = torch.load('v2_baseline.pt')
v2_model = V2BaseModel(...)
v2_model.load_state_dict(v2_checkpoint['model_state_dict'])

# Then load V8.1 cultural layer
v8_checkpoint = torch.load('best_model_v8.1.pt')
cultural_layer = CulturalCalibrationLayer(...)
cultural_layer.load_state_dict(v8_checkpoint['cultural_layer_state_dict'])

# Combine
v8_model = V8Model(v2_model, cultural_layer)
```

## Model Verification

After downloading, verify the checkpoint:

```python
import torch

checkpoint = torch.load('best_model_v9.1_improved.pt', map_location='cpu')
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Validation Accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
print(f"Test Accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
print(f"Model Keys: {len(checkpoint['model_state_dict'].keys())}")
```

## File Structure

```
models/
├── README.md                    # This file
└── (checkpoints not in repo - too large)
```

## Alternative: Hugging Face Hub

For better discoverability, consider uploading to Hugging Face Hub:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Upload model
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="best_model_v9.1_improved.pt",
    path_in_repo="pytorch_model.bin",
    repo_id="deval245/veridex-v9.1",
    repo_type="model"
)
```

## License

All models are released under VERIDEX Research License. See [LICENSE](../LICENSE) for details.

**Note**: Model weights are provided for academic review only. Redistribution, commercial use, and training derivative models are strictly prohibited without written permission.

---

**Note**: Update download links after uploading models to cloud storage.


# Vision Transformer (ViT) Implementation from Scratch

This repository contains a PyTorch implementation of the Vision Transformer (ViT) model as described in the paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929).

## Overview

The Vision Transformer (ViT) treats image classification as a sequence prediction task. It splits an image into fixed-size patches, linearly embeds them, adds position embeddings, and processes the resulting sequence with a standard Transformer encoder.

### Key Components

1. **Patch Embedding (`PatchEmbedding` class)**
   - Splits images into fixed-size patches
   - Projects patches to embedding dimension
   - Adds learnable classification token
   - Adds learnable position embeddings

2. **Multi-Head Attention (`MultiHeadAttention` class)**
   - Implements scaled dot-product attention
   - Supports multiple attention heads
   - Includes dropout for regularization

3. **MLP Block (`MLP` class)**
   - Feed-forward network
   - GELU activation
   - Dropout layers

4. **Transformer Block (`TransformerBlock` class)**
   - Combines attention and MLP
   - Uses Layer Normalization
   - Implements residual connections

5. **Vision Transformer (`VisionTransformer` class)**
   - Complete model architecture
   - Configurable parameters for different model sizes

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
einops>=0.6.0
```

## Usage

1. Install the required packages:
```bash
pip install torch torchvision einops
```

2. Train the model:
```bash
python train.py
```

## Model Configuration

The default configuration uses:
- Image size: 224x224
- Patch size: 16x16
- Embedding dimension: 768
- Number of layers: 12
- Number of attention heads: 12
- MLP ratio: 4.0
- Dropout: 0.1

You can modify these parameters in `train.py` when initializing the model.

## Training Details

The training script (`train.py`) includes:
- CIFAR-10 dataset with standard preprocessing
- AdamW optimizer
- Cross-entropy loss
- Basic data augmentation (resize and center crop)
- Progress tracking with loss and accuracy metrics

## References

1. Dosovitskiy, A., et al. (2021). ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929)
2. The official implementation: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

## License

MIT License

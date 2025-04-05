# EyeToy: Multiscale Neighborhood Attention (MSNAT)

## Overview

EyeToy is an advanced deep learning framework implementing a multi-scale neighborhood attention mechanism that simultaneously processes input at different granularities, enabling efficient global context approximation in linear time.

## Key Innovations

### Multi-Scale Local Attention
- Parallel processing of input through multiple attention head groups
- Concurrent analysis at different scales and receptive fields
- Varying token merging factors and attention window sizes
- Example early layer configuration:
  - Attention head groups with token merging factors: 1x, 2x, 4x, 8x
  - Corresponding attention windows: 17, 13, 11, 9
- Exponential receptive field growth with linear computational complexity

### Global Context Mechanism
- Efficient global information processing using register tokens
- O(N) complexity message-passing pathway
- Dedicated context token for final global representation aggregation

## Architecture Components

### Attention Mechanisms
- Multi-scale, multi-head neighborhood attention configurations
- Flexible attention window and stride settings
- Implementations for 1D, 2D, 3D data


### Transformer Layers
- Multi-Scale Multi-Head Neighborhood Attention Transformers
- Global Attention Transformers
- Positional encoding

### Encoders
- Dimension-specific encoder implementations
- Initial projection and transformation layers
- Advanced global attention mechanisms

## Technical Advantages
- Concurrent processing of fine-grained local and coarse-grained global contexts
- Efficient approximation of global context
- Linear computational complexity
- Adaptable to various input dimensions and data types

## Usage Example

```python
from eyetoy.encoder import Encoder2D
from eyetoy.params import DEFAULT_IMG_ENCODER_PARAMS

# Create a 2D encoder with default multi-scale parameters
encoder = Encoder2D(**DEFAULT_IMG_ENCODER_PARAMS.__dict__)

# Process input with multi-scale neighborhood attention
output, context_token = encoder(input_tensor)
```

## Dependencies
- PyTorch
- NATTEN (Neighborhood Attention library)
- torchvision

## Acknowledgments
- [PyTorch](https://pytorch.org/)
- [NATTEN: Neighborhood Attention](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)

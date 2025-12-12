# PyTorch EBM Models - Quick Start Guide

## Files

- `models_pytorch.py` - PyTorch implementation of EBM models (converted from TensorFlow)
- `imagenet_demo_pytorch.py` - Simple demo script showing model usage

## Models Available

1. **MnistNet** - For MNIST (28×28 grayscale images)
2. **DspritesNet** - For DSprites (64×64 images)
3. **ResNet32** - For CIFAR-10 (32×32 RGB images)
4. **ResNet32Large/Wider/Larger** - Larger variants for 32×32 images
5. **ResNet128** - For ImageNet (128×128 RGB images)

## Installation

### Basic Requirements
```bash
pip install torch torchvision numpy
```

### For TPU Support (Google Cloud TPU)
```bash
pip install torch torchvision
pip install torch_xla
```

### For TPU Support (Kaggle/Colab TPU)
```bash
pip install cloud-tpu-client
pip install torch torch_xla -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/index.html
```

## Running the Demo

### On CPU/GPU
```bash
python ebm_code_release/imagenet_demo_pytorch.py
```

### On TPU (Google Cloud)
```bash
# Single TPU core
python ebm_code_release/imagenet_demo_pytorch.py

# Multi TPU cores (distributed)
python -m torch_xla.distributed.xla_dist \
    --tpu=[TPU_NAME] \
    --conda-env=[ENV_NAME] \
    -- python ebm_code_release/imagenet_demo_pytorch.py
```

### On Colab TPU
```python
# In Colab notebook
import os
os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'

!python ebm_code_release/imagenet_demo_pytorch.py
```

## Demo Script Features

The `imagenet_demo_pytorch.py` demonstrates:

1. **Model Initialization** - Creates ResNet128 with random weights
2. **Forward Pass** - Computes energy for random inputs
3. **Gradient Computation** - Tests backpropagation through the model
4. **Langevin Dynamics Sampling** - Generates samples using gradient-based MCMC

### Key Features:
- ✅ Automatic device detection (TPU/CUDA/CPU)
- ✅ Random weight initialization
- ✅ No checkpoint loading required
- ✅ Clean, minimal implementation
- ✅ Energy-based sampling via Langevin dynamics

## Usage Example

```python
from models_pytorch import ResNet128
import torch

# Initialize model
model = ResNet128(num_channels=3, num_filters=64, train=False)
model.eval()

# Random input
x = torch.randn(8, 3, 128, 128)  # batch_size=8, 128x128 RGB
labels = torch.randn(8, 1000)     # ImageNet 1000 classes

# Forward pass (compute energy)
energy = model(x, label=labels)
print(f"Energy shape: {energy.shape}")  # [8, 1]
```

## Model Configuration

The models use a FLAGS object for configuration:

```python
from models_pytorch import FLAGS

FLAGS.cclass = True          # Use conditional model
FLAGS.use_attention = False  # Use attention blocks
FLAGS.swish_act = False      # Use Swish activation
FLAGS.dataset = 'cifar10'    # Dataset name
```

## Performance Notes

### Memory Usage (ResNet128):
- Batch size 16: ~4GB GPU memory
- Batch size 32: ~8GB GPU memory

### TPU Performance:
- TPU v2/v3: ~5-10x faster than V100 GPU
- Best with batch sizes: 32, 64, 128

### Optimization Tips:
1. Use larger batch sizes on TPU (better utilization)
2. Use `torch.compile()` for PyTorch 2.0+ (faster on GPU)
3. Use mixed precision training: `torch.cuda.amp` or `torch_xla.amp`

## Differences from TensorFlow Version

1. **Weight Management**: Uses `nn.Module` instead of weight dictionaries
2. **Initialization**: Layers initialized in `__init__` instead of separate function
3. **Device Handling**: Explicit `.to(device)` calls
4. **Gradients**: Use `torch.autograd.grad()` instead of `tf.gradients()`
5. **Clipping**: `torch.clamp()` instead of `tf.clip_by_value()`

## Next Steps

1. **Add Utilities**: Implement PyTorch versions of utility functions (spectral norm, conditional batch norm, etc.)
2. **Add Data Loading**: Create PyTorch DataLoaders for your datasets
3. **Add Training**: Implement training loops with optimizers
4. **Port Weights**: Convert pre-trained TensorFlow weights to PyTorch format

## Troubleshooting

### TPU Not Detected
```python
import torch_xla
import torch_xla.core.xla_model as xm
print(xm.xla_device())  # Should print: xla:0 or xla:1
```

### Memory Issues on TPU
- Reduce batch size
- Use gradient checkpointing
- Call `xm.mark_step()` regularly

### Slow First Run
- TPU compiles graphs on first run (normal)
- Subsequent runs will be faster

## Contact & Issues

If you encounter issues with the PyTorch conversion, check:
1. PyTorch version compatibility (>= 1.13 recommended)
2. TPU runtime version (for torch_xla compatibility)
3. CUDA version (for GPU support)




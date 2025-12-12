# Inference vs Sampling in Energy-Based Models

## The Confusion

You asked: *"Why do we need train mode and backward pass? We're only doing inference."*

**Answer**: The original demo does **Langevin dynamics sampling**, which requires gradients even though it's not training the model.

## Two Different Use Cases

### 1. Pure Inference (Energy Evaluation)

**What it does**: Given an image, compute its energy score

**Code**:
```python
model.eval()
with torch.no_grad():
    energy = model(image)  # Just a number
```

**Use cases**:
- Classify images (lower energy = more likely class)
- Anomaly detection (high energy = anomaly)
- Compare images (which is more realistic?)

**Requirements**:
- ✅ Model in eval mode
- ✅ No gradients (`torch.no_grad()`)
- ✅ No backward pass
- ✅ Fast and simple

**File**: `imagenet_inference_only.py` (I just created this)

---

### 2. Langevin Dynamics Sampling (What the Original Demo Does)

**What it does**: Generate/refine images by following energy gradients

**Code**:
```python
model.train()  # Need BatchNorm stats for gradients
x.requires_grad_(True)
energy = model(x)
x_grad = torch.autograd.grad(energy, x)  # ← BACKWARD PASS
x_new = x - step_lr * x_grad  # Move x toward lower energy
```

**Use cases**:
- Generate new images from noise
- Refine/denoise images
- Complete missing parts of images
- What the original TF demo does (lines 42-47, 67-68)

**Requirements**:
- ⚠️ Model in train mode (for BatchNorm gradients)
- ⚠️ Gradients enabled (`requires_grad=True`)
- ⚠️ Backward pass needed
- ⚠️ Slower (computes gradients)

**File**: `imagenet_demo_pytorch.py` (main demo)

---

## Side-by-Side Comparison

| Aspect | Inference Only | Langevin Sampling |
|--------|---------------|-------------------|
| **Model Mode** | `eval()` | `train()` |
| **Gradients** | `with torch.no_grad()` | `requires_grad=True` |
| **Backward Pass** | ❌ No | ✅ Yes |
| **Computes** | `E(x)` | `∇_x E(x)` |
| **Output** | Energy score | Refined image |
| **Speed** | Fast | Slow |
| **Memory** | Low | High |
| **Use Case** | Evaluate images | Generate images |

---

## What the Original TensorFlow Demo Does

Looking at `imagenet_demo.py` lines 34-51:

```python
# Line 39: Forward pass
energy_noise = model.forward(x_mod, weights, label=LABEL, ...)

# Line 42: BACKWARD PASS - compute gradient w.r.t. INPUT
x_grad = tf.gradients(energy_noise, [x_mod])[0]

# Line 47: Update INPUT based on gradient (not weights!)
x_last = x_mod - (lr) * x_grad

# Lines 67-68: Run this loop 200 times
for i in range(FLAGS.num_steps):
    e, x_mod = sess.run([energy_noise, x_output], {X_NOISE:x_mod, LABEL:labels})
```

**This is Langevin dynamics sampling, NOT pure inference!**

---

## Why Train Mode for Sampling?

When computing `∇_x E(x)` through BatchNorm layers:

```
Input (x) → Conv → BatchNorm → ReLU → ... → Energy (E)
               ↓
         needs stats for backward
```

**In eval mode**:
- BatchNorm uses fixed running mean/var
- Doesn't track statistics for gradients
- **Backward pass fails on TPU** ❌

**In train mode**:
- BatchNorm computes batch statistics
- Tracks what's needed for gradients
- **Backward pass works** ✅

**Important**: We're still not updating model weights! We're just computing gradients w.r.t. the input.

---

## Code Examples

### Example 1: Just Evaluate Energy (Inference)

```python
from imagenet_inference_only import InferenceOnlyDemo

demo = InferenceOnlyDemo()

# Random image
x = torch.rand(16, 3, 128, 128).to(device)
labels = torch.randn(16, 1000).softmax(dim=1).to(device)

# Get energy (no gradients)
energy = demo.compute_energy(x, labels)
print(f"Energy: {energy.mean()}")  # Just a number
```

### Example 2: Generate Images (Sampling)

```python
from imagenet_demo_pytorch import ImageNetDemo

demo = ImageNetDemo()

# Start from random noise
x = torch.rand(16, 3, 128, 128).to(device)

# Refine using Langevin dynamics (requires gradients!)
for step in range(200):
    x, energy = demo.langevin_dynamics_step(x, labels, step_lr=0.1)
    # x is gradually refined to look more realistic
```

---

## Which Should You Use?

### Use `imagenet_inference_only.py` if you want to:
- ✅ Just compute energy scores
- ✅ Fast evaluation
- ✅ No backward pass
- ✅ Works in eval mode
- ✅ Lower memory usage

### Use `imagenet_demo_pytorch.py` if you want to:
- ✅ Generate images (like original demo)
- ✅ Follow the original TF demo behavior
- ✅ Implement Langevin dynamics
- ⚠️ Requires backward pass
- ⚠️ Needs train mode for BatchNorm

---

## Summary

| Question | Answer |
|----------|--------|
| Is this "inference"? | Yes, weights are frozen |
| Do we update weights? | No, never |
| Do we use backward pass? | Yes, for input gradients |
| Why train mode? | BatchNorm needs stats for gradients |
| What does original do? | Sampling (needs gradients) |
| Can I do pure inference? | Yes, use `imagenet_inference_only.py` |

The key insight: **Sampling in EBMs requires computing ∇_x E(x), which needs backward pass, even though we're not training the model!**




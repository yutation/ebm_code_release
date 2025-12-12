"""
Simple PyTorch demo for ResNet128 EBM model
Uses random inputs and weights, runs on TPU
"""

import torch
import torch.nn.functional as F
from models_pytorch import ResNet128, FLAGS
import numpy as np


# Try to import TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.profiler as xp
    USE_TPU = True
    print("TPU support detected")
except ImportError:
    USE_TPU = False
    print("TPU not available, using CPU/GPU")

# Configuration
FLAGS.cclass = True  # Conditional model
FLAGS.use_attention = False
FLAGS.swish_act = False

class ImageNetDemo:
    def __init__(self, num_filters=64, batch_size=16, num_steps=200, step_lr=180.0):
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.step_lr = step_lr
        
        # Get device (TPU, CUDA, or CPU)
        if USE_TPU:
            self.device = xm.xla_device()
            print(f"Using TPU device: {self.device}")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using CUDA device: {self.device}")
        else:
            self.device = torch.device('cpu')
            print(f"Using CPU device: {self.device}")
        
        # Initialize model in training mode for proper BatchNorm gradient computation
        self.model = ResNet128(num_channels=3, num_filters=num_filters, train=True)
        self.model = self.model.to(self.device)
        # Keep in train mode to allow gradients through BatchNorm on TPU
        self.model.train()
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def langevin_dynamics_step(self, x, labels, step_lr):
        """
        Perform one step of Langevin dynamics
        x_{t+1} = x_t - step_lr * grad_x E(x_t) + noise
        """
        # Add small noise
        #Note: Step3
        torch_xla.sync()
        with xp.Trace('step3'):
            noise = torch.randn_like(x) * 0.005
        torch_xla.sync()
        x_noisy = x + noise
        x_noisy.requires_grad_(True)
        
        # Compute energy
        #Note: Step1
        torch_xla.sync()
        with xp.Trace('step1'):
            energy = self.model(x_noisy, label=labels)
        # torch_xla.sync()
        
        # Compute gradient
        #Note: Step2
        with xp.Trace('step2'):
            x_grad = torch.autograd.grad(energy.sum(), x_noisy, create_graph=False)[0]
        torch_xla.sync()
        
        
        # Update with gradient descent on energy
        #Note: Step4
        with xp.Trace('step4'):
            x_new = x_noisy - step_lr * x_grad
        torch_xla.sync()
        
        # Clip to valid range [0, 1]
        x_new = torch.clamp(x_new, 0, 1)
        
        return x_new.detach(), energy.detach()
    
    def sample(self, num_classes=1000):
        """
        Generate samples using Langevin dynamics
        """
        print(f"\nGenerating {self.batch_size} samples with {self.num_steps} Langevin steps...")
        
        # Random initialization: uniform [0, 1]
        x = torch.rand(self.batch_size, 3, 128, 128, device=self.device)
        
        # Random class labels (one-hot encoded)
        label_indices = torch.randint(0, num_classes, (self.batch_size,), device=self.device)
        labels = F.one_hot(label_indices, num_classes=num_classes).float()
        
        print(f"Sampling for classes: {label_indices.cpu().numpy()}")
        
        energies = []
        
        # Langevin dynamics sampling
        for step in range(self.num_steps):
            x, energy = self.langevin_dynamics_step(x, labels, self.step_lr)
            energies.append(energy.mean().item())
            
            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{self.num_steps}, Mean Energy: {energies[-1]:.4f}")
                # Synchronize TPU every few steps
            torch_xla.sync()
                    
        
        # Final synchronization

        torch_xla.sync()
        
        return x, energies
    
    def forward_pass(self):
        """
        Simple forward pass with random inputs
        """
        print("\n=== Forward Pass Test ===")
        
        # Random input: batch_size x 3 x 128 x 128
        x = torch.randn(self.batch_size, 3, 128, 128, device=self.device)
        
        # Random labels (for conditional model)
        labels = torch.randn(self.batch_size, 1000, device=self.device)
        labels = F.softmax(labels, dim=1)  # Make it a probability distribution
        
        # Forward pass
        with torch.no_grad():
            energy = self.model(x, label=labels)
        
        print(f"Input shape: {x.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Output energy shape: {energy.shape}")
        print(f"Energy values: min={energy.min().item():.4f}, "
              f"max={energy.max().item():.4f}, mean={energy.mean().item():.4f}")
        
        return energy
    
    def test_gradient_computation(self):
        """
        Test gradient computation through the model
        """
        print("\n=== Gradient Computation Test ===")
        
        # Random input
        x = torch.randn(4, 3, 128, 128, device=self.device, requires_grad=True)
        labels = torch.randn(4, 1000, device=self.device)
        labels = F.softmax(labels, dim=1)
        
        # Forward pass
        energy = self.model(x, label=labels)
        
        # Backward pass
        grad_output = torch.ones_like(energy)
        x_grad = torch.autograd.grad(energy, x, grad_outputs=grad_output)[0]
        
        print(f"Input gradient shape: {x_grad.shape}")
        print(f"Gradient norm: {x_grad.norm().item():.4f}")
        print(f"Gradient range: [{x_grad.min().item():.4f}, {x_grad.max().item():.4f}]")
        
        return x_grad


def main():
    print("=" * 60)
    print("PyTorch ImageNet EBM Demo")
    print("=" * 60)
    
    # Initialize demo
    demo = ImageNetDemo(
        num_filters=64,
        batch_size=16,
        num_steps=5,  # Reduced for quick demo
        step_lr=0.1    # Reduced for stability
    )
    xp.start_server(9012)
    
    print("\nRunning tests...")
    
    # Test 1: Simple forward pass
    # try:
    #     demo.forward_pass()
    # except Exception as e:
    #     print(f"Forward pass test failed: {e}")
    
    # # Test 2: Gradient computation
    # try:
    #     demo.test_gradient_computation()
    # except Exception as e:
    #     print(f"Gradient test failed: {e}")
    
    # Test 3: Langevin dynamics sampling (matches original TF demo)
    print("\n=== Langevin Dynamics Sampling ===")
    try:
        xp.start_trace("./traces/imagenet_demo")
        samples, energies = demo.sample(num_classes=1000)
        print(samples.shape)
        print(energies)
        xp.stop_trace()
        print(f"\nFinal samples shape: {samples.shape}")
        print(f"Final samples range: [{samples.min().item():.4f}, {samples.max().item():.4f}]")
        print(f"Energy trajectory: start={energies[0]:.4f}, end={energies[-1]:.4f}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nLangevin sampling failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: BatchNorm backward pass on TPU with train mode enabled.")
        print("If this fails, the issue is likely related to TPU/XLA BatchNorm gradients.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()


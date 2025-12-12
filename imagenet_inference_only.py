"""
Simple PyTorch inference-only demo for ResNet128 EBM model
Pure forward pass, no gradients, no backward pass
"""

import torch
import torch.nn.functional as F
from models_pytorch import ResNet128, FLAGS
import numpy as np

# Try to import TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    USE_TPU = True
    print("TPU support detected")
except ImportError:
    USE_TPU = False
    print("TPU not available, using CPU/GPU")

# Configuration
FLAGS.cclass = True
FLAGS.use_attention = False
FLAGS.swish_act = False


class InferenceOnlyDemo:
    """Pure inference - just compute energy values"""
    
    def __init__(self, num_filters=64, batch_size=16):
        self.num_filters = num_filters
        self.batch_size = batch_size
        
        # Get device
        if USE_TPU:
            self.device = xm.xla_device()
            print(f"Using TPU device: {self.device}")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using CUDA device: {self.device}")
        else:
            self.device = torch.device('cpu')
            print(f"Using CPU device: {self.device}")
        
        # Initialize model in EVAL mode (true inference)
        self.model = ResNet128(num_channels=3, num_filters=num_filters, train=False)
        self.model = self.model.to(self.device)
        self.model.eval()  # Inference mode
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def compute_energy(self, images, labels):
        """
        Pure inference: compute energy for given images
        No gradients, no backward pass
        """
        with torch.no_grad():  # No gradient computation
            energy = self.model(images, label=labels)
        
        if USE_TPU:
            xm.mark_step()
        
        return energy
    
    def evaluate_random_images(self, num_samples=16, num_classes=1000):
        """
        Evaluate energy of random images (pure forward pass)
        """
        print(f"\n=== Pure Inference: Evaluating {num_samples} random images ===")
        
        # Random images
        images = torch.randn(num_samples, 3, 128, 128, device=self.device)
        
        # Random labels
        label_indices = torch.randint(0, num_classes, (num_samples,), device=self.device)
        labels = F.one_hot(label_indices, num_classes=num_classes).float()
        
        # Compute energy (inference only)
        energies = self.compute_energy(images, labels)
        
        print(f"Energy shape: {energies.shape}")
        print(f"Energy statistics:")
        print(f"  Min:  {energies.min().item():.4f}")
        print(f"  Max:  {energies.max().item():.4f}")
        print(f"  Mean: {energies.mean().item():.4f}")
        print(f"  Std:  {energies.std().item():.4f}")
        
        return energies
    
    def compare_energies(self, num_classes=1000):
        """
        Compare energies of different types of images
        """
        print(f"\n=== Comparing Energy of Different Inputs ===")
        
        batch_size = 8
        
        # 1. Random noise
        noise_images = torch.randn(batch_size, 3, 128, 128, device=self.device)
        
        # 2. Uniform random [0, 1]
        uniform_images = torch.rand(batch_size, 3, 128, 128, device=self.device)
        
        # 3. All zeros (black)
        black_images = torch.zeros(batch_size, 3, 128, 128, device=self.device)
        
        # 4. All ones (white)
        white_images = torch.ones(batch_size, 3, 128, 128, device=self.device)
        
        # Random labels for all
        label_indices = torch.randint(0, num_classes, (batch_size,), device=self.device)
        labels = F.one_hot(label_indices, num_classes=num_classes).float()
        
        # Compute energies
        with torch.no_grad():
            noise_energy = self.model(noise_images, label=labels)
            uniform_energy = self.model(uniform_images, label=labels)
            black_energy = self.model(black_images, label=labels)
            white_energy = self.model(white_images, label=labels)
        
        print(f"Random Noise Energy:    mean={noise_energy.mean().item():.4f}, std={noise_energy.std().item():.4f}")
        print(f"Uniform [0,1] Energy:   mean={uniform_energy.mean().item():.4f}, std={uniform_energy.std().item():.4f}")
        print(f"Black Images Energy:    mean={black_energy.mean().item():.4f}, std={black_energy.std().item():.4f}")
        print(f"White Images Energy:    mean={white_energy.mean().item():.4f}, std={white_energy.std().item():.4f}")
        
        return {
            'noise': noise_energy,
            'uniform': uniform_energy,
            'black': black_energy,
            'white': white_energy
        }
    
    def benchmark_throughput(self, num_iterations=100):
        """
        Benchmark inference throughput
        """
        print(f"\n=== Benchmarking Inference Throughput ===")
        
        # Dummy data
        images = torch.randn(self.batch_size, 3, 128, 128, device=self.device)
        labels = torch.randn(self.batch_size, 1000, device=self.device)
        labels = F.softmax(labels, dim=1)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(images, label=labels)
        
        if USE_TPU:
            xm.mark_step()
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(images, label=labels)
        
        if USE_TPU:
            xm.mark_step()
        
        elapsed = time.time() - start_time
        throughput = (num_iterations * self.batch_size) / elapsed
        
        print(f"Processed {num_iterations} batches of {self.batch_size} images")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Throughput: {throughput:.2f} images/second")
        print(f"Latency: {elapsed/num_iterations*1000:.2f} ms/batch")


def main():
    print("=" * 60)
    print("PyTorch ImageNet EBM - INFERENCE ONLY Demo")
    print("No gradients, no backward pass, pure forward pass")
    print("=" * 60)
    
    # Initialize demo
    demo = InferenceOnlyDemo(num_filters=64, batch_size=16)
    
    # Test 1: Evaluate random images
    demo.evaluate_random_images(num_samples=16)
    
    # Test 2: Compare different inputs
    demo.compare_energies()
    
    # Test 3: Benchmark (optional)
    print("\nSkipping benchmark (uncomment to run)")
    # demo.benchmark_throughput(num_iterations=100)
    
    print("\n" + "=" * 60)
    print("Inference-only demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from argparse import Namespace

# Note: You'll need to create a PyTorch version of these utility functions
# from utils_pytorch import (conv_block, attention, init_conv_weight, init_attention_weight, 
#                            init_res_weight, smart_res_block, smart_res_block_optim, 
#                            smart_conv_block, smart_fc_block, smart_atten_block, 
#                            groupsort, smart_convt_block, swish)

# Placeholder for FLAGS - you may want to use argparse or a config file instead
FLAGS = Namespace(
    swish_act=False,
    cclass=False,
    datasource='mnist',
    dshape_only=False,
    dpos_only=False,
    dsize_only=False,
    drot_only=False,
    use_attention=False,
    dataset='cifar10'
)


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class MnistNet(nn.Module):
    def __init__(self, num_channels=1, num_filters=64):
        super(MnistNet, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.datasource = FLAGS.datasource
        
        if FLAGS.cclass:
            self.label_size = 10
        else:
            self.label_size = 0
        
        classes = 1
        
        # Define layers
        self.c1_pre = nn.Conv2d(1 if not FLAGS.cclass else 1 + self.label_size, 
                                 64, kernel_size=3, padding=1)
        self.c1 = nn.Conv2d(64, self.dim_hidden, kernel_size=4, stride=2, padding=1)
        self.c2 = nn.Conv2d(self.dim_hidden, 2*self.dim_hidden, kernel_size=4, stride=2, padding=1)
        self.c3 = nn.Conv2d(2*self.dim_hidden, 4*self.dim_hidden, kernel_size=4, stride=2, padding=1)
        self.fc_dense = nn.Linear(4*4*4*self.dim_hidden, 2*self.dim_hidden)
        self.fc5 = nn.Linear(2*self.dim_hidden, 1)
        
        # Activation
        if FLAGS.swish_act:
            self.act = Swish()
        else:
            self.act = nn.LeakyReLU(0.2)
    
    def forward(self, inp, label=None, **kwargs):
        batch_size = inp.size(0)
        inp = inp.view(batch_size, 1, 28, 28)
        
        # Conditional class concatenation
        if FLAGS.cclass and label is not None:
            label_d = label.view(batch_size, self.label_size, 1, 1)
            label_d = label_d.expand(batch_size, self.label_size, 28, 28)
            inp = torch.cat([inp, label_d], dim=1)
        
        h1 = self.act(self.c1_pre(inp))
        h2 = self.act(self.c1(h1))
        h3 = self.act(self.c2(h2))
        h4 = self.act(self.c3(h3))
        
        h5 = h4.view(batch_size, -1)
        h6 = self.act(self.fc_dense(h5))
        output = self.fc5(h6)
        
        return output


class DspritesNet(nn.Module):
    def __init__(self, num_channels=1, num_filters=64, cond_size=False, cond_shape=False, 
                 cond_pos=False, cond_rot=False, label_size=1):
        super(DspritesNet, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.img_size = 64
        self.label_size = label_size
        
        if FLAGS.cclass:
            self.label_size = 3
        
        try:
            if FLAGS.dshape_only:
                self.label_size = 3
            if FLAGS.dpos_only:
                self.label_size = 2
            if FLAGS.dsize_only:
                self.label_size = 1
            if FLAGS.drot_only:
                self.label_size = 2
        except:
            pass
        
        if cond_size:
            self.label_size = 1
        if cond_shape:
            self.label_size = 3
        if cond_pos:
            self.label_size = 2
        if cond_rot:
            self.label_size = 2
        
        self.cond_size = cond_size
        self.cond_shape = cond_shape
        self.cond_pos = cond_pos
        
        # Define layers
        self.c1_pre = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.c1 = nn.Conv2d(32, self.dim_hidden, kernel_size=4, stride=2, padding=1)
        self.c2 = nn.Conv2d(self.dim_hidden, 2*self.dim_hidden, kernel_size=4, stride=2, padding=1)
        self.c3 = nn.Conv2d(2*self.dim_hidden, 2*self.dim_hidden, kernel_size=4, stride=2, padding=1)
        self.c4 = nn.Conv2d(2*self.dim_hidden, 2*self.dim_hidden, kernel_size=4, stride=2, padding=1)
        self.fc_dense = nn.Linear(2*4*4*self.dim_hidden, 2*self.dim_hidden)
        self.fc5 = nn.Linear(2*self.dim_hidden, 1)
        
        # Activation
        if FLAGS.swish_act:
            self.act = Swish()
        else:
            self.act = nn.LeakyReLU(0.2)
    
    def forward(self, inp, label=None, return_logit=False, **kwargs):
        batch_size = inp.size(0)
        inp = inp.view(batch_size, 1, 64, 64)
        
        if not FLAGS.cclass:
            label = None
        
        h1 = self.act(self.c1_pre(inp))
        h2 = self.act(self.c1(h1))
        h3 = self.act(self.c2(h2))
        h4 = self.act(self.c3(h3))
        h5 = self.act(self.c4(h4))
        
        hidden6 = h5.view(batch_size, -1)
        hidden7 = self.act(self.fc_dense(hidden6))
        energy = self.fc5(hidden7)
        
        if return_logit:
            return hidden7
        else:
            return energy


class ResBlock(nn.Module):
    """Basic residual block"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x, act=F.leaky_relu):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = act(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = act(out)
        
        return out


class ResNet32(nn.Module):
    def __init__(self, num_channels=3, num_filters=128):
        super(ResNet32, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        
        if FLAGS.cclass:
            classes = 10
        else:
            classes = 1
        
        # Initial convolution
        self.c1_pre = nn.Conv2d(num_channels, self.dim_hidden, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_optim = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_1 = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_2 = self._make_res_block(self.dim_hidden, 2*self.dim_hidden, stride=2)
        self.res_3 = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_4 = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_5 = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        
        # Final layers
        self.fc5 = nn.Linear(2*self.dim_hidden, 1)
        
    def _make_res_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, inp, label=None, **kwargs):
        if not FLAGS.cclass:
            label = None
        
        act = F.leaky_relu
        
        # Forward pass
        x = self.c1_pre(inp)
        x = self.res_optim(x, act)
        x = self.res_1(x, act)
        x = self.res_2(x, act)
        x = self.res_3(x, act)
        x = self.res_4(x, act)
        x = self.res_5(x, act)
        
        x = F.relu(x)
        x = torch.sum(x, dim=[2, 3])  # Global sum pooling
        energy = self.fc5(x)
        
        return energy


class ResNet32Large(nn.Module):
    def __init__(self, num_channels=3, num_filters=128, train=False):
        super(ResNet32Large, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.dropout = train
        self.train_mode = train
        
        if FLAGS.cclass:
            classes = 10
        else:
            classes = 1
        
        # Initial convolution
        self.c1_pre = nn.Conv2d(num_channels, self.dim_hidden, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_optim = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_1 = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_2 = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_3 = self._make_res_block(self.dim_hidden, 2*self.dim_hidden, stride=2)
        self.res_4 = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_5 = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_6 = self._make_res_block(2*self.dim_hidden, 4*self.dim_hidden, stride=2)
        self.res_7 = self._make_res_block(4*self.dim_hidden, 4*self.dim_hidden)
        self.res_8 = self._make_res_block(4*self.dim_hidden, 4*self.dim_hidden)
        
        # Final layer
        self.fc5 = nn.Linear(4*self.dim_hidden, 1)
        
        # Dropout
        if self.dropout:
            self.dropout_layer = nn.Dropout(0.5)
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, inp, label=None, **kwargs):
        if not FLAGS.cclass:
            label = None
        
        # Forward pass
        x = self.c1_pre(inp)
        x = self.res_optim(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.res_6(x)
        x = self.res_7(x)
        x = self.res_8(x)
        
        if FLAGS.cclass:
            x = F.leaky_relu(x)
        else:
            x = F.relu(x)
        
        x = torch.sum(x, dim=[2, 3])  # Global sum pooling
        
        if self.dropout:
            x = self.dropout_layer(x)
        
        energy = self.fc5(x)
        
        return energy


class ResNet32Wider(nn.Module):
    def __init__(self, num_channels=3, num_filters=128, train=False):
        super(ResNet32Wider, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.dropout = train
        self.train_mode = train
        
        if FLAGS.cclass and FLAGS.dataset == "cifar10":
            classes = 10
        elif FLAGS.cclass and FLAGS.dataset == "imagenet":
            classes = 1000
        else:
            classes = 1
        
        # Initial convolution
        self.c1_pre = nn.Conv2d(num_channels, 128, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_optim = self._make_res_block(128, self.dim_hidden)
        self.res_1 = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_2 = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_3 = self._make_res_block(self.dim_hidden, 2*self.dim_hidden, stride=2)
        self.res_4 = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_5 = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_6 = self._make_res_block(2*self.dim_hidden, 4*self.dim_hidden, stride=2)
        self.res_7 = self._make_res_block(4*self.dim_hidden, 4*self.dim_hidden)
        self.res_8 = self._make_res_block(4*self.dim_hidden, 4*self.dim_hidden)
        
        # Final layer
        self.fc5 = nn.Linear(4*self.dim_hidden, 1)
        
        # Activation
        if FLAGS.swish_act:
            self.act = Swish()
        else:
            self.act = nn.LeakyReLU(0.2)
        
        # Dropout
        if self.dropout:
            self.dropout_layer = nn.Dropout(0.5)
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, inp, label=None, **kwargs):
        if not FLAGS.cclass:
            label = None
        
        # Forward pass
        x = self.act(self.c1_pre(inp))
        x = self.res_optim(x, self.act if not FLAGS.swish_act else self.act)
        x = self.res_1(x, self.act)
        x = self.res_2(x, self.act)
        x = self.res_3(x, self.act)
        x = self.res_4(x, self.act)
        x = self.res_5(x, self.act)
        x = self.res_6(x, self.act)
        x = self.res_7(x, self.act)
        x = self.res_8(x, self.act)
        
        if FLAGS.swish_act:
            x = self.act(x)
        else:
            x = F.relu(x)
        
        x = torch.sum(x, dim=[2, 3])  # Global sum pooling
        
        if self.dropout:
            x = self.dropout_layer(x)
        
        energy = self.fc5(x)
        
        return energy


class ResNet32Larger(nn.Module):
    def __init__(self, num_channels=3, num_filters=128):
        super(ResNet32Larger, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        
        if FLAGS.cclass:
            classes = 10
        else:
            classes = 1
        
        # Initial convolution
        self.c1_pre = nn.Conv2d(num_channels, self.dim_hidden, kernel_size=3, padding=1)
        
        # Residual blocks
        self.res_optim = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_1 = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_2 = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_2a = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_2b = self._make_res_block(self.dim_hidden, self.dim_hidden)
        self.res_3 = self._make_res_block(self.dim_hidden, 2*self.dim_hidden, stride=2)
        self.res_4 = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_5 = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_5a = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_5b = self._make_res_block(2*self.dim_hidden, 2*self.dim_hidden)
        self.res_6 = self._make_res_block(2*self.dim_hidden, 4*self.dim_hidden, stride=2)
        self.res_7 = self._make_res_block(4*self.dim_hidden, 4*self.dim_hidden)
        self.res_8 = self._make_res_block(4*self.dim_hidden, 4*self.dim_hidden)
        self.res_8a = self._make_res_block(4*self.dim_hidden, 4*self.dim_hidden)
        self.res_8b = self._make_res_block(4*self.dim_hidden, 4*self.dim_hidden)
        
        # Final layer
        self.fc5 = nn.Linear(4*self.dim_hidden, 1)
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, inp, label=None, **kwargs):
        if not FLAGS.cclass:
            label = None
        
        # Forward pass
        x = self.c1_pre(inp)
        x = self.res_optim(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_2a(x)
        x = self.res_2b(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.res_5a(x)
        x = self.res_5b(x)
        x = self.res_6(x)
        x = self.res_7(x)
        x = self.res_8(x)
        x = self.res_8a(x)
        x = self.res_8b(x)
        
        if FLAGS.cclass:
            x = F.leaky_relu(x)
        else:
            x = F.relu(x)
        
        x = torch.sum(x, dim=[2, 3])  # Global sum pooling
        energy = self.fc5(x)
        
        return energy


class ResNet128(nn.Module):
    """Construct the convolutional network for 128x128 images"""
    
    def __init__(self, num_channels=3, num_filters=64, train=False):
        super(ResNet128, self).__init__()
        
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.dropout = train
        self.train_mode = train
        
        classes = 1000
        
        # Initial convolution
        self.c1_pre = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        
        # Residual blocks with progressive downsampling
        self.res_optim = self._make_res_block(64, self.dim_hidden, stride=2)
        self.res_3 = self._make_res_block(self.dim_hidden, 2*self.dim_hidden, stride=2)
        self.res_5 = self._make_res_block(2*self.dim_hidden, 4*self.dim_hidden, stride=2)
        self.res_7 = self._make_res_block(4*self.dim_hidden, 8*self.dim_hidden, stride=2)
        self.res_9 = self._make_res_block(8*self.dim_hidden, 8*self.dim_hidden, stride=2)
        self.res_10 = self._make_res_block(8*self.dim_hidden, 8*self.dim_hidden)
        
        # Final layer
        self.fc5 = nn.Linear(8*self.dim_hidden, 1)
        
        # Activation
        if FLAGS.swish_act:
            self.act = Swish()
        else:
            self.act = nn.LeakyReLU(0.2)
        
        # Dropout
        if self.dropout:
            self.dropout_layer = nn.Dropout(0.5)
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return ResBlock(in_channels, out_channels, stride, downsample)
    
    def forward(self, inp, label=None, **kwargs):
        if not FLAGS.cclass:
            label = None
        
        # Forward pass
        x = self.act(self.c1_pre(inp))
        x = self.res_optim(x, self.act)
        x = self.res_3(x, self.act)
        x = self.res_5(x, self.act)
        x = self.res_7(x, self.act)
        x = self.res_9(x, self.act)
        x = self.res_10(x, self.act)
        
        if FLAGS.swish_act:
            x = self.act(x)
        else:
            x = F.relu(x)
        
        x = torch.sum(x, dim=[2, 3])  # Global sum pooling
        
        if self.dropout:
            x = self.dropout_layer(x)
        
        energy = self.fc5(x)
        
        return energy


# Example usage and testing
if __name__ == "__main__":
    # Test MnistNet
    print("Testing MnistNet...")
    mnist_model = MnistNet(num_channels=1, num_filters=64)
    mnist_input = torch.randn(8, 784)  # Batch of 8, flattened 28x28 images
    mnist_output = mnist_model(mnist_input)
    print(f"MnistNet output shape: {mnist_output.shape}")
    
    # Test DspritesNet
    print("\nTesting DspritesNet...")
    dsprites_model = DspritesNet(num_channels=1, num_filters=64)
    dsprites_input = torch.randn(8, 1, 64, 64)  # Batch of 8, 64x64 images
    dsprites_output = dsprites_model(dsprites_input)
    print(f"DspritesNet output shape: {dsprites_output.shape}")
    
    # Test ResNet32
    print("\nTesting ResNet32...")
    resnet32_model = ResNet32(num_channels=3, num_filters=128)
    resnet32_input = torch.randn(8, 3, 32, 32)  # Batch of 8, 32x32 RGB images
    resnet32_output = resnet32_model(resnet32_input)
    print(f"ResNet32 output shape: {resnet32_output.shape}")
    
    # Test ResNet128
    print("\nTesting ResNet128...")
    resnet128_model = ResNet128(num_channels=3, num_filters=64)
    resnet128_input = torch.randn(4, 3, 128, 128)  # Batch of 4, 128x128 RGB images
    resnet128_output = resnet128_model(resnet128_input)
    print(f"ResNet128 output shape: {resnet128_output.shape}")
    
    print("\nAll models created successfully!")


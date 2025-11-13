"""
SPAN (Swift Parameter-free Attention Network) architecture for image super-resolution.
Modified from: https://github.com/hongyuanyu/SPAN

Reference:
    Hongyuan Yu, Chenghua Li, Salma Abdel Magid, Yulun Zhang, Jinjin Gu, Hanspeter Pfister, Dong Liu
    "SPAN: Fast and Simple Learnable Spectral Transformation for Super-Resolution"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PCFN(nn.Module):
    """Pure ConvFFN (Pointwise Convolution Feed-Forward Network)"""
    def __init__(self, dim, ffn_expansion_factor=2):
        super(PCFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, 
                               stride=1, padding=1, groups=hidden_features * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class DSPAN(nn.Module):
    """Depth-wise Separable Parameter-free Attention"""
    def __init__(self, dim):
        super(DSPAN, self).__init__()
        self.norm = nn.BatchNorm2d(dim)
        
        # Depthwise convolution
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        
        # Spatial convolutions
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3)
        
        # Channel convolutions  
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        
        # Output projection
        self.conv_squeeze = nn.Conv2d(2, 2, kernel_size=7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, kernel_size=1)

    def forward(self, x):
        x = self.norm(x)
        
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        
        return x * attn


class SPANBlock(nn.Module):
    """SPAN Block = DSPAN + PCFN"""
    def __init__(self, dim, ffn_expansion_factor=2):
        super(SPANBlock, self).__init__()
        
        self.attn = DSPAN(dim)
        self.ffn = PCFN(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


@ARCH_REGISTRY.register()
class SPAN(nn.Module):
    """
    SPAN: Swift Parameter-free Attention Network for Efficient Super-Resolution
    
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        feature_channels (int): Channel number of intermediate features. Default: 48.
        upscale (int): Upsampling factor. Support 2, 3, 4. Default: 2.
        num_blocks (int): Number of SPAN blocks. Default: 12.
        ffn_expansion_factor (float): FFN expansion factor. Default: 2.0.
        bias (bool): Whether to use bias in convolution layers. Default: False.
    """

    def __init__(self, 
                 num_in_ch=3, 
                 num_out_ch=3, 
                 feature_channels=48,
                 upscale=2, 
                 num_blocks=12,
                 ffn_expansion_factor=2.0,
                 bias=False):
        super(SPAN, self).__init__()
        
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.feature_channels = feature_channels
        self.upscale = upscale
        self.num_blocks = num_blocks
        
        # Input projection
        self.conv_first = nn.Conv2d(num_in_ch, feature_channels, kernel_size=3, 
                                    stride=1, padding=1, bias=bias)
        
        # SPAN blocks
        self.blocks = nn.ModuleList([
            SPANBlock(feature_channels, ffn_expansion_factor) 
            for _ in range(num_blocks)
        ])
        
        # Middle convolution
        self.conv_body = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, 
                                   stride=1, padding=1, bias=bias)
        
        # Upsampling
        if upscale == 2 or upscale == 3:
            self.upsampler = nn.Sequential(
                nn.Conv2d(feature_channels, num_out_ch * (upscale ** 2), kernel_size=3, 
                         stride=1, padding=1, bias=bias),
                nn.PixelShuffle(upscale)
            )
        elif upscale == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels * 4, kernel_size=3, 
                         stride=1, padding=1, bias=bias),
                nn.PixelShuffle(2),
                nn.Conv2d(feature_channels, num_out_ch * 4, kernel_size=3, 
                         stride=1, padding=1, bias=bias),
                nn.PixelShuffle(2)
            )
        else:
            raise ValueError(f'Unsupported upscale factor: {upscale}. Only 2, 3, 4 are supported.')

    def forward(self, x):
        # Shallow feature extraction
        x = self.conv_first(x)
        
        # Store for residual connection
        shortcut = x.clone()
        
        # Deep feature extraction with SPAN blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.conv_body(x)
        
        # Global residual connection
        x = x + shortcut
        
        # Upsampling
        x = self.upsampler(x)
        
        return x


@ARCH_REGISTRY.register()
class SPANPlus(SPAN):
    """
    Enhanced SPAN for Real-ESRGAN training.
    Supports more feature channels and blocks for better quality.
    """
    def __init__(self, 
                 num_in_ch=3, 
                 num_out_ch=3, 
                 feature_channels=64,  # Increased from 48
                 upscale=2, 
                 num_blocks=24,  # Increased from 12
                 ffn_expansion_factor=2.0,
                 bias=False):
        super(SPANPlus, self).__init__(
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            feature_channels=feature_channels,
            upscale=upscale,
            num_blocks=num_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias
        )


if __name__ == '__main__':
    # Test the model
    model = SPAN(upscale=2, feature_channels=48, num_blocks=12)
    print(f'SPAN parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {y.shape}')
    
    model_plus = SPANPlus(upscale=2, feature_channels=64, num_blocks=24)
    print(f'SPANPlus parameters: {sum(p.numel() for p in model_plus.parameters()) / 1e6:.2f}M')

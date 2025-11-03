#!/usr/bin/env python3
"""
??????Real-ESRGAN???FLOPs?GMACs
?????????????????
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ???????????????
class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution."""
    
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == 'relu':
                activation = nn.ReLU(inplace=True)
            elif act_type == 'prelu':
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == 'leakyrelu':
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
        out += base
        return out

def calculate_conv2d_flops(in_channels, out_channels, kernel_size, input_h, input_w):
    """
    ????Conv2d??FLOPs?MACs
    
    ??Conv2d:
    - MACs = ???? ? ???? ? ????? ? ????? ? ????? ? ?????
    - FLOPs = 2 ? MACs (????MAC??1????1???)
    """
    # ??padding?stride??????????
    output_h, output_w = input_h, input_w
    
    # ??MACs
    macs = output_h * output_w * kernel_size * kernel_size * in_channels * out_channels
    
    # ??FLOPs
    flops = 2 * macs
    
    return macs, flops

def analyze_model_complexity():
    print("="*70)
    print("Real-ESRGAN ??????? (????)")
    print("="*70)
    
    # ???? - realesr-animevideov3 (AnimeVideo-v3)
    num_in_ch = 3
    num_out_ch = 3
    num_feat = 64
    num_conv = 16
    upscale = 4
    
    # ?????
    resolutions = [
        (480, 640, "640?480 (VGA)"),
        (720, 1280, "1280?720 (HD)"),
        (1080, 1920, "1920?1080 (Full HD)")
    ]
    
    print(f"\n????:")
    print(f"  - ?????: {num_in_ch}")
    print(f"  - ?????: {num_out_ch}")
    print(f"  - ???????: {num_feat}")
    print(f"  - ?????: {num_conv + 2} (??? + {num_conv}?body + ????)")
    print(f"  - ?????: {upscale}x")
    
    for h, w, res_name in resolutions:
        print(f"\n{'='*70}")
        print(f"?????: {res_name}")
        print(f"{'='*70}")
        
        total_macs = 0
        total_flops = 0
        
        # ???: 3 -> 64
        macs, flops = calculate_conv2d_flops(num_in_ch, num_feat, 3, h, w)
        total_macs += macs
        total_flops += flops
        print(f"\n?1? Conv (3?64):")
        print(f"  MACs:  {macs:>15,} ({macs:.3e})")
        print(f"  FLOPs: {flops:>15,} ({flops:.3e})")
        
        # Body?: 64 -> 64 (??num_conv?)
        macs, flops = calculate_conv2d_flops(num_feat, num_feat, 3, h, w)
        body_macs = macs * num_conv
        body_flops = flops * num_conv
        total_macs += body_macs
        total_flops += body_flops
        print(f"\nBody {num_conv}? Conv (64?64 ?{num_conv}):")
        print(f"  MACs:  {body_macs:>15,} ({body_macs:.3e})")
        print(f"  FLOPs: {body_flops:>15,} ({body_flops:.3e})")
        
        # ????: 64 -> 48 (3 * 4 * 4 for upscaling)
        macs, flops = calculate_conv2d_flops(num_feat, num_out_ch * upscale * upscale, 3, h, w)
        total_macs += macs
        total_flops += flops
        print(f"\n???? Conv (64?{num_out_ch * upscale * upscale}):")
        print(f"  MACs:  {macs:>15,} ({macs:.3e})")
        print(f"  FLOPs: {flops:>15,} ({flops:.3e})")
        
        # ??
        gmacs = total_macs / 1e9
        gflops = total_flops / 1e9
        
        print(f"\n{'-'*70}")
        print(f"??:")
        print(f"{'-'*70}")
        print(f"  ?MACs:   {total_macs:>15,} = {gmacs:>8.3f} GMACs")
        print(f"  ?FLOPs:  {total_flops:>15,} = {gflops:>8.3f} GFLOPs")
        print(f"\n  ??: FLOPs / MACs = {total_flops / total_macs:.2f} (?? = 2)")
        print(f"  ??: GFLOPs / GMACs = {gflops / gmacs:.2f} (?? = 2)")
    
    print(f"\n{'='*70}")
    print("??:")
    print(f"{'='*70}")
    print("1. FLOPs (??????) = 2 ? MACs (??????)")
    print("2. ?? 1 MAC = 1??? + 1??? = 2?????")
    print("3. ?? GFLOPs = 2 ? GMACs")
    print("4. ??????????FLOPs ? GMACs ???????")
    print("5. ?????????????????????")
    print(f"{'='*70}")

def test_with_thop():
    """??thop????????????"""
    try:
        from thop import profile, clever_format
        
        print(f"\n\n{'='*70}")
        print("?? thop ???????")
        print(f"{'='*70}")
        
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4)
        model.eval()
        
        input_tensor = torch.randn(1, 3, 480, 640)
        
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops = 2 * macs
        
        macs_str, params_str = clever_format([macs, params], "%.3f")
        flops_str = clever_format([flops], "%.3f")[0]
        gmacs = macs / 1e9
        gflops = flops / 1e9
        
        print(f"\n?????: 640?480")
        print(f"  ???: {params_str}")
        print(f"  MACs: {macs_str} ({macs:.3e}) = {gmacs:.3f} GMACs")
        print(f"  FLOPs: {flops_str} ({flops:.3e}) = {gflops:.3f} GFLOPs")
        print(f"  ??: FLOPs / MACs = {flops / macs:.2f}")
        
        return True
    except ImportError:
        print(f"\n{'='*70}")
        print("thop ????????")
        print("????: pip install thop")
        print(f"{'='*70}")
        return False

if __name__ == '__main__':
    # ????
    analyze_model_complexity()
    
    # ????thop??
    test_with_thop()

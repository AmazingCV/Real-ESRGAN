#!/usr/bin/env python3
"""
??Real-ESRGAN???FLOPs?GMACs
??FLOPs?GMACs?????
"""

import torch
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("??: thop????????fvcore")
    
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("??: fvcore???")

from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def calculate_with_thop(model, input_tensor):
    """??thop??FLOPs????"""
    print("\n" + "="*60)
    print("?? thop ???:")
    print("="*60)
    
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2 * macs  # 1 MAC = 2 FLOPs (1??? + 1???)
    
    # ???????
    macs_str, params_str = clever_format([macs, params], "%.3f")
    flops_str = clever_format([flops], "%.3f")[0]
    gmacs = macs / 1e9
    gflops = flops / 1e9
    
    print(f"??? (Params): {params_str}")
    print(f"MACs: {macs_str} ({macs:.2e})")
    print(f"GMACs: {gmacs:.3f}G")
    print(f"FLOPs: {flops_str} ({flops:.2e})")
    print(f"GFLOPs: {gflops:.3f}G")
    print(f"\n????: FLOPs / MACs = {flops / macs:.2f} (?????2)")
    print(f"????: GFLOPs / GMACs = {gflops / gmacs:.2f} (?????2)")
    
    return {'macs': macs, 'flops': flops, 'params': params, 'gmacs': gmacs, 'gflops': gflops}

def calculate_with_fvcore(model, input_tensor):
    """??fvcore??FLOPs"""
    print("\n" + "="*60)
    print("?? fvcore ???:")
    print("="*60)
    
    flops_analyzer = FlopCountAnalysis(model, input_tensor)
    total_flops = flops_analyzer.total()
    
    # fvcore????MACs????FLOPs?????MACs?
    # ???https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md
    macs = total_flops
    flops = 2 * macs
    gmacs = macs / 1e9
    gflops = flops / 1e9
    
    print(f"MACs (fvcore??FLOPs): {macs:.2e}")
    print(f"GMACs: {gmacs:.3f}G")
    print(f"??FLOPs (2*MACs): {flops:.2e}")
    print(f"GFLOPs: {gflops:.3f}G")
    print(f"\n????: FLOPs / MACs = {flops / macs:.2f} (????2)")
    
    # ??????
    print("\n????:")
    print(flop_count_table(flops_analyzer))
    
    return {'macs': macs, 'flops': flops, 'gmacs': gmacs, 'gflops': gflops}

def main():
    print("="*60)
    print("Real-ESRGAN ???????")
    print("="*60)
    
    # ?????????
    configs = [
        {
            'name': 'RealESRGAN_x4plus_anime_6B',
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_conv': 16,
            'upscale': 4
        },
        {
            'name': 'realesr-animevideov3 (?????)',
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_conv': 16,
            'upscale': 4
        }
    ]
    
    # ??????????
    resolutions = [
        (480, 640, "640x480"),
        (720, 1280, "1280x720"),
        (1080, 1920, "1920x1080")
    ]
    
    for config in configs[:1]:  # ????????
        print(f"\n{'#'*60}")
        print(f"??: {config['name']}")
        print(f"{'#'*60}")
        
        # ????
        model = SRVGGNetCompact(
            num_in_ch=config['num_in_ch'],
            num_out_ch=config['num_out_ch'],
            num_feat=config['num_feat'],
            num_conv=config['num_conv'],
            upscale=config['upscale']
        )
        model.eval()
        
        for h, w, res_name in resolutions[:1]:  # ?????????
            print(f"\n?????: {res_name}")
            print("-"*60)
            
            # ??????
            input_tensor = torch.randn(1, 3, h, w)
            
            # ??thop??
            if THOP_AVAILABLE:
                try:
                    thop_results = calculate_with_thop(model, input_tensor)
                except Exception as e:
                    print(f"thop????: {e}")
            else:
                print("\nthop??????")
            
            # ??fvcore??
            if FVCORE_AVAILABLE:
                try:
                    fvcore_results = calculate_with_fvcore(model, input_tensor)
                except Exception as e:
                    print(f"fvcore????: {e}")
            else:
                print("\nfvcore??????")
    
    print("\n" + "="*60)
    print("??:")
    print("="*60)
    print("1. FLOPs (??????) = 2 ? MACs (??????)")
    print("2. ??1?MAC??1????1?????2?????")
    print("3. ?? GFLOPs = 2 ? GMACs")
    print("4. ????????FLOPs?GMACs???????")
    print("="*60)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
测试SPAN网络是否正常工作
"""

import torch
from realesrgan.archs.span_arch import SPAN, SPANPlus


def test_span():
    """测试标准SPAN网络"""
    print("=" * 60)
    print("测试标准SPAN网络 (x2超分)")
    print("=" * 60)
    
    # 创建模型
    model = SPAN(
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=48,
        upscale=2,
        num_blocks=12
    )
    
    # 测试前向传播
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    
    # 输出信息
    print(f"✅ SPAN网络测试通过!")
    print(f"   输入尺寸: {x.shape}")
    print(f"   输出尺寸: {y.shape}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()
    
    return model


def test_spanplus():
    """测试增强版SPANPlus网络"""
    print("=" * 60)
    print("测试增强版SPANPlus网络 (x2超分)")
    print("=" * 60)
    
    # 创建模型
    model = SPANPlus(
        num_in_ch=3,
        num_out_ch=3,
        feature_channels=64,
        upscale=2,
        num_blocks=24
    )
    
    # 测试前向传播
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    
    # 输出信息
    print(f"✅ SPANPlus网络测试通过!")
    print(f"   输入尺寸: {x.shape}")
    print(f"   输出尺寸: {y.shape}")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()
    
    return model


def test_architecture_registry():
    """测试网络是否已注册到basicsr"""
    print("=" * 60)
    print("检查网络注册状态")
    print("=" * 60)
    
    try:
        from basicsr.utils.registry import ARCH_REGISTRY
        
        span_registered = 'SPAN' in ARCH_REGISTRY._obj_map
        spanplus_registered = 'SPANPlus' in ARCH_REGISTRY._obj_map
        
        print(f"SPAN注册状态: {'✅ 已注册' if span_registered else '❌ 未注册'}")
        print(f"SPANPlus注册状态: {'✅ 已注册' if spanplus_registered else '❌ 未注册'}")
        
        if span_registered or spanplus_registered:
            print("\n可用的SPAN相关架构:")
            for name in ARCH_REGISTRY._obj_map.keys():
                if 'SPAN' in name or 'span' in name:
                    print(f"  - {name}")
        print()
        
    except Exception as e:
        print(f"❌ 检查注册状态时出错: {e}")
        print()


def compare_models():
    """对比不同模型的参数量和计算复杂度"""
    print("=" * 60)
    print("模型对比")
    print("=" * 60)
    
    models_config = [
        ('SPAN', {'feature_channels': 48, 'num_blocks': 12}),
        ('SPANPlus', {'feature_channels': 64, 'num_blocks': 24}),
    ]
    
    print(f"{'模型':<15} {'参数量':<15} {'特征通道':<15} {'块数量':<10}")
    print("-" * 60)
    
    for name, config in models_config:
        if name == 'SPAN':
            model = SPAN(upscale=2, **config)
        else:
            model = SPANPlus(upscale=2, **config)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name:<15} {params:>6.2f}M{'':<8} {config['feature_channels']:<15} {config['num_blocks']:<10}")
    
    print()


if __name__ == '__main__':
    try:
        # 测试SPAN
        test_span()
        
        # 测试SPANPlus
        test_spanplus()
        
        # 检查注册状态
        test_architecture_registry()
        
        # 模型对比
        compare_models()
        
        print("=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        print("\n下一步:")
        print("1. 准备训练数据集")
        print("2. 运行训练: python realesrgan/train.py -opt options/train_realesrgan_x2plus_span.yml")
        print("3. 使用训练好的模型推理: python inference_realesrgan_span.py -n span_x2 -i inputs -o results")
        print("\n详细使用说明请查看: docs/SPAN_USAGE.md")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

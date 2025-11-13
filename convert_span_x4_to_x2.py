#!/usr/bin/env python3
"""
将SPAN x4模型权重转换为x2模型权重
用于从预训练的x4模型迁移学习到x2模型
"""

import argparse
import torch
from realesrgan.archs.span_arch import SPAN, SPANPlus


def convert_span_weights(x4_path, output_path, model_type='SPAN', feature_channels=48, num_blocks=12):
    """
    转换SPAN x4权重到x2
    
    Args:
        x4_path: x4模型权重路径
        output_path: 输出x2模型权重路径
        model_type: 模型类型 ('SPAN' 或 'SPANPlus')
        feature_channels: 特征通道数
        num_blocks: SPAN块数量
    """
    print(f"正在从 {x4_path} 加载x4模型权重...")
    
    # 加载x4模型权重
    checkpoint_x4 = torch.load(x4_path, map_location='cpu')
    
    # 提取state_dict
    if isinstance(checkpoint_x4, dict):
        if 'params_ema' in checkpoint_x4:
            state_dict_x4 = checkpoint_x4['params_ema']
        elif 'params' in checkpoint_x4:
            state_dict_x4 = checkpoint_x4['params']
        elif 'state_dict' in checkpoint_x4:
            state_dict_x4 = checkpoint_x4['state_dict']
        else:
            state_dict_x4 = checkpoint_x4
    else:
        state_dict_x4 = checkpoint_x4
    
    print(f"x4模型包含 {len(state_dict_x4)} 个参数")
    
    # 创建x2模型
    print(f"创建{model_type} x2模型...")
    if model_type == 'SPAN':
        model_x2 = SPAN(
            num_in_ch=3,
            num_out_ch=3,
            feature_channels=feature_channels,
            upscale=2,
            num_blocks=num_blocks
        )
    else:  # SPANPlus
        model_x2 = SPANPlus(
            num_in_ch=3,
            num_out_ch=3,
            feature_channels=feature_channels,
            upscale=2,
            num_blocks=num_blocks
        )
    
    state_dict_x2 = model_x2.state_dict()
    
    # 过滤掉upsampler的权重，只保留特征提取部分
    filtered_dict = {}
    skipped_keys = []
    
    for k, v in state_dict_x4.items():
        if k in state_dict_x2 and 'upsampler' not in k:
            # 检查形状是否匹配
            if v.shape == state_dict_x2[k].shape:
                filtered_dict[k] = v
            else:
                skipped_keys.append(f"{k} (形状不匹配: {v.shape} vs {state_dict_x2[k].shape})")
        else:
            if 'upsampler' in k:
                skipped_keys.append(f"{k} (上采样层，跳过)")
            else:
                skipped_keys.append(f"{k} (不在x2模型中)")
    
    print(f"\n成功转换 {len(filtered_dict)} 个参数")
    print(f"跳过 {len(skipped_keys)} 个参数:")
    for key in skipped_keys[:10]:  # 只显示前10个
        print(f"  - {key}")
    if len(skipped_keys) > 10:
        print(f"  ... 还有 {len(skipped_keys) - 10} 个")
    
    # 更新x2模型权重
    state_dict_x2.update(filtered_dict)
    model_x2.load_state_dict(state_dict_x2, strict=False)
    
    # 保存转换后的权重
    print(f"\n保存x2模型权重到 {output_path}...")
    torch.save({
        'params_ema': model_x2.state_dict(),
        'conversion_info': {
            'source': x4_path,
            'model_type': model_type,
            'feature_channels': feature_channels,
            'num_blocks': num_blocks,
            'converted_params': len(filtered_dict),
            'skipped_params': len(skipped_keys)
        }
    }, output_path)
    
    print("✅ 转换完成!")
    print(f"\n使用方法:")
    print(f"在训练配置文件中设置:")
    print(f"  pretrain_network_g: {output_path}")
    print(f"  strict_load_g: false")


def main():
    parser = argparse.ArgumentParser(description='将SPAN x4模型权重转换为x2')
    parser.add_argument('--x4_path', type=str, required=True, help='x4模型权重路径')
    parser.add_argument('--output', type=str, required=True, help='输出x2模型权重路径')
    parser.add_argument('--model_type', type=str, default='SPAN', 
                       choices=['SPAN', 'SPANPlus'], help='模型类型')
    parser.add_argument('--feature_channels', type=int, default=48,
                       help='特征通道数 (SPAN: 48, SPANPlus: 64)')
    parser.add_argument('--num_blocks', type=int, default=12,
                       help='SPAN块数量 (SPAN: 12, SPANPlus: 24)')
    
    args = parser.parse_args()
    
    convert_span_weights(
        x4_path=args.x4_path,
        output_path=args.output,
        model_type=args.model_type,
        feature_channels=args.feature_channels,
        num_blocks=args.num_blocks
    )


if __name__ == '__main__':
    main()

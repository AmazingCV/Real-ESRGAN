#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外图像训练准备工具
自动化准备Real-ESRGAN红外图像训练的完整流程
"""

import os
import sys
import argparse
import subprocess
import yaml
from pathlib import Path


def check_dependencies():
    """检查必要的依赖"""
    print("检查依赖...")
    required_packages = ['cv2', 'numpy', 'torch', 'yaml']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"缺少以下依赖: {', '.join(missing)}")
        print("请运行: pip install opencv-python numpy torch pyyaml")
        return False
    
    print("✓ 依赖检查通过")
    return True


def check_image_directory(directory):
    """检查图像目录"""
    if not os.path.isdir(directory):
        return False, f"目录不存在: {directory}"
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    images = [f for f in os.listdir(directory) 
              if f.lower().endswith(image_extensions)]
    
    if not images:
        return False, f"目录中没有找到图像文件: {directory}"
    
    return True, f"找到 {len(images)} 张图像"


def generate_meta_info(input_dir, output_file):
    """生成meta_info.txt文件"""
    print(f"\n生成meta_info文件: {output_file}")
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    images = sorted([f for f in os.listdir(input_dir) 
                     if f.lower().endswith(image_extensions)])
    
    if not images:
        print(f"错误: 在 {input_dir} 中没有找到图像文件")
        return False
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入meta_info
    with open(output_file, 'w') as f:
        for img in images:
            f.write(f"{img}\n")
    
    print(f"✓ 已生成meta_info文件，包含 {len(images)} 张图像")
    return True


def estimate_degradation_params(hr_dir, lr_dir):
    """运行退化参数估计"""
    print("\n" + "="*60)
    print("运行退化参数估计...")
    print("="*60)
    
    cmd = [
        sys.executable,
        'estimate_degradation_params.py',
        '--hr_dir', hr_dir,
        '--lr_dir', lr_dir,
        '--visualize',
        '--output', 'degradation_analysis.png'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n✓ 退化参数估计完成")
        print("✓ 分析图表已保存到: degradation_analysis.png")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: 退化参数估计失败")
        return False


def update_config_file(config_template, output_config, params):
    """更新配置文件参数"""
    print(f"\n更新配置文件: {output_config}")
    
    with open(config_template, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新数据集路径
    if 'hr_dir' in params:
        config['datasets']['train']['dataroot_gt'] = params['hr_dir']
    
    if 'meta_info' in params:
        config['datasets']['train']['meta_info'] = params['meta_info']
    
    # 更新通道数
    if 'num_channels' in params:
        config['network_g']['num_in_ch'] = params['num_channels']
        config['network_g']['num_out_ch'] = params['num_channels']
        config['network_d']['num_in_ch'] = params['num_channels']
    
    # 保存更新后的配置
    with open(output_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ 配置文件已更新: {output_config}")
    return True


def create_directory_structure(base_dir):
    """创建训练所需的目录结构"""
    print(f"\n创建目录结构在: {base_dir}")
    
    dirs = [
        'train',  # HR训练图像
        'val/hr',  # 验证集HR
        'val/lr',  # 验证集LR
    ]
    
    for d in dirs:
        path = os.path.join(base_dir, d)
        os.makedirs(path, exist_ok=True)
        print(f"  创建: {path}")
    
    print("✓ 目录结构创建完成")
    return True


def print_next_steps(config_file, has_lr_images):
    """打印下一步操作指南"""
    print("\n" + "="*60)
    print("准备工作完成！下一步操作:")
    print("="*60)
    
    if has_lr_images:
        print("\n1. 查看退化参数分析结果:")
        print("   - 查看终端输出的参数建议")
        print("   - 打开 degradation_analysis.png 查看可视化分析")
        
        print(f"\n2. 手动调整配置文件 {config_file}:")
        print("   根据分析结果修改以下参数（查找 '← 修改这里！'）:")
        print("   - blur_sigma, blur_sigma2")
        print("   - noise_range, noise_range2")
        print("   - poisson_scale_range, poisson_scale_range2")
        print("   - jpeg_range, jpeg_range2")
    else:
        print(f"\n1. 手动编辑配置文件 {config_file}")
        print("   根据你对红外图像的了解调整退化参数")
    
    print("\n3. (可选) 准备验证集:")
    print("   将验证图像放到对应目录:")
    print("   - HR验证图像 -> datasets/infrared/val/hr/")
    print("   - LR验证图像 -> datasets/infrared/val/lr/")
    
    print("\n4. 下载预训练模型 (如果需要):")
    print("   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRNet_x4plus.pth")
    print("   mv RealESRNet_x4plus.pth experiments/pretrained_models/")
    
    print("\n5. 开始训练:")
    print(f"   python realesrgan/train.py -opt {config_file}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='红外图像训练准备工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 完整流程（有LR和HR图像对）:
   python prepare_infrared_training.py \\
       --hr_dir ./my_infrared_images/hr \\
       --lr_dir ./my_infrared_images/lr \\
       --output_dir ./datasets/infrared \\
       --estimate

2. 仅准备目录结构（只有HR图像）:
   python prepare_infrared_training.py \\
       --hr_dir ./my_infrared_images \\
       --output_dir ./datasets/infrared \\
       --no-estimate

3. 指定图像通道数:
   python prepare_infrared_training.py \\
       --hr_dir ./my_infrared_images/hr \\
       --output_dir ./datasets/infrared \\
       --channels 1  # 灰度图像
        """
    )
    
    parser.add_argument('--hr_dir', type=str, required=True,
                        help='HR（高分辨率）图像目录')
    parser.add_argument('--lr_dir', type=str,
                        help='LR（低分辨率）图像目录（用于参数估计）')
    parser.add_argument('--output_dir', type=str, default='./datasets/infrared',
                        help='输出目录（默认: ./datasets/infrared）')
    parser.add_argument('--channels', type=int, choices=[1, 3], default=3,
                        help='图像通道数: 1=灰度, 3=RGB（默认: 3）')
    parser.add_argument('--estimate', action='store_true',
                        help='运行退化参数估计（需要--lr_dir）')
    parser.add_argument('--no-estimate', dest='estimate', action='store_false',
                        help='跳过退化参数估计')
    parser.set_defaults(estimate=True)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Real-ESRGAN 红外图像训练准备工具")
    print("="*60)
    
    # 1. 检查依赖
    if not check_dependencies():
        return
    
    # 2. 检查HR图像目录
    print(f"\n检查HR图像目录: {args.hr_dir}")
    success, msg = check_image_directory(args.hr_dir)
    if not success:
        print(f"错误: {msg}")
        return
    print(f"✓ {msg}")
    
    # 3. 检查LR图像目录（如果提供）
    has_lr_images = False
    if args.lr_dir:
        print(f"\n检查LR图像目录: {args.lr_dir}")
        success, msg = check_image_directory(args.lr_dir)
        if not success:
            print(f"警告: {msg}")
            print("将跳过退化参数估计")
            args.estimate = False
        else:
            print(f"✓ {msg}")
            has_lr_images = True
    elif args.estimate:
        print("\n警告: 未提供LR图像目录，无法运行退化参数估计")
        args.estimate = False
    
    # 4. 创建输出目录结构
    create_directory_structure(args.output_dir)
    
    # 5. 生成meta_info文件
    meta_info_file = os.path.join(args.output_dir, 'meta_info.txt')
    if not generate_meta_info(args.hr_dir, meta_info_file):
        print("错误: 生成meta_info文件失败")
        return
    
    # 6. 运行退化参数估计（如果需要）
    if args.estimate and has_lr_images:
        if not estimate_degradation_params(args.hr_dir, args.lr_dir):
            print("警告: 退化参数估计失败，但可以继续手动配置")
    
    # 7. 准备配置文件
    config_template = 'options/train_realesrgan_x4plus_infrared.yml'
    output_config = os.path.join(args.output_dir, 'train_config.yml')
    
    if os.path.exists(config_template):
        params = {
            'hr_dir': args.hr_dir,
            'meta_info': meta_info_file,
            'num_channels': args.channels,
        }
        update_config_file(config_template, output_config, params)
    else:
        print(f"\n警告: 配置模板不存在: {config_template}")
        print("请手动创建配置文件")
    
    # 8. 打印下一步指南
    print_next_steps(output_config, has_lr_images)
    
    print("\n✓ 准备工作全部完成！")


if __name__ == '__main__':
    main()

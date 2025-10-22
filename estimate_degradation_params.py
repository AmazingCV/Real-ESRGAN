#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
估计退化参数的工具脚本
通过分析真实的LR/HR图像对，估计合适的训练参数范围
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


class DegradationEstimator:
    """退化参数估计器"""
    
    def __init__(self):
        self.results = []
        
    def load_image_pair(self, hr_path, lr_path):
        """加载HR和LR图像对"""
        hr_img = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
        lr_img = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
        
        if hr_img is None:
            raise ValueError(f"无法读取HR图像: {hr_path}")
        if lr_img is None:
            raise ValueError(f"无法读取LR图像: {lr_path}")
            
        # 转换为浮点数 [0, 1]
        if hr_img.dtype == np.uint16:
            hr_img = hr_img.astype(np.float32) / 65535.0
        else:
            hr_img = hr_img.astype(np.float32) / 255.0
            
        if lr_img.dtype == np.uint16:
            lr_img = lr_img.astype(np.float32) / 65535.0
        else:
            lr_img = lr_img.astype(np.float32) / 255.0
            
        return hr_img, lr_img
    
    def estimate_scale_factor(self, hr_img, lr_img):
        """估计缩放比例"""
        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = lr_img.shape[:2]
        
        scale_h = hr_h / lr_h
        scale_w = hr_w / lr_w
        scale = (scale_h + scale_w) / 2
        
        return scale, scale_h, scale_w
    
    def estimate_blur_sigma(self, img):
        """
        估计图像的模糊程度 (sigma)
        使用拉普拉斯方差方法
        """
        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)
        
        # 计算拉普拉斯方差
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 计算频域能量分布
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # 计算高频能量比例
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        # 创建高频掩码
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 > radius**2
        
        high_freq_energy = np.sum(magnitude_spectrum[mask])
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        return laplacian_var, high_freq_ratio
    
    def estimate_noise_level(self, img):
        """
        估计噪声水平
        使用中值绝对偏差(MAD)方法
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = img
        
        # 使用高通滤波提取噪声
        # Laplacian kernel
        noise = cv2.Laplacian(gray, cv2.CV_64F)
        
        # MAD估计
        sigma = np.median(np.abs(noise - np.median(noise))) / 0.6745
        
        # 转换为0-255范围的噪声水平
        noise_level = sigma * 255
        
        return noise_level
    
    def estimate_jpeg_quality(self, img):
        """
        估计JPEG压缩质量
        通过检测块效应
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (img * 255).astype(np.uint8)
        
        h, w = gray.shape
        block_size = 8
        
        # 计算水平和垂直块边界的不连续性
        h_diff = 0
        v_diff = 0
        h_count = 0
        v_count = 0
        
        # 检测垂直块边界
        for i in range(block_size, h, block_size):
            if i < h - 1:
                diff = np.abs(gray[i, :].astype(np.float32) - gray[i-1, :].astype(np.float32))
                h_diff += np.mean(diff)
                h_count += 1
        
        # 检测水平块边界
        for j in range(block_size, w, block_size):
            if j < w - 1:
                diff = np.abs(gray[:, j].astype(np.float32) - gray[:, j-1].astype(np.float32))
                v_diff += np.mean(diff)
                v_count += 1
        
        avg_h_diff = h_diff / h_count if h_count > 0 else 0
        avg_v_diff = v_diff / v_count if v_count > 0 else 0
        block_artifact = (avg_h_diff + avg_v_diff) / 2
        
        # 根据块效应估计JPEG质量
        # 块效应越大，质量越低
        if block_artifact > 5:
            estimated_quality = 30 + (100 - 30) * (1 - min(block_artifact / 20, 1))
        else:
            estimated_quality = 70 + (95 - 70) * (1 - min(block_artifact / 5, 1))
        
        return estimated_quality, block_artifact
    
    def analyze_pair(self, hr_path, lr_path):
        """分析一对HR/LR图像"""
        print(f"\n{'='*60}")
        print(f"分析图像对:")
        print(f"  HR: {hr_path}")
        print(f"  LR: {lr_path}")
        print(f"{'='*60}")
        
        # 加载图像
        hr_img, lr_img = self.load_image_pair(hr_path, lr_path)
        
        # 1. 缩放比例
        scale, scale_h, scale_w = self.estimate_scale_factor(hr_img, lr_img)
        print(f"\n1. 缩放比例分析:")
        print(f"   平均缩放: {scale:.2f}x")
        print(f"   高度缩放: {scale_h:.2f}x")
        print(f"   宽度缩放: {scale_w:.2f}x")
        
        # 2. 模糊程度分析
        hr_lap_var, hr_hf_ratio = self.estimate_blur_sigma(hr_img)
        lr_lap_var, lr_hf_ratio = self.estimate_blur_sigma(lr_img)
        
        print(f"\n2. 模糊程度分析:")
        print(f"   HR图像 - 拉普拉斯方差: {hr_lap_var:.2f}, 高频比例: {hr_hf_ratio:.4f}")
        print(f"   LR图像 - 拉普拉斯方差: {lr_lap_var:.2f}, 高频比例: {lr_hf_ratio:.4f}")
        
        # 根据高频能量比例估计模糊sigma
        # 高频比例下降越多，模糊越严重
        if hr_hf_ratio > 0:
            blur_ratio = lr_hf_ratio / hr_hf_ratio
            # 经验公式：sigma大致与高频能量损失相关
            estimated_sigma = max(0.2, min(3.0, (1 - blur_ratio) * 5))
        else:
            estimated_sigma = 0.5
        
        print(f"   估计的模糊sigma: {estimated_sigma:.2f}")
        
        # 3. 噪声水平分析
        hr_noise = self.estimate_noise_level(hr_img)
        lr_noise = self.estimate_noise_level(lr_img)
        
        print(f"\n3. 噪声水平分析:")
        print(f"   HR图像噪声: {hr_noise:.2f}")
        print(f"   LR图像噪声: {lr_noise:.2f}")
        print(f"   LR额外噪声: {max(0, lr_noise - hr_noise):.2f}")
        
        # 4. JPEG压缩质量分析
        lr_jpeg_quality, lr_block_artifact = self.estimate_jpeg_quality(lr_img)
        
        print(f"\n4. JPEG压缩质量分析 (LR图像):")
        print(f"   块效应强度: {lr_block_artifact:.2f}")
        print(f"   估计JPEG质量: {lr_jpeg_quality:.0f}")
        
        # 保存结果
        result = {
            'hr_path': hr_path,
            'lr_path': lr_path,
            'scale': scale,
            'blur_sigma': estimated_sigma,
            'noise_level': lr_noise,
            'jpeg_quality': lr_jpeg_quality,
            'hr_laplacian_var': hr_lap_var,
            'lr_laplacian_var': lr_lap_var,
            'hr_high_freq_ratio': hr_hf_ratio,
            'lr_high_freq_ratio': lr_hf_ratio,
        }
        
        self.results.append(result)
        return result
    
    def summarize_results(self):
        """汇总所有分析结果，给出参数建议"""
        if not self.results:
            print("没有分析结果")
            return
        
        print(f"\n{'='*60}")
        print("汇总分析结果与参数建议")
        print(f"{'='*60}")
        print(f"\n分析了 {len(self.results)} 对图像\n")
        
        # 提取所有参数
        scales = [r['scale'] for r in self.results]
        blur_sigmas = [r['blur_sigma'] for r in self.results]
        noise_levels = [r['noise_level'] for r in self.results]
        jpeg_qualities = [r['jpeg_quality'] for r in self.results]
        
        # 计算统计信息
        print("统计信息:")
        print(f"  缩放比例: 平均={np.mean(scales):.2f}, 范围=[{np.min(scales):.2f}, {np.max(scales):.2f}]")
        print(f"  模糊sigma: 平均={np.mean(blur_sigmas):.2f}, 范围=[{np.min(blur_sigmas):.2f}, {np.max(blur_sigmas):.2f}]")
        print(f"  噪声水平: 平均={np.mean(noise_levels):.2f}, 范围=[{np.min(noise_levels):.2f}, {np.max(noise_levels):.2f}]")
        print(f"  JPEG质量: 平均={np.mean(jpeg_qualities):.0f}, 范围=[{np.min(jpeg_qualities):.0f}, {np.max(jpeg_qualities):.0f}]")
        
        # 给出配置文件参数建议
        print(f"\n{'='*60}")
        print("推荐的配置文件参数:")
        print(f"{'='*60}\n")
        
        # 扩展范围以涵盖变化
        blur_min = max(0.2, np.min(blur_sigmas) * 0.7)
        blur_max = min(3.0, np.max(blur_sigmas) * 1.3)
        
        noise_min = max(1, int(np.min(noise_levels) * 0.5))
        noise_max = min(30, int(np.max(noise_levels) * 1.5))
        
        jpeg_min = max(30, int(np.min(jpeg_qualities) * 0.8))
        jpeg_max = min(95, int(np.max(jpeg_qualities) * 1.1))
        
        print("# 第一次退化过程")
        print(f"blur_sigma: [{blur_min:.1f}, {blur_max:.1f}]  # 模糊核标准差")
        print(f"noise_range: [{noise_min}, {noise_max}]  # 高斯噪声范围")
        print(f"poisson_scale_range: [{noise_min*0.05:.2f}, {noise_max*0.1:.2f}]  # 泊松噪声范围")
        print(f"jpeg_range: [{jpeg_min}, {jpeg_max}]  # JPEG压缩质量范围")
        
        print("\n# 第二次退化过程")
        blur2_min = max(0.2, blur_min * 0.5)
        blur2_max = max(1.5, blur_max * 0.5)
        noise2_min = max(1, int(noise_min * 0.8))
        noise2_max = max(25, int(noise_max * 0.8))
        
        print(f"blur_sigma2: [{blur2_min:.1f}, {blur2_max:.1f}]")
        print(f"noise_range2: [{noise2_min}, {noise2_max}]")
        print(f"poisson_scale_range2: [{noise2_min*0.05:.2f}, {noise2_max*0.1:.2f}]")
        print(f"jpeg_range2: [{jpeg_min}, {jpeg_max}]")
        
        print("\n# 其他建议")
        avg_scale = np.mean(scales)
        if avg_scale < 2.5:
            print(f"scale: 2  # 检测到的平均缩放为 {avg_scale:.1f}x，建议使用2x模型")
        else:
            print(f"scale: 4  # 检测到的平均缩放为 {avg_scale:.1f}x，建议使用4x模型")
        
        print(f"\ngt_size: 256  # 训练时的GT裁剪大小")
        
        # 红外图像特殊建议
        print("\n# 红外图像特殊建议:")
        print("# 1. 红外图像通常是单通道或伪彩色，建议:")
        print("#    - 如果是单通道灰度图: num_in_ch: 1, num_out_ch: 1")
        print("#    - 如果是伪彩色RGB: 保持 num_in_ch: 3, num_out_ch: 3")
        print("# 2. 红外图像可能有不同的噪声特性，可以调整:")
        print("#    gray_noise_prob: 0.6  # 增加灰度噪声概率")
        print("#    gaussian_noise_prob: 0.6  # 适当增加高斯噪声概率")
        print("# 3. 如果红外图像模糊较少，可以降低:")
        print("#    sinc_prob: 0.05  # 降低sinc滤波器概率")
        print("#    final_sinc_prob: 0.5  # 降低最终sinc滤波器概率")
        
        return {
            'blur_sigma': [blur_min, blur_max],
            'blur_sigma2': [blur2_min, blur2_max],
            'noise_range': [noise_min, noise_max],
            'noise_range2': [noise2_min, noise2_max],
            'jpeg_range': [jpeg_min, jpeg_max],
            'scale': int(round(avg_scale)),
        }
    
    def visualize_analysis(self, output_path='degradation_analysis.png'):
        """可视化分析结果"""
        if not self.results:
            print("没有分析结果可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 模糊sigma分布
        blur_sigmas = [r['blur_sigma'] for r in self.results]
        axes[0, 0].hist(blur_sigmas, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('模糊 Sigma')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('模糊程度分布')
        axes[0, 0].axvline(np.mean(blur_sigmas), color='r', linestyle='--', label=f'平均: {np.mean(blur_sigmas):.2f}')
        axes[0, 0].legend()
        
        # 2. 噪声水平分布
        noise_levels = [r['noise_level'] for r in self.results]
        axes[0, 1].hist(noise_levels, bins=20, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('噪声水平')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('噪声水平分布')
        axes[0, 1].axvline(np.mean(noise_levels), color='r', linestyle='--', label=f'平均: {np.mean(noise_levels):.2f}')
        axes[0, 1].legend()
        
        # 3. JPEG质量分布
        jpeg_qualities = [r['jpeg_quality'] for r in self.results]
        axes[1, 0].hist(jpeg_qualities, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_xlabel('JPEG 质量')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('JPEG压缩质量分布')
        axes[1, 0].axvline(np.mean(jpeg_qualities), color='r', linestyle='--', label=f'平均: {np.mean(jpeg_qualities):.0f}')
        axes[1, 0].legend()
        
        # 4. 高频能量比例对比
        hr_hf = [r['hr_high_freq_ratio'] for r in self.results]
        lr_hf = [r['lr_high_freq_ratio'] for r in self.results]
        x = np.arange(len(self.results))
        width = 0.35
        axes[1, 1].bar(x - width/2, hr_hf, width, label='HR高频比例', alpha=0.7)
        axes[1, 1].bar(x + width/2, lr_hf, width, label='LR高频比例', alpha=0.7)
        axes[1, 1].set_xlabel('图像对索引')
        axes[1, 1].set_ylabel('高频能量比例')
        axes[1, 1].set_title('HR vs LR 高频能量对比')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n分析图表已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='估计Real-ESRGAN训练的退化参数范围',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个图像对
  python estimate_degradation_params.py --hr hr_image.png --lr lr_image.png
  
  # 分析目录中的多个图像对
  python estimate_degradation_params.py --hr_dir ./hr_images --lr_dir ./lr_images
  
  # 分析并生成可视化图表
  python estimate_degradation_params.py --hr_dir ./hr_images --lr_dir ./lr_images --visualize
        """
    )
    
    parser.add_argument('--hr', type=str, help='单个HR图像路径')
    parser.add_argument('--lr', type=str, help='单个LR图像路径')
    parser.add_argument('--hr_dir', type=str, help='HR图像目录')
    parser.add_argument('--lr_dir', type=str, help='LR图像目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化分析图表')
    parser.add_argument('--output', type=str, default='degradation_analysis.png', help='可视化输出文件路径')
    
    args = parser.parse_args()
    
    estimator = DegradationEstimator()
    
    # 单个图像对分析
    if args.hr and args.lr:
        if not os.path.exists(args.hr):
            print(f"错误: HR图像不存在: {args.hr}")
            return
        if not os.path.exists(args.lr):
            print(f"错误: LR图像不存在: {args.lr}")
            return
        
        estimator.analyze_pair(args.hr, args.lr)
    
    # 目录批量分析
    elif args.hr_dir and args.lr_dir:
        if not os.path.isdir(args.hr_dir):
            print(f"错误: HR目录不存在: {args.hr_dir}")
            return
        if not os.path.isdir(args.lr_dir):
            print(f"错误: LR目录不存在: {args.lr_dir}")
            return
        
        # 查找匹配的图像对
        hr_files = sorted([f for f in os.listdir(args.hr_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        lr_files = sorted([f for f in os.listdir(args.lr_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        
        # 尝试按文件名匹配
        matched_pairs = []
        for hr_file in hr_files:
            hr_base = os.path.splitext(hr_file)[0]
            for lr_file in lr_files:
                lr_base = os.path.splitext(lr_file)[0]
                if hr_base == lr_base or hr_base in lr_base or lr_base in hr_base:
                    matched_pairs.append((
                        os.path.join(args.hr_dir, hr_file),
                        os.path.join(args.lr_dir, lr_file)
                    ))
                    break
        
        # 如果没有匹配到，尝试按顺序配对
        if not matched_pairs and len(hr_files) == len(lr_files):
            print("警告: 按文件名匹配失败，尝试按顺序配对...")
            matched_pairs = [(os.path.join(args.hr_dir, hr_files[i]), 
                            os.path.join(args.lr_dir, lr_files[i])) 
                           for i in range(len(hr_files))]
        
        if not matched_pairs:
            print(f"错误: 未找到匹配的图像对")
            print(f"HR图像数量: {len(hr_files)}, LR图像数量: {len(lr_files)}")
            return
        
        print(f"找到 {len(matched_pairs)} 对图像")
        
        for hr_path, lr_path in matched_pairs:
            try:
                estimator.analyze_pair(hr_path, lr_path)
            except Exception as e:
                print(f"错误: 分析图像对失败: {e}")
                continue
    
    else:
        parser.print_help()
        return
    
    # 汇总结果
    if estimator.results:
        estimator.summarize_results()
        
        if args.visualize:
            try:
                estimator.visualize_analysis(args.output)
            except Exception as e:
                print(f"警告: 生成可视化图表失败: {e}")
                print("继续输出文本结果...")


if __name__ == '__main__':
    main()

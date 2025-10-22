#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试退化参数估计工具
用于验证工具是否能正常运行
"""

import cv2
import numpy as np
import sys
import os

def create_test_images(output_dir='./test_images'):
    """创建测试用的HR/LR图像对"""
    os.makedirs(f"{output_dir}/hr", exist_ok=True)
    os.makedirs(f"{output_dir}/lr", exist_ok=True)
    
    print("创建测试图像...")
    
    # 创建一个简单的测试图像（灰度）
    hr_img = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
    # 添加一些结构
    hr_img[100:200, 100:200] = 255
    hr_img[300:400, 300:400] = 50
    
    # 创建LR图像（下采样 + 模糊 + 噪声）
    lr_img = cv2.resize(hr_img, (128, 128), interpolation=cv2.INTER_CUBIC)
    lr_img = cv2.GaussianBlur(lr_img, (5, 5), 1.0)
    noise = np.random.randn(*lr_img.shape) * 10
    lr_img = np.clip(lr_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    lr_img = cv2.resize(lr_img, (512, 512), interpolation=cv2.INTER_CUBIC)
    
    # 保存灰度测试图像
    cv2.imwrite(f"{output_dir}/hr/test_gray.png", hr_img)
    cv2.imwrite(f"{output_dir}/lr/test_gray.png", lr_img)
    print(f"  ✓ 创建灰度测试图像: {output_dir}/hr/test_gray.png")
    
    # 创建RGB测试图像
    hr_img_rgb = cv2.cvtColor(hr_img, cv2.COLOR_GRAY2BGR)
    lr_img_rgb = cv2.cvtColor(lr_img, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite(f"{output_dir}/hr/test_rgb.png", hr_img_rgb)
    cv2.imwrite(f"{output_dir}/lr/test_rgb.png", lr_img_rgb)
    print(f"  ✓ 创建RGB测试图像: {output_dir}/hr/test_rgb.png")
    
    print(f"\n测试图像已创建在: {output_dir}/")
    return output_dir

def test_image_loading():
    """测试图像加载"""
    print("\n" + "="*60)
    print("测试1: 图像加载")
    print("="*60)
    
    test_dir = create_test_images()
    
    try:
        from estimate_degradation_params import DegradationEstimator
        estimator = DegradationEstimator()
        
        # 测试灰度图像
        print("\n测试灰度图像加载...")
        hr_gray, lr_gray = estimator.load_image_pair(
            f"{test_dir}/hr/test_gray.png",
            f"{test_dir}/lr/test_gray.png"
        )
        print(f"  HR shape: {hr_gray.shape}, dtype: {hr_gray.dtype}, range: [{hr_gray.min():.3f}, {hr_gray.max():.3f}]")
        print(f"  LR shape: {lr_gray.shape}, dtype: {lr_gray.dtype}, range: [{lr_gray.min():.3f}, {lr_gray.max():.3f}]")
        print("  ✓ 灰度图像加载成功")
        
        # 测试RGB图像
        print("\n测试RGB图像加载...")
        hr_rgb, lr_rgb = estimator.load_image_pair(
            f"{test_dir}/hr/test_rgb.png",
            f"{test_dir}/lr/test_rgb.png"
        )
        print(f"  HR shape: {hr_rgb.shape}, dtype: {hr_rgb.dtype}, range: [{hr_rgb.min():.3f}, {hr_rgb.max():.3f}]")
        print(f"  LR shape: {lr_rgb.shape}, dtype: {lr_rgb.dtype}, range: [{lr_rgb.min():.3f}, {lr_rgb.max():.3f}]")
        print("  ✓ RGB图像加载成功")
        
        return True, estimator, test_dir
        
    except Exception as e:
        print(f"  ✗ 图像加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, test_dir

def test_blur_estimation(estimator, test_dir):
    """测试模糊估计"""
    print("\n" + "="*60)
    print("测试2: 模糊程度估计")
    print("="*60)
    
    try:
        hr_gray, lr_gray = estimator.load_image_pair(
            f"{test_dir}/hr/test_gray.png",
            f"{test_dir}/lr/test_gray.png"
        )
        
        print("\n估计HR图像模糊程度...")
        hr_lap, hr_hf = estimator.estimate_blur_sigma(hr_gray)
        print(f"  拉普拉斯方差: {hr_lap:.2f}")
        print(f"  高频比例: {hr_hf:.4f}")
        print("  ✓ HR图像分析成功")
        
        print("\n估计LR图像模糊程度...")
        lr_lap, lr_hf = estimator.estimate_blur_sigma(lr_gray)
        print(f"  拉普拉斯方差: {lr_lap:.2f}")
        print(f"  高频比例: {lr_hf:.4f}")
        print("  ✓ LR图像分析成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 模糊估计失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_noise_estimation(estimator, test_dir):
    """测试噪声估计"""
    print("\n" + "="*60)
    print("测试3: 噪声水平估计")
    print("="*60)
    
    try:
        hr_gray, lr_gray = estimator.load_image_pair(
            f"{test_dir}/hr/test_gray.png",
            f"{test_dir}/lr/test_gray.png"
        )
        
        print("\n估计HR图像噪声...")
        hr_noise = estimator.estimate_noise_level(hr_gray)
        print(f"  噪声水平: {hr_noise:.2f}")
        print("  ✓ HR噪声估计成功")
        
        print("\n估计LR图像噪声...")
        lr_noise = estimator.estimate_noise_level(lr_gray)
        print(f"  噪声水平: {lr_noise:.2f}")
        print("  ✓ LR噪声估计成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 噪声估计失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jpeg_estimation(estimator, test_dir):
    """测试JPEG质量估计"""
    print("\n" + "="*60)
    print("测试4: JPEG质量估计")
    print("="*60)
    
    try:
        _, lr_gray = estimator.load_image_pair(
            f"{test_dir}/hr/test_gray.png",
            f"{test_dir}/lr/test_gray.png"
        )
        
        print("\n估计LR图像JPEG质量...")
        quality, artifact = estimator.estimate_jpeg_quality(lr_gray)
        print(f"  块效应强度: {artifact:.2f}")
        print(f"  估计质量: {quality:.0f}")
        print("  ✓ JPEG质量估计成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ JPEG质量估计失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_analysis(estimator, test_dir):
    """测试完整分析流程"""
    print("\n" + "="*60)
    print("测试5: 完整分析流程")
    print("="*60)
    
    try:
        print("\n运行完整分析...")
        result = estimator.analyze_pair(
            f"{test_dir}/hr/test_gray.png",
            f"{test_dir}/lr/test_gray.png"
        )
        
        print("\n分析结果:")
        for key, value in result.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print("\n  ✓ 完整分析成功")
        return True
        
    except Exception as e:
        print(f"  ✗ 完整分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("退化参数估计工具 - 测试程序")
    print("="*60)
    
    results = []
    
    # 测试1: 图像加载
    success, estimator, test_dir = test_image_loading()
    results.append(("图像加载", success))
    
    if not success:
        print("\n基础测试失败，无法继续")
        return
    
    # 测试2: 模糊估计
    success = test_blur_estimation(estimator, test_dir)
    results.append(("模糊估计", success))
    
    # 测试3: 噪声估计
    success = test_noise_estimation(estimator, test_dir)
    results.append(("噪声估计", success))
    
    # 测试4: JPEG质量估计
    success = test_jpeg_estimation(estimator, test_dir)
    results.append(("JPEG质量估计", success))
    
    # 测试5: 完整分析
    success = test_full_analysis(estimator, test_dir)
    results.append(("完整分析", success))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！工具可以正常使用。")
        print(f"\n下一步:")
        print(f"  python3 estimate_degradation_params.py --hr_dir 你的HR目录 --lr_dir 你的LR目录 --visualize")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
    
    # 清理测试图像
    print(f"\n清理测试图像...")
    import shutil
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"  ✓ 已删除 {test_dir}")

if __name__ == '__main__':
    main()

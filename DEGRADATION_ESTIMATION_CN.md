# 退化参数估计工具使用说明

## 概述

`estimate_degradation_params.py` 是一个帮助你为红外图像训练确定合适退化参数的工具。它通过分析真实的LR/HR图像对，自动估计以下参数：

- **模糊程度** (blur_sigma)
- **噪声水平** (noise_range, poisson_scale_range)  
- **JPEG压缩质量** (jpeg_range)
- **缩放比例** (scale)

## 安装依赖

**重要**: 在使用工具前，必须先安装依赖！

```bash
# 一次性安装所有依赖
pip install opencv-python numpy scipy scikit-image matplotlib

# 或者分步安装
pip install opencv-python  # 图像处理
pip install numpy          # 数值计算
pip install scipy          # 科学计算
pip install scikit-image   # 图像分析
pip install matplotlib     # 可视化（用于--visualize参数）
```

**验证安装**：
```bash
python3 -c "import cv2, numpy, scipy, skimage, matplotlib; print('所有依赖安装成功！')"
```

## 使用方法

### 1. 准备你的图像对

确保你有对应的HR和LR图像：
- HR图像：高分辨率的红外图像
- LR图像：对应的低分辨率红外图像

图像可以是以下格式：`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

### 2. 分析单个图像对

```bash
python estimate_degradation_params.py --hr path/to/hr_image.png --lr path/to/lr_image.png
```

### 3. 批量分析多个图像对

```bash
python estimate_degradation_params.py --hr_dir ./hr_images --lr_dir ./lr_images
```

工具会自动尝试匹配文件名相同或相似的图像对。

### 4. 生成可视化分析图表

```bash
python estimate_degradation_params.py --hr_dir ./hr_images --lr_dir ./lr_images --visualize --output analysis.png
```

## 输出说明

工具会输出以下信息：

### 1. 每对图像的详细分析
- 缩放比例分析
- 模糊程度分析（拉普拉斯方差、高频能量）
- 噪声水平分析
- JPEG压缩质量分析

### 2. 汇总统计信息
所有图像对的参数统计（平均值、范围）

### 3. 推荐的配置参数
直接可用于配置文件的参数建议，例如：

```yaml
# 第一次退化过程
blur_sigma: [0.3, 2.1]  # 模糊核标准差
noise_range: [5, 25]  # 高斯噪声范围
poisson_scale_range: [0.25, 2.50]  # 泊松噪声范围
jpeg_range: [45, 85]  # JPEG压缩质量范围

# 第二次退化过程
blur_sigma2: [0.2, 1.1]
noise_range2: [4, 20]
poisson_scale_range2: [0.20, 2.00]
jpeg_range2: [45, 85]
```

## 分析原理

### 模糊估计
- 使用拉普拉斯方差检测图像清晰度
- 分析频域中的高频能量分布
- 比较HR和LR的高频能量损失来估计模糊程度

### 噪声估计
- 使用中值绝对偏差(MAD)方法
- 通过高通滤波提取噪声成分
- 估计0-255范围的噪声水平

### JPEG质量估计
- 检测8x8块边界的不连续性（块效应）
- 块效应越明显，JPEG质量越低
- 输出30-95范围的质量估计

## 使用建议

### 1. 图像对数量
- 建议至少使用3-5对图像进行分析
- 图像对越多，参数估计越准确
- 确保图像对具有代表性（包含不同场景、不同退化程度）

### 2. 图像质量
- HR图像应该是高质量的原始图像
- LR图像应该是真实的低质量图像（不是简单的下采样）
- 确保HR和LR是完全对应的同一场景

### 3. 红外图像特殊考虑

红外图像与自然图像有以下不同，需要特殊处理：

#### a) 通道数设置
```yaml
# 如果是灰度红外图像
network_g:
  num_in_ch: 1
  num_out_ch: 1

# 如果是伪彩色RGB红外图像
network_g:
  num_in_ch: 3
  num_out_ch: 3
```

#### b) 噪声特性
红外图像通常有不同的噪声模式：
```yaml
gray_noise_prob: 0.6  # 增加灰度噪声概率
gaussian_noise_prob: 0.6  # 适当增加高斯噪声概率
```

#### c) 模糊特性
如果红外图像模糊较少：
```yaml
sinc_prob: 0.05  # 降低sinc滤波器概率
second_blur_prob: 0.6  # 降低第二次模糊概率
final_sinc_prob: 0.5  # 降低最终sinc滤波器概率
```

#### d) 数据增强
红外图像可能不需要过多的颜色变换：
```yaml
use_hflip: True  # 水平翻转
use_rot: False  # 如果图像有方向性，关闭旋转
```

## 下一步：创建训练配置文件

根据工具的输出，你可以：

1. 复制一个现有的配置文件（如 `train_realesrgan_x4plus.yml`）
2. 根据估计的参数修改退化参数部分
3. 修改数据集路径和其他训练参数
4. 开始训练！

示例配置文件见：`options/train_realesrgan_x4plus_infrared.yml`（下一步创建）

## 故障排除

### 问题1: 图像对匹配失败
**解决方案**: 确保HR和LR文件名相同或相似，或者文件数量相同按顺序匹配

### 问题2: 估计的参数范围过大或过小
**解决方案**: 
- 检查图像对是否正确对应
- 增加分析的图像对数量
- 手动微调参数范围

### 问题3: 可视化失败
**解决方案**: 
- 确保安装了matplotlib: `pip install matplotlib`
- 检查输出路径是否有写入权限

## 进阶使用

### 自定义分析

你可以修改 `estimate_degradation_params.py` 中的分析方法：

- `estimate_blur_sigma()`: 调整模糊估计算法
- `estimate_noise_level()`: 调整噪声估计方法
- `estimate_jpeg_quality()`: 调整JPEG质量估计

### 集成到训练流程

可以在训练前自动运行此工具：

```bash
#!/bin/bash
# 1. 估计参数
python estimate_degradation_params.py --hr_dir ./data/hr --lr_dir ./data/lr --visualize

# 2. 根据输出手动调整配置文件
# 3. 开始训练
python realesrgan/train.py -opt options/train_realesrgan_infrared.yml
```

## 参考

- Real-ESRGAN论文: [Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://arxiv.org/abs/2107.10833)
- 退化模型详解: 查看 `realesrgan/data/realesrgan_dataset.py`

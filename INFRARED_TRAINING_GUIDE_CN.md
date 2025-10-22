# Real-ESRGAN 红外图像训练快速指南

本指南帮助你快速开始用红外图像微调Real-ESRGAN模型。

## 📋 目录

1. [准备工作](#准备工作)
2. [方案一：有LR/HR图像对（推荐）](#方案一有lrhr图像对推荐)
3. [方案二：只有HR图像](#方案二只有hr图像)
4. [开始训练](#开始训练)
5. [常见问题](#常见问题)

## 🔧 准备工作

### 1. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 额外依赖（用于参数估计）
pip install scikit-image matplotlib
```

### 2. 准备数据集

你的红外图像可以是：
- **灰度图像** (单通道，如热成像)
- **RGB伪彩色图像** (三通道，如彩色映射的红外图)
- **16位图像** (高动态范围红外图像)

支持格式：`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

## 🎯 方案一：有LR/HR图像对（推荐）

如果你有真实的低分辨率(LR)和高分辨率(HR)红外图像对，这是最佳方案！

### 步骤1: 准备图像对

组织你的图像：
```
my_infrared_data/
├── hr/              # 高分辨率图像
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── lr/              # 低分辨率图像（对应HR）
    ├── image_001.png
    ├── image_002.png
    └── ...
```

**重要**: 确保LR和HR图像文件名一致或相似！

### 步骤2: 运行自动化准备脚本

```bash
python prepare_infrared_training.py \
    --hr_dir ./my_infrared_data/hr \
    --lr_dir ./my_infrared_data/lr \
    --output_dir ./datasets/infrared \
    --channels 3  # 灰度图用1，RGB用3
```

这个脚本会自动：
- ✓ 分析LR/HR图像对，估计退化参数
- ✓ 生成可视化分析图表 (`degradation_analysis.png`)
- ✓ 创建目录结构
- ✓ 生成meta_info文件
- ✓ 创建配置文件模板

### 步骤3: 查看分析结果

1. **终端输出**: 查看推荐的参数范围
   ```
   推荐的配置文件参数:
   ====================================
   blur_sigma: [0.3, 2.1]
   noise_range: [5, 25]
   jpeg_range: [45, 85]
   ...
   ```

2. **可视化图表**: 打开 `degradation_analysis.png` 查看详细分析

### 步骤4: 调整配置文件

编辑 `datasets/infrared/train_config.yml`，找到标记为 `← 修改这里！` 的地方，填入步骤3得到的参数：

```yaml
# 示例：根据分析结果修改
blur_sigma: [0.3, 2.1]  # ← 从分析结果复制
noise_range: [5, 25]    # ← 从分析结果复制
jpeg_range: [45, 85]    # ← 从分析结果复制
```

### 步骤5: 跳到 [开始训练](#开始训练)

## 📝 方案二：只有HR图像

如果你只有高分辨率红外图像，需要手动估计退化参数。

### 步骤1: 准备HR图像

```
my_infrared_data/
└── train/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

### 步骤2: 运行准备脚本（跳过参数估计）

```bash
python prepare_infrared_training.py \
    --hr_dir ./my_infrared_data/train \
    --output_dir ./datasets/infrared \
    --channels 3 \
    --no-estimate  # 跳过参数估计
```

### 步骤3: 手动配置退化参数

编辑 `datasets/infrared/train_config.yml`，根据你的经验调整参数：

#### 3.1 对于高质量红外图像（轻度退化）：

```yaml
# 轻度模糊
blur_sigma: [0.2, 1.0]
blur_sigma2: [0.2, 0.8]

# 轻度噪声
noise_range: [1, 15]
noise_range2: [1, 10]

# 高JPEG质量
jpeg_range: [60, 95]
jpeg_range2: [60, 95]

# 较少模糊处理
sinc_prob: 0.05
second_blur_prob: 0.5
final_sinc_prob: 0.3
```

#### 3.2 对于低质量红外图像（严重退化）：

```yaml
# 严重模糊
blur_sigma: [0.5, 3.0]
blur_sigma2: [0.3, 1.5]

# 严重噪声
noise_range: [5, 30]
noise_range2: [3, 25]

# 低JPEG质量
jpeg_range: [30, 70]
jpeg_range2: [30, 70]

# 较多模糊处理
sinc_prob: 0.1
second_blur_prob: 0.8
final_sinc_prob: 0.8
```

#### 3.3 红外图像特殊设置（推荐）：

```yaml
# 红外图像通常有不同的噪声特性
gray_noise_prob: 0.6      # 增加灰度噪声
gaussian_noise_prob: 0.6  # 增加高斯噪声
```

## 🚀 开始训练

### 1. 下载预训练模型（推荐，可加速训练）

```bash
# 创建目录
mkdir -p experiments/pretrained_models

# 下载RealESRNet预训练模型
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRNet_x4plus.pth \
     -O experiments/pretrained_models/RealESRNet_x4plus.pth
```

**注意**: 如果你的红外图像是灰度图（`num_in_ch: 1`），预训练模型的通道数不匹配。
在配置文件中设置 `strict_load_g: false` 允许部分加载。

### 2. 开始训练

```bash
python realesrgan/train.py -opt datasets/infrared/train_config.yml
```

### 3. 监控训练

训练日志保存在：
- **Tensorboard**: `tb_logger/` 目录
  ```bash
  tensorboard --logdir tb_logger/
  ```
- **模型检查点**: `experiments/train_RealESRGAN_infrared_x4plus/models/`
- **可视化结果**: `experiments/train_RealESRGAN_infrared_x4plus/visualization/`

### 4. 使用训练好的模型

训练完成后，使用模型进行推理：

```bash
python inference_realesrgan.py \
    -n RealESRGAN_infrared \
    -i inputs/infrared_test \
    -o results \
    --model_path experiments/train_RealESRGAN_infrared_x4plus/models/net_g_latest.pth
```

## ❓ 常见问题

### Q1: 如何确定我的图像是灰度还是RGB？

```python
import cv2
img = cv2.imread('your_image.png', cv2.IMREAD_UNCHANGED)
print(f"形状: {img.shape}")
# 输出: (H, W) = 灰度, (H, W, 3) = RGB
```

在配置文件中设置：
- 灰度: `num_in_ch: 1, num_out_ch: 1`
- RGB: `num_in_ch: 3, num_out_ch: 3`

### Q2: 训练时GPU内存不足怎么办？

减小配置文件中的批次大小：
```yaml
batch_size_per_gpu: 4  # 从8降到4
gt_size: 128          # 从256降到128
```

### Q3: 如何知道参数设置是否合理？

观察训练过程中的验证结果：
1. 如果生成的LR图像比真实LR图像**更模糊/噪声更多** → 退化参数设置过强
2. 如果生成的LR图像比真实LR图像**更清晰/噪声更少** → 退化参数设置过弱
3. 理想情况：生成的LR图像与真实LR图像相似

### Q4: 训练需要多少数据？

建议：
- **最少**: 100-200张HR图像
- **推荐**: 500-1000张HR图像  
- **最佳**: 2000+张HR图像

数据越多，模型泛化能力越强。

### Q5: 训练需要多长时间？

取决于：
- **GPU性能**: RTX 3090约需1-2天（400k iterations）
- **数据量**: 数据越多，每个epoch越慢
- **批次大小**: batch_size越大，训练越快

可以根据需要调整 `total_iter` 参数。

### Q6: 如何微调已有模型？

设置配置文件中的 `resume_state`:
```yaml
path:
  resume_state: experiments/train_RealESRGAN_infrared_x4plus/training_states/400000.state
```

### Q7: 16位红外图像如何处理？

Real-ESRGAN自动支持16位图像！确保：
1. 使用支持16位的格式（`.png`, `.tif`）
2. 读取时保留原始位深度
3. 训练和推理会自动处理

### Q8: 只运行参数估计工具，不准备训练

```bash
python estimate_degradation_params.py \
    --hr_dir ./my_infrared_data/hr \
    --lr_dir ./my_infrared_data/lr \
    --visualize
```

## 📚 更多信息

- **详细的参数估计说明**: 查看 `DEGRADATION_ESTIMATION_CN.md`
- **配置文件详解**: 查看 `options/train_realesrgan_x4plus_infrared.yml` 中的注释
- **Real-ESRGAN论文**: https://arxiv.org/abs/2107.10833
- **BasicSR文档**: https://github.com/XPixelGroup/BasicSR

## 🔧 工具脚本总结

| 脚本 | 用途 |
|------|------|
| `estimate_degradation_params.py` | 分析LR/HR图像对，估计退化参数 |
| `prepare_infrared_training.py` | 自动化准备训练所需的所有文件 |
| `realesrgan/train.py` | 开始训练 |
| `inference_realesrgan.py` | 使用训练好的模型进行推理 |

## 💡 最佳实践

1. **先用少量数据测试**: 用100张图像训练10k iterations，确认流程正确
2. **监控验证结果**: 定期查看生成的图像质量
3. **保存检查点**: 每5k iterations保存一次（默认设置）
4. **使用Tensorboard**: 实时监控损失曲线
5. **迭代优化**: 根据结果调整退化参数，重新训练

## 🎉 完成！

按照上述步骤，你应该能够成功训练红外图像的Real-ESRGAN模型。

如果遇到问题，请检查：
1. 数据路径是否正确
2. 配置文件参数是否合理
3. GPU显存是否足够
4. 依赖是否完整安装

祝训练顺利！🚀

# 使用SPAN网络进行Real-ESRGAN训练和推理

本文档介绍如何在Real-ESRGAN项目中使用SPAN（Swift Parameter-free Attention Network）轻量级网络进行x2超分辨率。

## 📋 目录

- [简介](#简介)
- [SPAN网络特点](#span网络特点)
- [安装和准备](#安装和准备)
- [训练](#训练)
- [推理](#推理)
- [从x4权重适配到x2](#从x4权重适配到x2)
- [模型对比](#模型对比)

## 简介

SPAN是一个轻量级的超分辨率网络，相比RRDBNet具有以下优势：
- **更少的参数量**：SPAN约1.5M参数，RRDBNet约17M参数
- **更快的推理速度**：适合实时应用
- **良好的性能**：在保持轻量级的同时达到接近的视觉质量

## SPAN网络特点

### 架构说明

SPAN使用了以下关键技术：
1. **DSPAN (Depth-wise Separable Parameter-free Attention)**：无参数的注意力机制
2. **PCFN (Pure ConvFFN)**：纯卷积前馈网络
3. **快速傅里叶卷积**：提高计算效率

### 两个版本

项目提供了两个SPAN版本：

| 模型 | 特征通道数 | SPAN块数量 | 参数量 | 适用场景 |
|------|-----------|-----------|--------|---------|
| SPAN | 48 | 12 | ~1.5M | 轻量级应用，实时处理 |
| SPANPlus | 64 | 24 | ~4.5M | 更高质量需求 |

## 安装和准备

### 1. 确保依赖已安装

```bash
pip install -r requirements.txt
```

### 2. 准备训练数据

按照原有Real-ESRGAN的数据准备流程准备DF2K数据集：

```bash
# 下载DF2K数据集并放置在datasets/DF2K目录下
# 生成meta信息文件
python scripts/generate_meta_info.py
```

## 训练

### 使用SPAN进行训练

#### 标准SPAN（轻量级）

```bash
# 单GPU训练
python realesrgan/train.py -opt options/train_realesrgan_x2plus_span.yml

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
realesrgan/train.py -opt options/train_realesrgan_x2plus_span.yml --launcher pytorch
```

#### SPANPlus（增强版）

如果需要更高质量，可以修改配置文件使用SPANPlus：

```yaml
# 在 options/train_realesrgan_x2plus_span.yml 中修改
network_g:
  type: SPANPlus  # 改为 SPANPlus
  num_in_ch: 3
  num_out_ch: 3
  feature_channels: 64  # 从48增加到64
  upscale: 2
  num_blocks: 24  # 从12增加到24
  ffn_expansion_factor: 2.0
  bias: false
```

### 训练配置说明

主要配置在 `options/train_realesrgan_x2plus_span.yml`：

- **scale**: 2 (x2超分辨率)
- **network_g.type**: SPAN 或 SPANPlus
- **feature_channels**: 48（SPAN）或 64（SPANPlus）
- **num_blocks**: 12（SPAN）或 24（SPANPlus）
- **total_iter**: 400000（可根据需要调整）

## 推理

### 基本推理

使用训练好的SPAN模型进行推理：

```bash
# 使用SPAN模型
python inference_realesrgan_span.py \
    -n span_x2 \
    -i inputs \
    -o results \
    --model_path experiments/pretrained_models/span_x2.pth

# 使用SPANPlus模型
python inference_realesrgan_span.py \
    -n spanplus_x2 \
    -i inputs \
    -o results \
    --model_path experiments/pretrained_models/spanplus_x2.pth
```

### 推理参数说明

- `-n`: 模型名称 (span_x2 或 spanplus_x2)
- `-i`: 输入图片或文件夹路径
- `-o`: 输出文件夹路径
- `--model_path`: 模型权重文件路径
- `-s, --outscale`: 最终输出缩放倍数（默认2）
- `-t, --tile`: 分块处理大小（0表示不分块，大图推荐512）
- `--face_enhance`: 使用GFPGAN增强人脸
- `--fp32`: 使用FP32精度（默认FP16）

### 批量处理

```bash
# 处理文件夹中所有图片
python inference_realesrgan_span.py \
    -n span_x2 \
    -i inputs/my_images \
    -o results/my_results \
    --model_path experiments/pretrained_models/span_x2.pth

# 处理大图（使用tile避免显存不足）
python inference_realesrgan_span.py \
    -n span_x2 \
    -i inputs/large_image.jpg \
    -o results \
    --model_path experiments/pretrained_models/span_x2.pth \
    --tile 512 \
    --tile_pad 10
```

## 从x4权重适配到x2

如果你有SPAN的x4预训练权重，想要适配到x2，需要进行以下修改：

### 方法1：从头训练（推荐）

由于upscale因子改变，建议从头训练x2模型以获得最佳效果：

```bash
python realesrgan/train.py -opt options/train_realesrgan_x2plus_span.yml
```

### 方法2：迁移学习

可以加载x4权重的特征提取部分（排除上采样层）：

```python
import torch

# 加载x4模型权重
checkpoint_x4 = torch.load('span_x4.pth')

# 创建x2模型
from realesrgan.archs.span_arch import SPAN
model_x2 = SPAN(upscale=2)

# 只加载特征提取部分的权重
state_dict_x2 = model_x2.state_dict()
state_dict_x4 = checkpoint_x4['params_ema'] if 'params_ema' in checkpoint_x4 else checkpoint_x4

# 过滤掉upsampler的权重
filtered_dict = {k: v for k, v in state_dict_x4.items() 
                 if k in state_dict_x2 and 'upsampler' not in k}

# 更新模型权重
state_dict_x2.update(filtered_dict)
model_x2.load_state_dict(state_dict_x2, strict=False)

# 保存适配后的权重
torch.save({'params_ema': model_x2.state_dict()}, 'span_x2_from_x4.pth')
```

然后在训练配置中使用：

```yaml
path:
  pretrain_network_g: experiments/pretrained_models/span_x2_from_x4.pth
  param_key_g: params_ema
  strict_load_g: false
```

### 方法3：调整网络结构

如果x4权重使用不同的网络配置，需要确保x2模型使用相同的配置：

```yaml
network_g:
  type: SPAN
  num_in_ch: 3
  num_out_ch: 3
  feature_channels: 48  # 确保与x4模型一致
  upscale: 2  # 改为2
  num_blocks: 12  # 确保与x4模型一致
  ffn_expansion_factor: 2.0
  bias: false
```

## 模型对比

### 参数量和速度对比

| 模型 | 参数量 | 推理速度* | 显存占用* |
|------|--------|----------|-----------|
| RRDBNet (x2) | ~17M | 1x | 1x |
| SPAN (x2) | ~1.5M | ~3x | ~0.5x |
| SPANPlus (x2) | ~4.5M | ~2x | ~0.7x |

*相对于RRDBNet的相对值，实际速度取决于硬件配置

### 质量对比

建议在相同数据集上训练后，使用以下指标评估：

```bash
# 使用验证脚本
python scripts/validate.py \
    --model_path experiments/pretrained_models/span_x2.pth \
    --data_path datasets/val/Set5 \
    --scale 2
```

常用评估指标：
- **PSNR**：峰值信噪比，越高越好
- **SSIM**：结构相似性，越高越好
- **LPIPS**：感知相似度，越低越好

## 训练技巧

### 1. 学习率调整

SPAN网络较轻量，可以使用稍高的学习率：

```yaml
train:
  optim_g:
    lr: !!float 2e-4  # RRDBNet通常用1e-4
```

### 2. 批次大小

由于SPAN参数更少，可以使用更大的batch size：

```yaml
datasets:
  train:
    batch_size_per_gpu: 16  # RRDBNet通常用12
```

### 3. 渐进式训练

可以先训练轻量级SPAN，再finetune到SPANPlus：

```bash
# 第一阶段：训练SPAN
python realesrgan/train.py -opt options/train_realesrgan_x2plus_span.yml

# 第二阶段：基于SPAN训练SPANPlus
# 修改配置文件指定pretrain_network_g为第一阶段的权重
python realesrgan/train.py -opt options/train_realesrgan_x2plus_spanplus.yml
```

## 常见问题

### Q: 显存不足怎么办？

A: 尝试以下方法：
1. 减小batch size
2. 使用FP16训练（默认已启用）
3. 减小gt_size（如256改为128）
4. 推理时使用--tile参数

### Q: SPAN和RRDBNet性能差距大吗？

A: SPAN参数量小约10倍，速度快约3倍，视觉质量稍有差距但对大多数应用场景足够。如需接近RRDBNet的质量，可使用SPANPlus。

### Q: 如何从SPAN官方x4权重开始训练？

A: 参考"从x4权重适配到x2"章节，建议使用方法2进行迁移学习，或直接从头训练。

### Q: 训练多久能看到效果？

A: 通常10-20k iterations后能看到明显效果，完整训练建议400k iterations。

## 参考资料

- SPAN论文：[SPAN: Fast and Simple Learnable Spectral Transformation for Super-Resolution](https://arxiv.org/abs/2301.10580)
- SPAN GitHub：https://github.com/hongyuanyu/SPAN
- Real-ESRGAN：https://github.com/xinntao/Real-ESRGAN

## 许可证

本项目遵循Real-ESRGAN的BSD 3-Clause License。

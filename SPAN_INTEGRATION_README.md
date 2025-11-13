# Real-ESRGAN + SPAN 集成说明

## 📝 概述

本次修改将轻量级SPAN（Swift Parameter-free Attention Network）网络集成到Real-ESRGAN项目中，用于x2超分辨率任务。

## 🎯 主要修改

### 1. 新增文件

#### 核心架构文件
- **`realesrgan/archs/span_arch.py`** - SPAN网络架构实现
  - `SPAN`: 标准版 (48通道, 12块, ~1.5M参数)
  - `SPANPlus`: 增强版 (64通道, 24块, ~4.5M参数)
  - 包含DSPAN注意力机制和PCFN前馈网络

#### 配置文件
- **`options/train_realesrgan_x2plus_span.yml`** - SPAN训练配置
  - 已配置好所有训练参数
  - 支持x2超分辨率
  - 可切换SPAN/SPANPlus

#### 脚本文件
- **`inference_realesrgan_span.py`** - SPAN推理脚本
  - 支持批量处理
  - 支持tile分块处理大图
  - 支持face_enhance
  
- **`test_span.py`** - 网络测试脚本
  - 测试SPAN/SPANPlus网络
  - 检查注册状态
  - 模型对比

- **`convert_span_x4_to_x2.py`** - 权重转换工具
  - 将x4权重转换为x2
  - 保留特征提取层
  - 重新初始化上采样层

#### 文档
- **`docs/SPAN_USAGE.md`** - 详细使用文档
  - 完整的训练/推理指南
  - 权重转换说明
  - 常见问题解答

## 🚀 快速开始

### 步骤1：测试网络

```bash
# 安装依赖（如果还没有安装）
pip install -r requirements.txt

# 测试SPAN网络
python test_span.py
```

### 步骤2：准备数据

```bash
# 准备DF2K数据集
# 将数据放在 datasets/DF2K 目录下
python scripts/generate_meta_info.py
```

### 步骤3：开始训练

```bash
# 单GPU训练
python realesrgan/train.py -opt options/train_realesrgan_x2plus_span.yml

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
realesrgan/train.py -opt options/train_realesrgan_x2plus_span.yml --launcher pytorch
```

### 步骤4：推理

```bash
# 使用训练好的模型
python inference_realesrgan_span.py \
    -n span_x2 \
    -i inputs \
    -o results \
    --model_path experiments/pretrained_models/span_x2.pth
```

## 🔄 从x4权重转换到x2

如果你有SPAN的x4预训练权重：

```bash
# 转换权重
python convert_span_x4_to_x2.py \
    --x4_path /path/to/span_x4.pth \
    --output experiments/pretrained_models/span_x2_from_x4.pth \
    --model_type SPAN \
    --feature_channels 48 \
    --num_blocks 12

# 在训练配置中使用转换后的权重
# 修改 options/train_realesrgan_x2plus_span.yml:
# path:
#   pretrain_network_g: experiments/pretrained_models/span_x2_from_x4.pth
#   strict_load_g: false
```

## 📊 网络对比

| 网络 | 参数量 | 速度 | 适用场景 |
|------|--------|------|---------|
| RRDBNet | ~17M | 基准 | 高质量需求 |
| SPAN | ~1.5M | 3x快 | 轻量级/实时 |
| SPANPlus | ~4.5M | 2x快 | 平衡性能和质量 |

## 🔧 配置选项

### 切换到SPANPlus

在 `options/train_realesrgan_x2plus_span.yml` 中修改：

```yaml
network_g:
  type: SPANPlus  # 从 SPAN 改为 SPANPlus
  feature_channels: 64  # 从 48 改为 64
  num_blocks: 24  # 从 12 改为 24
```

### 调整学习率

```yaml
train:
  optim_g:
    lr: !!float 2e-4  # 可根据需要调整
```

### 调整批次大小

```yaml
datasets:
  train:
    batch_size_per_gpu: 16  # SPAN较轻量，可以用更大的batch
```

## 📁 项目结构变化

```
Real-ESRGAN/
├── realesrgan/
│   └── archs/
│       └── span_arch.py          [新增] SPAN网络架构
├── options/
│   └── train_realesrgan_x2plus_span.yml  [新增] SPAN训练配置
├── docs/
│   └── SPAN_USAGE.md             [新增] 使用文档
├── inference_realesrgan_span.py  [新增] SPAN推理脚本
├── test_span.py                  [新增] 测试脚本
├── convert_span_x4_to_x2.py      [新增] 权重转换工具
└── SPAN_INTEGRATION_README.md    [新增] 本文件
```

## ⚙️ 技术细节

### SPAN网络架构

SPAN网络包含以下组件：

1. **DSPAN (Depth-wise Separable Parameter-free Attention)**
   - 深度可分离卷积
   - 空间和通道注意力
   - 无参数注意力机制

2. **PCFN (Pure ConvFFN)**
   - 纯卷积前馈网络
   - GELU激活函数
   - 扩展因子2.0

3. **上采样模块**
   - x2: 单次PixelShuffle
   - x4: 两次PixelShuffle (2x2)

### 训练策略

- **初始学习率**: 2e-4（比RRDBNet的1e-4稍高）
- **学习率衰减**: MultiStepLR [200k, 300k, 350k, 375k]
- **总迭代次数**: 400k
- **损失函数**:
  - L1 Loss (像素级)
  - Perceptual Loss (感知损失)
  - GAN Loss (对抗损失)

## 🐛 常见问题

### Q: 显存不足

**A:** 尝试：
- 减小 `batch_size_per_gpu`
- 减小 `gt_size` (如 256 → 128)
- 使用 `tile` 参数分块处理

### Q: 如何使用预训练权重？

**A:** 修改配置文件：
```yaml
path:
  pretrain_network_g: path/to/span_x2.pth
  param_key_g: params_ema
  strict_load_g: true
```

### Q: SPAN和RRDBNet差距大吗？

**A:** 
- 参数量: SPAN ~1/10 RRDBNet
- 速度: SPAN ~3x RRDBNet
- 质量: 略有差距，但对多数场景足够
- 建议: 追求极致质量用RRDBNet，注重效率用SPAN

## 📚 参考资料

- **SPAN论文**: [SPAN: Fast and Simple Learnable Spectral Transformation for Super-Resolution](https://arxiv.org/abs/2301.10580)
- **SPAN GitHub**: https://github.com/hongyuanyu/SPAN
- **Real-ESRGAN**: https://github.com/xinntao/Real-ESRGAN

## 📝 下一步建议

1. ✅ **测试网络**: 运行 `python test_span.py`
2. ✅ **准备数据**: 下载并准备DF2K数据集
3. 🔄 **开始训练**: 使用提供的配置文件
4. 🔄 **监控训练**: 使用tensorboard查看训练进度
5. 🔄 **评估模型**: 在验证集上测试性能
6. 🔄 **微调优化**: 根据结果调整超参数

## 💡 提示

- SPAN适合快速迭代实验
- 建议先用SPAN训练，满意后可尝试SPANPlus
- 如果有x4权重，可以通过迁移学习加速训练
- 训练时建议启用tensorboard监控loss变化

## 📄 许可证

遵循Real-ESRGAN的BSD 3-Clause License。

---

**最后更新**: 2025-11-13
**版本**: 1.0

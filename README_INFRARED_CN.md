# 红外图像训练工具包 - 文件说明

为了帮助你用红外图像微调Real-ESRGAN模型，我创建了以下工具和文档。

## 📁 新增文件列表

### 🔧 核心工具脚本

| 文件 | 用途 | 使用场景 |
|------|------|----------|
| `estimate_degradation_params.py` | **退化参数估计工具** | 当你有LR/HR图像对，需要确定训练参数范围时 |
| `prepare_infrared_training.py` | **一键准备训练** | 自动化整个准备流程，从数据准备到配置生成 |
| `example_usage.sh` | **使用示例脚本** | 查看各种使用场景的命令示例 |

### 📋 配置文件

| 文件 | 用途 |
|------|------|
| `options/train_realesrgan_x4plus_infrared.yml` | **红外图像训练配置模板**，包含详细的参数说明和修改指导 |

### 📚 文档

| 文件 | 内容 |
|------|------|
| `INFRARED_TRAINING_GUIDE_CN.md` | **完整训练指南**：从准备到训练的完整流程 |
| `DEGRADATION_ESTIMATION_CN.md` | **参数估计工具详解**：分析原理和使用方法 |
| `README_INFRARED_CN.md` | **本文件**：所有工具的快速索引 |

## 🚀 快速开始

### 场景1：你有真实的LR和HR图像对（最佳方案）

```bash
# 一键准备训练
python prepare_infrared_training.py \
    --hr_dir ./my_infrared_images/hr \
    --lr_dir ./my_infrared_images/lr \
    --output_dir ./datasets/infrared \
    --channels 3  # 灰度图用1，RGB用3

# 这会自动：
# ✓ 分析图像对，估计最佳参数
# ✓ 生成可视化分析图表 (degradation_analysis.png)
# ✓ 创建配置文件 (datasets/infrared/train_config.yml)
# ✓ 准备所有必要的目录结构

# 根据输出调整配置文件后，开始训练
python realesrgan/train.py -opt datasets/infrared/train_config.yml
```

### 场景2：只有HR图像（需要手动设置参数）

```bash
# 准备训练目录
python prepare_infrared_training.py \
    --hr_dir ./my_infrared_images \
    --output_dir ./datasets/infrared \
    --channels 1 \
    --no-estimate

# 手动编辑配置文件中的退化参数
# vim datasets/infrared/train_config.yml

# 开始训练
python realesrgan/train.py -opt datasets/infrared/train_config.yml
```

### 场景3：只想分析参数，不立即训练

```bash
# 仅运行参数估计
python estimate_degradation_params.py \
    --hr_dir ./my_infrared_images/hr \
    --lr_dir ./my_infrared_images/lr \
    --visualize \
    --output my_analysis.png

# 查看终端输出和生成的图表
# 根据结果手动创建或修改配置文件
```

## 🎯 工作流程图

```
┌─────────────────────────────────────────────────────────────┐
│  你的数据                                                    │
├─────────────────────────────────────────────────────────────┤
│  有LR+HR图像对？                                             │
│                                                              │
│  是 ──→ prepare_infrared_training.py (with --lr_dir)        │
│         └─→ 自动分析参数                                     │
│         └─→ 生成配置文件                                     │
│         └─→ 微调参数                                         │
│         └─→ 开始训练                                         │
│                                                              │
│  否 ──→ prepare_infrared_training.py (--no-estimate)        │
│         └─→ 生成配置模板                                     │
│         └─→ 手动设置参数                                     │
│         └─→ 开始训练                                         │
└─────────────────────────────────────────────────────────────┘
```

## 📖 详细文档索引

### 如果你想...

- **快速开始训练** → 阅读 `INFRARED_TRAINING_GUIDE_CN.md`
- **理解参数估计原理** → 阅读 `DEGRADATION_ESTIMATION_CN.md`  
- **查看使用示例** → 运行 `./example_usage.sh`
- **了解配置参数** → 查看 `options/train_realesrgan_x4plus_infrared.yml` 中的注释

## 🔍 工具功能详解

### 1. estimate_degradation_params.py

**功能**: 通过分析真实的LR/HR图像对，估计合适的退化参数范围

**估计的参数**:
- ✓ 模糊程度 (`blur_sigma`)
- ✓ 噪声水平 (`noise_range`, `poisson_scale_range`)
- ✓ JPEG压缩质量 (`jpeg_range`)
- ✓ 缩放比例 (`scale`)

**输出**:
- 终端输出：每对图像的详细分析 + 汇总建议
- 可视化图表：参数分布直方图

**基本用法**:
```bash
# 分析单个图像对
python estimate_degradation_params.py --hr hr.png --lr lr.png

# 批量分析目录
python estimate_degradation_params.py \
    --hr_dir ./hr_images \
    --lr_dir ./lr_images \
    --visualize
```

### 2. prepare_infrared_training.py

**功能**: 自动化准备训练所需的所有文件和目录

**执行的任务**:
1. 检查依赖和数据
2. 创建目录结构
3. 生成meta_info.txt
4. 运行参数估计（如果有LR图像）
5. 生成配置文件
6. 显示下一步指南

**基本用法**:
```bash
# 完整流程（有LR+HR）
python prepare_infrared_training.py \
    --hr_dir ./hr \
    --lr_dir ./lr \
    --output_dir ./datasets/infrared

# 仅准备目录（只有HR）
python prepare_infrared_training.py \
    --hr_dir ./hr \
    --output_dir ./datasets/infrared \
    --no-estimate
```

### 3. train_realesrgan_x4plus_infrared.yml

**功能**: 专门为红外图像设计的训练配置模板

**特点**:
- 包含所有退化参数的详细说明
- 标记需要修改的地方 (`← 修改这里！`)
- 包含红外图像特殊建议
- 提供使用说明和检查清单

**关键配置项**:
```yaml
# 根据你的图像类型调整
num_in_ch: 3  # 灰度=1, RGB=3
num_out_ch: 3  # 与num_in_ch一致

# 根据参数估计结果修改
blur_sigma: [?, ?]
noise_range: [?, ?]
jpeg_range: [?, ?]

# 数据路径
dataroot_gt: datasets/infrared/train
meta_info: datasets/infrared/meta_info.txt
```

## ⚙️ 常用命令速查

```bash
# 查看使用示例
./example_usage.sh

# 分析参数
python estimate_degradation_params.py --hr_dir HR路径 --lr_dir LR路径 --visualize

# 准备训练（有LR）
python prepare_infrared_training.py --hr_dir HR路径 --lr_dir LR路径 --output_dir 输出路径

# 准备训练（无LR）
python prepare_infrared_training.py --hr_dir HR路径 --output_dir 输出路径 --no-estimate

# 开始训练
python realesrgan/train.py -opt 配置文件路径

# 监控训练
tensorboard --logdir tb_logger/

# 使用模型推理
python inference_realesrgan.py -n 模型名 -i 输入目录 -o 输出目录 --model_path 模型路径
```

## 💡 重要提示

### 关于通道数

**灰度红外图像** (如热成像):
```yaml
num_in_ch: 1
num_out_ch: 1
```

**RGB伪彩色红外图像**:
```yaml
num_in_ch: 3
num_out_ch: 3
```

**检查方法**:
```python
import cv2
img = cv2.imread('your_image.png', cv2.IMREAD_UNCHANGED)
print(img.shape)  # (H, W) = 灰度, (H, W, 3) = RGB
```

### 关于预训练模型

如果通道数不匹配（如你用灰度图但预训练模型是RGB），在配置中设置：
```yaml
strict_load_g: false  # 允许部分加载
```

### 关于数据量

- 最少：100-200张HR图像
- 推荐：500-1000张HR图像
- 最佳：2000+张HR图像

## 🐛 故障排除

| 问题 | 解决方案 |
|------|----------|
| GPU内存不足 | 减小 `batch_size_per_gpu` 和 `gt_size` |
| 图像对匹配失败 | 确保LR和HR文件名相同或相似 |
| 训练结果不理想 | 调整退化参数范围，查看验证结果 |
| 无法加载预训练模型 | 设置 `strict_load_g: false` |
| 依赖缺失 | `pip install -r requirements.txt` |

## 📞 获取帮助

- **查看详细教程**: `INFRARED_TRAINING_GUIDE_CN.md`
- **理解参数估计**: `DEGRADATION_ESTIMATION_CN.md`
- **查看配置说明**: `options/train_realesrgan_x4plus_infrared.yml`
- **Real-ESRGAN官方文档**: https://github.com/xinntao/Real-ESRGAN

## 🎉 总结

现在你有了完整的工具链来训练红外图像的Real-ESRGAN模型：

1. ✅ **参数估计工具** - 自动分析图像，确定最佳参数
2. ✅ **自动化准备脚本** - 一键完成所有准备工作
3. ✅ **优化的配置模板** - 专为红外图像设计
4. ✅ **详细的文档** - 覆盖各种使用场景
5. ✅ **示例和指南** - 快速上手

祝训练顺利！🚀

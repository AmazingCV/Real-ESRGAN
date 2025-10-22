# 快速修复指南

## 🔥 OpenCV错误已修复！

如果你之前遇到了这个错误：
```
error: (-213:The function/feature is not implemented) Unsupported combination of source format
```

**好消息**：这个问题已经修复！✅

---

## 📦 第一步：安装依赖

```bash
# 复制粘贴这一行即可
pip install opencv-python numpy scipy scikit-image matplotlib
```

**等待安装完成后**，验证一下：
```bash
python3 -c "import cv2; print('✓ OpenCV安装成功，版本:', cv2.__version__)"
```

---

## 🧪 第二步：测试工具

运行自动测试：
```bash
python3 test_estimate_tool.py
```

**期望看到**：
```
测试结果汇总
====================================
图像加载            : ✓ 通过
模糊估计            : ✓ 通过
噪声估计            : ✓ 通过
JPEG质量估计        : ✓ 通过
完整分析            : ✓ 通过

🎉 所有测试通过！工具可以正常使用。
```

---

## 🚀 第三步：分析你的图像

```bash
# 方法1: 一键准备训练（推荐）
python3 prepare_infrared_training.py \
    --hr_dir /path/to/your/hr_images \
    --lr_dir /path/to/your/lr_images \
    --output_dir ./datasets/infrared

# 方法2: 仅分析参数
python3 estimate_degradation_params.py \
    --hr_dir /path/to/your/hr_images \
    --lr_dir /path/to/your/lr_images \
    --visualize
```

---

## ❓ 常见问题速查

### Q1: "command not found: python"
```bash
# 使用 python3
python3 estimate_degradation_params.py --help
```

### Q2: "No module named 'cv2'"
```bash
pip install opencv-python
```

### Q3: "No module named 'skimage'"
```bash
pip install scikit-image
```

### Q4: 可视化失败
```bash
pip install matplotlib
```

### Q5: 图像对匹配失败
**确保**：LR和HR文件名相同或相似
```
hr/
  image_001.png
  image_002.png
lr/
  image_001.png  ← 名字要一样！
  image_002.png
```

### Q6: 仍然报OpenCV错误
```bash
# 重装OpenCV
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python
```

---

## 📝 修复内容

我修复了以下问题：

1. ✅ **图像格式兼容性** - 支持所有常见格式
2. ✅ **灰度图像处理** - 正确处理单通道图像  
3. ✅ **数据类型转换** - 统一标准化流程
4. ✅ **OpenCV滤波器** - 修复格式不匹配问题

---

## 💡 快速开始

**如果你有LR/HR图像对**：
```bash
# 一条命令搞定！
python3 prepare_infrared_training.py \
    --hr_dir 你的HR目录 \
    --lr_dir 你的LR目录 \
    --output_dir ./datasets/infrared
```

工具会自动：
- 分析你的图像
- 估计最佳参数
- 生成配置文件
- 显示推荐值

**然后**：
```bash
# 根据推荐调整配置文件
vim datasets/infrared/train_config.yml

# 开始训练
python3 realesrgan/train.py -opt datasets/infrared/train_config.yml
```

---

## 📚 详细文档

- **完整指南**: `INFRARED_TRAINING_GUIDE_CN.md`
- **修复详情**: `BUG_FIX_CN.md`
- **参数估计**: `DEGRADATION_ESTIMATION_CN.md`

---

## ✅ 检查清单

安装前检查：
- [ ] Python 3.7+ 已安装
- [ ] pip 可用

安装依赖：
- [ ] `pip install opencv-python numpy scipy scikit-image matplotlib`
- [ ] 验证：`python3 -c "import cv2; print('OK')"`

准备数据：
- [ ] HR图像已准备
- [ ] LR图像已准备（如果有）
- [ ] 文件名匹配正确

运行工具：
- [ ] 测试脚本通过：`python3 test_estimate_tool.py`
- [ ] 分析成功：`python3 estimate_degradation_params.py ...`

---

## 🎉 完成！

现在你可以：
1. ✅ 无错误运行参数估计工具
2. ✅ 自动获取推荐参数范围
3. ✅ 开始训练红外图像模型

不再需要"大海捞针"式地测试参数！🚀

---

**需要帮助？** 查看 `BUG_FIX_CN.md` 获取详细的故障排除指南。

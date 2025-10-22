#!/bin/bash
# Real-ESRGAN 红外图像训练 - 使用示例

echo "========================================"
echo "Real-ESRGAN 红外图像训练示例"
echo "========================================"

# ============================================
# 示例1: 完整流程（有LR/HR图像对）
# ============================================
echo -e "\n示例1: 完整流程（有LR和HR图像）"
echo "------------------------------------"

# 假设你的数据组织如下：
# ./my_data/
# ├── hr/  (高分辨率红外图像)
# └── lr/  (低分辨率红外图像)

# 一键准备训练
echo "命令："
echo "python prepare_infrared_training.py \\"
echo "    --hr_dir ./my_data/hr \\"
echo "    --lr_dir ./my_data/lr \\"
echo "    --output_dir ./datasets/infrared \\"
echo "    --channels 3"  # 如果是灰度图，改为 --channels 1
echo ""
echo "这个命令会："
echo "  1. 分析你的LR/HR图像对"
echo "  2. 估计最佳的退化参数范围"
echo "  3. 生成可视化分析图表"
echo "  4. 创建训练配置文件"
echo "  5. 准备所有必要的目录结构"

# ============================================
# 示例2: 只有HR图像
# ============================================
echo -e "\n示例2: 只有HR图像（无LR）"
echo "------------------------------------"

echo "命令："
echo "python prepare_infrared_training.py \\"
echo "    --hr_dir ./my_data/train \\"
echo "    --output_dir ./datasets/infrared \\"
echo "    --channels 1 \\"  # 灰度红外图像
echo "    --no-estimate"
echo ""
echo "然后手动编辑配置文件中的退化参数"

# ============================================
# 示例3: 仅运行参数估计
# ============================================
echo -e "\n示例3: 仅分析图像，估计参数"
echo "------------------------------------"

echo "命令："
echo "python estimate_degradation_params.py \\"
echo "    --hr_dir ./my_data/hr \\"
echo "    --lr_dir ./my_data/lr \\"
echo "    --visualize \\"
echo "    --output my_analysis.png"
echo ""
echo "这会生成详细的参数分析报告和可视化图表"

# ============================================
# 示例4: 开始训练
# ============================================
echo -e "\n示例4: 开始训练"
echo "------------------------------------"

echo "1. 首先下载预训练模型（可选，但推荐）："
echo "   mkdir -p experiments/pretrained_models"
echo "   wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRNet_x4plus.pth \\"
echo "        -O experiments/pretrained_models/RealESRNet_x4plus.pth"
echo ""
echo "2. 开始训练："
echo "   python realesrgan/train.py -opt datasets/infrared/train_config.yml"
echo ""
echo "3. 监控训练（另开一个终端）："
echo "   tensorboard --logdir tb_logger/"

# ============================================
# 示例5: 使用训练好的模型
# ============================================
echo -e "\n示例5: 使用训练好的模型进行推理"
echo "------------------------------------"

echo "命令："
echo "python inference_realesrgan.py \\"
echo "    -n RealESRGAN_infrared \\"
echo "    -i inputs/test_infrared \\"
echo "    -o results \\"
echo "    --model_path experiments/train_RealESRGAN_infrared_x4plus/models/net_g_400000.pth \\"
echo "    --fp32  # 如果是灰度图或遇到问题，使用FP32"

# ============================================
# 快速开始流程
# ============================================
echo -e "\n========================================"
echo "快速开始流程总结"
echo "========================================"

echo -e "\n如果你有LR和HR图像对（推荐）："
echo "  1. python prepare_infrared_training.py --hr_dir HR路径 --lr_dir LR路径 --output_dir ./datasets/infrared"
echo "  2. 查看终端输出和 degradation_analysis.png"
echo "  3. 根据建议调整 datasets/infrared/train_config.yml"
echo "  4. python realesrgan/train.py -opt datasets/infrared/train_config.yml"

echo -e "\n如果只有HR图像："
echo "  1. python prepare_infrared_training.py --hr_dir HR路径 --output_dir ./datasets/infrared --no-estimate"
echo "  2. 手动编辑 datasets/infrared/train_config.yml 中的退化参数"
echo "  3. python realesrgan/train.py -opt datasets/infrared/train_config.yml"

echo -e "\n========================================"
echo "更多详细信息，请查看："
echo "  - INFRARED_TRAINING_GUIDE_CN.md （完整训练指南）"
echo "  - DEGRADATION_ESTIMATION_CN.md （参数估计详解）"
echo "========================================"

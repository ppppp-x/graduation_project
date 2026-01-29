---
name: run_experiment
description: 执行一次完整的模型训练与评估循环。
---

# 实验执行工作流

你正在进行一项严谨的科学实验。请严格遵守以下步骤。

## 第一步：配置检查
- 询问用户：“你想训练哪个模型变体？（例如：DnCNN_Baseline, DnCNN_CBAM）”
- 询问用户：“这次实验的名称是什么？（例如：exp01_baseline）”
- 确认 `data/train` 目录存在且包含图像文件。

## 第二步：代码准备
- 验证 `src/models/` 下是否存在指定的模型类。
- 如果模型不存在，询问用户是否需要现在生成它。
- 创建一个配置字典 (JSON/Dict) 来记录超参数（学习率、Batch Size、Patch Size）。

## 第三步：执行训练
- 使用指定的模型和实验名称运行训练脚本 `python src/train.py`。
- **关键监控**：监控前 100 次迭代 (iteration) 的 Loss 值。如果 Loss 为 `NaN` 或完全不下降，**立即停止**并报告错误。

## 第四步：自动评估
- 训练结束后，自动加载效果最好的权重文件 (`best.pth`)。
- 在验证集上运行 `python src/eval.py`。
- 计算并输出 **PSNR** (峰值信噪比) 和 **SSIM** (结构相似性)。

## 第五步：生成报告
- 创建或追加内容到 `experiments/experiment_log.md` 文件中。
- 记录以下信息：
    - 实验名称
    - 模型架构
    - 最高 PSNR / 最高 SSIM
    - 训练耗时
- 生成一张视觉对比图并保存为 `experiments/<实验名称>/visual_check.png`。
- **最终动作**：向用户展示 PSNR 结果，并询问：“是否需要对注意力模块进行改进？”
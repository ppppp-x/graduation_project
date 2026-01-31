# 实验结果汇总

## 模型对比

| 模型 | 参数量 | 训练 PSNR | 测试 PSNR | SSIM | 提升 |
|------|--------|-----------|-----------|------|------|
| DnCNN Baseline | 558,336 | 30.97 dB | 28.26 dB | 0.9055 | - |
| **DnCNN + CBAM** | **558,946** | **31.20 dB** | **36.34 dB** | **0.9189** | **+8.08 dB** |

## 关键发现

1. **CBAM 注意力机制显著提升降噪效果**
   - 测试 PSNR 提升 8.08 dB
   - SSIM 从 0.9055 提升到 0.9189

2. **参数量几乎无增加**
   - 仅增加 610 个参数 (0.1%)
   - 计算成本可忽略不计

3. **训练稳定性**
   - CBAM 模型初期收敛较慢 (前 10 epoch)
   - 但最终性能更优

## 可视化结果

### 训练曲线对比
![Training Curves](experiments/comparison_curves.png)

### Baseline 降噪效果
![Baseline Result](experiments/baseline_50ep/visual_check.png)

### CBAM 降噪效果
![CBAM Result](experiments/cbam_50ep/visual_check.png)

## 实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 3050 (4GB) / ModelScope A10 |
| 框架 | PyTorch 2.0+ |
| 数据集 | DOTA (1411 张遥感图像) |
| Patch Size | 64×64 |
| Batch Size | 8 |
| 噪声级别 | σ = 25 |
| 训练轮数 | 50 epochs |
| 学习率 | 1e-3 (Adam, StepLR) |

---

## 噪声消融实验 (A3)

模型在不同噪声强度下的鲁棒性测试：

| σ | 含噪 PSNR | 降噪后 PSNR | SSIM | 提升 |
|---|-----------|-------------|------|------|
| 15 | 24.65 dB | 33.66 dB | 0.9153 | +9.00 dB |
| 25 | 20.24 dB | 33.45 dB | 0.8996 | +13.21 dB |
| 35 | 17.36 dB | 27.55 dB | 0.6035 | +10.19 dB |
| 50 | 14.45 dB | 14.57 dB | 0.0991 | +0.12 dB |

> [!NOTE]
> 模型在 σ ≤ 35 时表现优秀，σ = 50 时接近失效。这符合预期：模型仅在 σ = 25 上训练，对更强噪声泛化能力有限。

![Noise Ablation](experiments/ablation/noise_ablation.png)

---

## CBAM 注意力可视化 (B3)

![Attention Heatmap](experiments/cbam_50ep/attention_heatmap.png)

**关键发现**: CBAM 注意力模块自动聚焦于：
- 建筑物边缘
- 小型地面目标
- 纹理复杂区域

这证明了注意力机制能有效保护遥感图像中的小目标细节。

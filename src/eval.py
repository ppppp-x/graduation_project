"""
DnCNN 评估脚本
生成对比图和计算 PSNR/SSIM
"""
import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(__file__))

from model import DnCNN


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """计算 PSNR"""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """计算 SSIM (结构相似性)"""
    # 转换为 [H, W, C] 格式
    if pred.ndim == 3 and pred.shape[0] == 3:
        pred = pred.transpose(1, 2, 0)
        target = target.transpose(1, 2, 0)
    
    return ssim(pred, target, channel_axis=2, data_range=1.0)


def add_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma / 255.0, img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0, 1)
    return noisy


def denoise_image(
    model: nn.Module,
    noisy_img: np.ndarray,
    device: torch.device,
    patch_size: int = 64,
) -> np.ndarray:
    """
    对整张图像进行降噪 (分块处理，避免爆显存)
    """
    model.eval()
    h, w, c = noisy_img.shape
    
    # 填充到 patch_size 的整数倍
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padded = np.pad(noisy_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    result = np.zeros_like(padded)
    
    with torch.no_grad():
        for i in range(0, padded.shape[0], patch_size):
            for j in range(0, padded.shape[1], patch_size):
                patch = padded[i:i+patch_size, j:j+patch_size]
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                denoised = model(patch_tensor)
                result[i:i+patch_size, j:j+patch_size] = denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    # 裁剪回原始尺寸
    result = result[:h, :w]
    return np.clip(result, 0, 1)


def save_comparison(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    save_path: str,
    metrics: dict,
):
    """
    保存对比图 (项目规则要求: [含噪输入 | 降噪输出 | 清晰真值])
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 含噪输入
    axes[0].imshow(noisy)
    axes[0].set_title(f"Noisy Input\nPSNR: {metrics['noisy_psnr']:.2f} dB")
    axes[0].axis('off')
    
    # 降噪输出
    axes[1].imshow(denoised)
    axes[1].set_title(f"Denoised Output\nPSNR: {metrics['denoised_psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}")
    axes[1].axis('off')
    
    # 清晰真值
    axes[2].imshow(clean)
    axes[2].set_title("Clean Ground Truth")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='DnCNN Evaluation')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--output', type=str, default='eval_result.png', help='Output path')
    parser.add_argument('--noise_sigma', type=float, default=25.0)
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    model = DnCNN(use_attention=args.use_attention).to(device)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded weights from: {args.weights}")
    print(f"Checkpoint PSNR: {checkpoint.get('psnr', 'N/A')}")
    
    # 加载图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Cannot read image: {args.image}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    clean = img.astype(np.float32) / 255.0
    
    # 添加噪声
    noisy = add_noise(clean, args.noise_sigma)
    
    # 降噪
    print("Denoising...")
    denoised = denoise_image(model, noisy, device, args.patch_size)
    
    # 计算指标
    metrics = {
        'noisy_psnr': psnr(noisy, clean),
        'denoised_psnr': psnr(denoised, clean),
        'ssim': compute_ssim(denoised, clean),
    }
    
    print(f"\n{'='*40}")
    print(f"Noisy PSNR:    {metrics['noisy_psnr']:.2f} dB")
    print(f"Denoised PSNR: {metrics['denoised_psnr']:.2f} dB")
    print(f"SSIM:          {metrics['ssim']:.4f}")
    print(f"Improvement:   +{metrics['denoised_psnr'] - metrics['noisy_psnr']:.2f} dB")
    print(f"{'='*40}")
    
    # 保存对比图
    save_comparison(clean, noisy, denoised, args.output, metrics)


if __name__ == '__main__':
    main()

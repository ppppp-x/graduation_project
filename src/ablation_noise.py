"""
噪声消融实验 - 不同噪声强度评估
测试模型在 σ=15, 25, 35, 50 下的鲁棒性
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

sys.path.insert(0, os.path.dirname(__file__))
from model import DnCNN


def add_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """添加高斯噪声"""
    noise = np.random.normal(0, sigma / 255.0, image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0, 1)
    return noisy


def denoise_image(model: torch.nn.Module, noisy_img: np.ndarray, 
                  device: torch.device, patch_size: int = 64) -> np.ndarray:
    """分块降噪"""
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
                patch_tensor = torch.from_numpy(
                    patch.transpose(2, 0, 1)
                ).unsqueeze(0).float().to(device)
                
                denoised = model(patch_tensor)
                result[i:i+patch_size, j:j+patch_size] = (
                    denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                )
    
    return np.clip(result[:h, :w], 0, 1)


def run_ablation(model_path: str, image_path: str, use_attention: bool,
                 output_dir: str, noise_levels: list = [15, 25, 35, 50]):
    """运行消融实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    model = DnCNN(use_attention=use_attention).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded: {model_path}")
    
    # 加载图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    clean = img.astype(np.float32) / 255.0
    
    # 裁剪到合适大小
    h, w = clean.shape[:2]
    clean = clean[:min(h, 512), :min(w, 512)]
    
    results = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    for sigma in noise_levels:
        print(f"\nTesting σ = {sigma}...")
        
        # 设置固定随机种子
        np.random.seed(42)
        noisy = add_noise(clean, sigma)
        
        # 降噪
        denoised = denoise_image(model, noisy, device)
        
        # 计算指标
        noisy_psnr = calc_psnr(clean, noisy, data_range=1.0)
        denoised_psnr = calc_psnr(clean, denoised, data_range=1.0)
        ssim = calc_ssim(clean, denoised, data_range=1.0, channel_axis=2)
        improvement = denoised_psnr - noisy_psnr
        
        results.append({
            'sigma': sigma,
            'noisy_psnr': noisy_psnr,
            'denoised_psnr': denoised_psnr,
            'ssim': ssim,
            'improvement': improvement
        })
        
        print(f"  Noisy PSNR: {noisy_psnr:.2f} dB")
        print(f"  Denoised PSNR: {denoised_psnr:.2f} dB")
        print(f"  Improvement: {improvement:+.2f} dB")
    
    # 生成表格
    print("\n" + "=" * 60)
    print(f"{'σ':>6} | {'Noisy PSNR':>12} | {'Denoised PSNR':>14} | {'SSIM':>8} | {'Δ PSNR':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['sigma']:>6} | {r['noisy_psnr']:>12.2f} | {r['denoised_psnr']:>14.2f} | {r['ssim']:>8.4f} | {r['improvement']:>+8.2f}")
    print("=" * 60)
    
    # 生成消融图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sigmas = [r['sigma'] for r in results]
    noisy_psnrs = [r['noisy_psnr'] for r in results]
    denoised_psnrs = [r['denoised_psnr'] for r in results]
    ssims = [r['ssim'] for r in results]
    
    # PSNR 曲线
    axes[0].plot(sigmas, noisy_psnrs, 'r--o', label='Noisy Input', linewidth=2)
    axes[0].plot(sigmas, denoised_psnrs, 'b-o', label='Denoised', linewidth=2)
    axes[0].set_xlabel('Noise Level (σ)', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR vs Noise Level', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # SSIM 曲线
    axes[1].plot(sigmas, ssims, 'g-o', linewidth=2)
    axes[1].set_xlabel('Noise Level (σ)', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM vs Noise Level', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'noise_ablation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nAblation plot saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--image', type=str, default='data/train/P0000.jpg')
    parser.add_argument('--output_dir', type=str, default='experiments/ablation')
    parser.add_argument('--use_attention', action='store_true')
    args = parser.parse_args()
    
    run_ablation(
        model_path=args.weights,
        image_path=args.image,
        use_attention=args.use_attention,
        output_dir=args.output_dir
    )

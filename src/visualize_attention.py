"""
CBAM 注意力热力图可视化
展示模型对不同区域的关注程度
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from model import DnCNN, CBAM


class DnCNNWithHooks(nn.Module):
    """带 hook 的 DnCNN，用于提取中间层特征"""
    
    def __init__(self, base_model: DnCNN):
        super().__init__()
        self.model = base_model
        self.attention_weights = None
        self.feature_maps = None
        
        # 注册 hook
        for name, module in self.model.named_modules():
            if isinstance(module, CBAM):
                module.register_forward_hook(self._attention_hook)
    
    def _attention_hook(self, module, input, output):
        """捕获注意力模块的输出"""
        # CBAM 输出是加权后的特征图
        # 计算通道维度的平均作为空间注意力图
        self.feature_maps = output.detach()
        # 空间注意力: 对通道取平均
        self.attention_weights = torch.mean(output, dim=1, keepdim=True).detach()
    
    def forward(self, x):
        return self.model(x)


def visualize_attention(model_path: str, image_path: str, output_path: str,
                        noise_sigma: float = 25.0):
    """生成注意力热力图"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 加载模型
    base_model = DnCNN(use_attention=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    
    model = DnCNNWithHooks(base_model)
    print(f"Loaded: {model_path}")
    
    # 加载图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 裁剪到合理大小
    h, w = img.shape[:2]
    crop_size = 256
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    img = img[start_h:start_h+crop_size, start_w:start_w+crop_size]
    
    clean = img.astype(np.float32) / 255.0
    
    # 添加噪声
    np.random.seed(42)
    noise = np.random.normal(0, noise_sigma / 255.0, clean.shape).astype(np.float32)
    noisy = np.clip(clean + noise, 0, 1)
    
    # 前向传播获取注意力
    noisy_tensor = torch.from_numpy(
        noisy.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)
    
    denoised = denoised_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    denoised = np.clip(denoised, 0, 1)
    
    # 获取注意力图
    if model.attention_weights is not None:
        attention = model.attention_weights.squeeze().cpu().numpy()
        # 归一化到 [0, 1]
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    else:
        print("[WARN] No attention weights captured!")
        attention = np.ones((crop_size, crop_size))
    
    # 调整注意力图大小以匹配原图
    if attention.shape != (crop_size, crop_size):
        attention = cv2.resize(attention, (crop_size, crop_size))
    
    # 生成可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行: 原图系列
    axes[0, 0].imshow(clean)
    axes[0, 0].set_title('Clean Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy)
    axes[0, 1].set_title(f'Noisy Input (σ={noise_sigma})', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(denoised)
    axes[0, 2].set_title('Denoised Output', fontsize=12)
    axes[0, 2].axis('off')
    
    # 第二行: 注意力系列
    axes[1, 0].imshow(attention, cmap='hot')
    axes[1, 0].set_title('CBAM Attention Map', fontsize=12)
    axes[1, 0].axis('off')
    
    # 注意力叠加在原图上
    overlay = noisy.copy()
    heatmap = plt.cm.jet(attention)[:, :, :3]
    overlay = 0.6 * overlay + 0.4 * heatmap
    axes[1, 1].imshow(np.clip(overlay, 0, 1))
    axes[1, 1].set_title('Attention Overlay', fontsize=12)
    axes[1, 1].axis('off')
    
    # 高注意力区域放大
    threshold = np.percentile(attention, 90)
    high_attention = attention > threshold
    highlighted = noisy.copy()
    highlighted[~high_attention] *= 0.3  # 降低非关注区域亮度
    axes[1, 2].imshow(highlighted)
    axes[1, 2].set_title('High Attention Regions (Top 10%)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle('CBAM Attention Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Attention heatmap saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--image', type=str, default='data/train/P0000.jpg')
    parser.add_argument('--output', type=str, default='experiments/cbam_50ep/attention_heatmap.png')
    parser.add_argument('--noise_sigma', type=float, default=25.0)
    args = parser.parse_args()
    
    visualize_attention(
        model_path=args.weights,
        image_path=args.image,
        output_path=args.output,
        noise_sigma=args.noise_sigma
    )

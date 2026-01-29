"""
DnCNN 遥感降噪数据集
针对 4GB 显存优化：64x64 patch, 方差过滤
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A


class DenoisingDataset(Dataset):
    """
    遥感影像降噪数据集
    
    特性:
    - 64x64 随机裁剪
    - 动态添加高斯噪声 (σ=25)
    - 方差过滤：跳过低对比度 patch
    """
    
    def __init__(
        self,
        image_dir: str,
        patch_size: int = 64,
        noise_sigma: float = 25.0,
        min_variance: float = 0.01,
        max_retries: int = 10,
    ):
        """
        Args:
            image_dir: 图像目录路径
            patch_size: 裁剪尺寸
            noise_sigma: 高斯噪声标准差 (0-255 scale)
            min_variance: patch 最小方差阈值 (归一化后)
            max_retries: 采样失败最大重试次数
        """
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.noise_sigma = noise_sigma / 255.0  # 归一化到 [0,1]
        self.min_variance = min_variance
        self.max_retries = max_retries
        
        # 扫描所有图片
        self.image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not self.image_files:
            raise ValueError(f"目录为空: {image_dir}")
        
        # albumentations 裁剪 (随机位置)
        self.transform = A.Compose([
            A.RandomCrop(width=patch_size, height=patch_size),
        ])
    
    def __len__(self) -> int:
        # 每张图采样多个 patch，epoch 长度 = 图片数 * 10
        return len(self.image_files) * 10
    
    def __getitem__(self, idx: int):
        for _ in range(self.max_retries):
            # 随机选一张图
            img_path = random.choice(self.image_files)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            h, w = img.shape[:2]
            if h < self.patch_size or w < self.patch_size:
                continue
            
            # 裁剪
            augmented = self.transform(image=img)
            patch = augmented['image']
            
            # BGR -> RGB, 归一化到 [0, 1]
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch = patch.astype(np.float32) / 255.0
            
            # 方差检查
            if patch.var() < self.min_variance:
                continue  # 低对比度，跳过
            
            # 添加噪声
            noise = np.random.normal(0, self.noise_sigma, patch.shape).astype(np.float32)
            noisy_patch = np.clip(patch + noise, 0, 1)
            
            # HWC -> CHW (PyTorch 格式)
            clean = torch.from_numpy(patch.transpose(2, 0, 1))
            noisy = torch.from_numpy(noisy_patch.transpose(2, 0, 1))
            
            return noisy, clean
        
        # 重试耗尽，返回随机噪声 (fallback，不应该发生)
        fallback = torch.rand(3, self.patch_size, self.patch_size)
        return fallback, fallback


def get_dataloader(
    image_dir: str,
    batch_size: int = 8,
    patch_size: int = 64,
    noise_sigma: float = 25.0,
    shuffle: bool = True,
):
    """
    获取 DataLoader
    
    注意: num_workers=0 (Windows 兼容性)
    """
    dataset = DenoisingDataset(
        image_dir=image_dir,
        patch_size=patch_size,
        noise_sigma=noise_sigma,
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,       # Windows 上多进程有坑
        pin_memory=True,     # 加速 CPU->GPU 传输
        drop_last=True,      # 避免最后一个不完整 batch
    )


if __name__ == "__main__":
    # 简单测试
    import torch
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    loader = get_dataloader("data/train", batch_size=8)
    noisy, clean = next(iter(loader))
    
    print(f"Noisy shape: {noisy.shape}")  # [8, 3, 64, 64]
    print(f"Clean shape: {clean.shape}")  # [8, 3, 64, 64]
    print(f"Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print("[PASS] Dataset works.")

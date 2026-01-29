"""
DnCNN 训练脚本
针对 4GB 显存优化
"""
import os
import sys
import time
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(__file__))

from dataset import DenoisingDataset, get_dataloader
from model import DnCNN, count_parameters


def set_seed(seed: int = 42):
    """设置随机种子保证可复现性 (项目规则要求)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_val ** 2 / mse).item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for noisy, clean in pbar:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        denoised = model(noisy)
        loss = criterion(denoised, clean)
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_psnr += psnr(denoised.detach(), clean)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr(denoised.detach(), clean):.2f}'
        })
    
    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    return avg_loss, avg_psnr


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """验证"""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        denoised = model(noisy)
        loss = criterion(denoised, clean)
        
        total_loss += loss.item()
        total_psnr += psnr(denoised, clean)
    
    avg_loss = total_loss / len(loader)
    avg_psnr = total_psnr / len(loader)
    return avg_loss, avg_psnr


def main():
    parser = argparse.ArgumentParser(description='DnCNN Training')
    parser.add_argument('--data_dir', type=str, default='data/train')
    parser.add_argument('--exp_name', type=str, default='exp01_baseline')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--noise_sigma', type=float, default=25.0)
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设备检查 (项目规则要求)
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, training on CPU (will be slow)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建输出目录 (项目规则要求)
    exp_dir = os.path.join('experiments', args.exp_name)
    weights_dir = os.path.join(exp_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    # 数据加载
    train_loader = get_dataloader(
        image_dir=args.data_dir,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        noise_sigma=args.noise_sigma,
    )
    print(f"Train batches: {len(train_loader)}")
    
    # 模型
    model = DnCNN(use_attention=args.use_attention).to(device)
    print(f"Model: DnCNN {'+ CBAM' if args.use_attention else '(baseline)'}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()
    
    # 训练记录
    best_psnr = 0.0
    history = {'train_loss': [], 'train_psnr': []}
    
    print(f"\n{'='*50}")
    print(f"Starting training: {args.exp_name}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        history['train_loss'].append(train_loss)
        history['train_psnr'].append(train_psnr)
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, PSNR={train_psnr:.2f} dB")
        
        # 保存最佳模型
        if train_psnr > best_psnr:
            best_psnr = train_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
            }, os.path.join(weights_dir, 'best.pth'))
            print(f"  -> Saved best model (PSNR: {best_psnr:.2f} dB)")
        
        # 定期保存
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': train_psnr,
            }, os.path.join(weights_dir, f'epoch_{epoch}.pth'))
        
        scheduler.step()
        
        # 早停检查: NaN 检测 (workflow 要求)
        if np.isnan(train_loss):
            print("[ERROR] Loss is NaN! Stopping training.")
            break
    
    elapsed = time.time() - start_time
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'psnr': train_psnr,
    }, os.path.join(weights_dir, 'last.pth'))
    
    print(f"\n{'='*50}")
    print(f"Training completed in {elapsed/60:.1f} minutes")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Weights saved to: {weights_dir}")
    print(f"{'='*50}")
    
    # 保存训练历史
    import json
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    main()

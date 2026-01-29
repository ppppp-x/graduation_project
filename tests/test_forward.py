"""
显存安全测试
验证 forward pass 不爆 4GB RTX 3050
"""
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import random

# 可复现性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def test_forward_pass():
    """测试 forward pass 显存使用"""
    from model import DnCNN, count_parameters
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type != 'cuda':
        print("[WARN] No GPU available, skipping VRAM check")
    
    # 创建模型
    model = DnCNN(use_attention=True).to(device)
    model.train()  # 训练模式 (BN 需要)
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # 模拟训练 batch
    batch_size = 8
    patch_size = 64
    x = torch.randn(batch_size, 3, patch_size, patch_size, device=device)
    
    # Forward pass
    torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
    
    y = model(x)
    
    # 模拟 backward (真正吃显存的地方)
    loss = y.mean()
    loss.backward()
    
    # 检查显存
    if device.type == 'cuda':
        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak VRAM: {vram_gb:.2f} GB")
        
        if vram_gb > 3.0:
            print(f"[FAIL] VRAM usage {vram_gb:.2f} GB > 3 GB limit!")
            return False
    
    # 检查输出形状
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"[PASS] Forward + backward OK")
    return True


def test_dataloader():
    """测试 DataLoader"""
    from dataset import get_dataloader
    
    loader = get_dataloader(
        image_dir="data/train",
        batch_size=8,
        patch_size=64,
    )
    
    noisy, clean = next(iter(loader))
    
    assert noisy.shape == (8, 3, 64, 64), f"Wrong noisy shape: {noisy.shape}"
    assert clean.shape == (8, 3, 64, 64), f"Wrong clean shape: {clean.shape}"
    assert 0 <= noisy.min() and noisy.max() <= 1, "Noisy not normalized"
    assert 0 <= clean.min() and clean.max() <= 1, "Clean not normalized"
    
    print(f"Noisy: {noisy.shape}, range=[{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"Clean: {clean.shape}, range=[{clean.min():.3f}, {clean.max():.3f}]")
    print(f"[PASS] DataLoader OK")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("DnCNN + CBAM 显存安全测试")
    print("=" * 50)
    
    success = True
    
    print("\n[1] Testing DataLoader...")
    success = test_dataloader() and success
    
    print("\n[2] Testing Forward + Backward...")
    success = test_forward_pass() and success
    
    print("\n" + "=" * 50)
    if success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)

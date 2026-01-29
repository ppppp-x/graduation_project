#!/usr/bin/env python
"""
魔塔 ModelScope 云端训练脚本
一键运行：python run_modelscope.py

这个脚本会:
1. 检查环境
2. 安装依赖
3. 运行两个实验 (baseline + CBAM)
4. 保存结果
"""
import os
import sys
import subprocess
import time


def install_deps():
    """安装依赖"""
    print("=" * 50)
    print("Step 1: Installing dependencies...")
    print("=" * 50)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"
    ], check=True)
    print("Done.\n")


def check_gpu():
    """检查 GPU"""
    print("=" * 50)
    print("Step 2: Checking GPU...")
    print("=" * 50)
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_mem:.1f} GB")
    else:
        print("[WARN] No GPU detected! Training will be slow.")
    print()


def run_training(exp_name: str, use_attention: bool, epochs: int = 50):
    """运行训练"""
    print("=" * 50)
    print(f"Step 3: Training {exp_name}...")
    print("=" * 50)
    
    cmd = [
        sys.executable, "src/train.py",
        "--exp_name", exp_name,
        "--epochs", str(epochs),
        "--batch_size", "16",  # 云端 GPU 显存大，可以用更大 batch
        "--patch_size", "128", # 云端可以用更大 patch
    ]
    
    if use_attention:
        cmd.append("--use_attention")
    
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = (time.time() - start) / 60
    
    print(f"\n{exp_name} completed in {elapsed:.1f} minutes.\n")


def run_evaluation(exp_name: str, use_attention: bool):
    """运行评估"""
    print("=" * 50)
    print(f"Step 4: Evaluating {exp_name}...")
    print("=" * 50)
    
    weights_path = f"experiments/{exp_name}/weights/best.pth"
    output_path = f"experiments/{exp_name}/visual_check.png"
    
    # 用第一张训练图测试
    test_image = "data/train/P0000.jpg"
    
    cmd = [
        sys.executable, "src/eval.py",
        "--weights", weights_path,
        "--image", test_image,
        "--output", output_path,
        "--patch_size", "128",
    ]
    
    if use_attention:
        cmd.append("--use_attention")
    
    subprocess.run(cmd, check=True)
    print()


def main():
    print("\n" + "=" * 60)
    print("  DnCNN 遥感降噪训练 - 魔塔 ModelScope 云端版")
    print("=" * 60 + "\n")
    
    # 1. 安装依赖
    install_deps()
    
    # 2. 检查 GPU
    check_gpu()
    
    # 3. 训练 baseline
    run_training("baseline_50ep", use_attention=False, epochs=50)
    run_evaluation("baseline_50ep", use_attention=False)
    
    # 4. 训练 CBAM 版本
    run_training("cbam_50ep", use_attention=True, epochs=50)
    run_evaluation("cbam_50ep", use_attention=True)
    
    # 5. 打印结果
    print("\n" + "=" * 60)
    print("  All experiments completed!")
    print("=" * 60)
    print("\nResults saved to:")
    print("  - experiments/baseline_50ep/")
    print("  - experiments/cbam_50ep/")
    print("\nDownload the 'experiments' folder to get your trained models.")


if __name__ == "__main__":
    main()

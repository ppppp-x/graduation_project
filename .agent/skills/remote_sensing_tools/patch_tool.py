import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preview_noisy_patch(image_dir, noise_level=25.0):
    """
    随机裁剪一个切片，添加噪声，并保存预览图。
    返回保存预览图的路径。
    """
    # 1. 随机找一张图片
    files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]
    if not files:
        return "错误: 目录下没有找到图片。"
    
    img_path = os.path.join(image_dir, np.random.choice(files))
    img = cv2.imread(img_path)
    
    # 2. 裁剪 256x256 的切片 (模拟 Data Loader 的行为)
    h, w, _ = img.shape
    # 防止图片小于裁剪尺寸
    if h <= 256 or w <= 256:
        return f"错误: 图片尺寸太小 ({h}x{w})，无法裁剪 256x256 的切片。"

    x = np.random.randint(0, w - 256)
    y = np.random.randint(0, h - 256)
    patch = img[y:y+256, x:x+256]
    
    # 3. 添加噪声 (Add Noise)
    noise = np.random.normal(0, noise_level, patch.shape).astype(np.float32)
    noisy_patch = patch.astype(np.float32) + noise
    noisy_patch = np.clip(noisy_patch, 0, 255).astype(np.uint8)
    
    # 4. 保存可视化对比图
    save_path = "preview_debug.png"
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # 显示清晰真值 (Ground Truth)
    ax[0].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Clean Ground Truth (真值)")
    ax[0].axis('off')
    
    # 显示含噪输入 (Noisy Input)
    ax[1].imshow(cv2.cvtColor(noisy_patch, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Noisy Input (Sigma={noise_level})")
    ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return f"预览图已保存至 {os.path.abspath(save_path)}。请检查图中的小目标是否依然可辨。"
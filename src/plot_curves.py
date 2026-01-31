"""
绘制训练曲线
"""
import json
import matplotlib.pyplot as plt
import os

def plot_training_curves(exp_dirs: list, labels: list, output_path: str):
    """
    绘制多个实验的训练曲线对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for exp_dir, label in zip(exp_dirs, labels):
        history_path = os.path.join(exp_dir, 'history.json')
        if not os.path.exists(history_path):
            print(f"Warning: {history_path} not found")
            continue
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss 曲线
        axes[0].plot(epochs, history['train_loss'], label=label, linewidth=2)
        
        # PSNR 曲线
        axes[1].plot(epochs, history['train_psnr'], label=label, linewidth=2)
    
    # 设置 Loss 图
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 设置 PSNR 图
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('Training PSNR', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {output_path}")


if __name__ == "__main__":
    # Baseline vs CBAM 对比
    plot_training_curves(
        exp_dirs=['experiments/baseline_50ep', 'experiments/cbam_50ep'],
        labels=['DnCNN Baseline', 'DnCNN + CBAM'],
        output_path='experiments/comparison_curves.png'
    )

"""
DnCNN-17 降噪网络 + 可选 CBAM 注意力
针对 4GB 显存优化
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """CBAM 通道注意力"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """完整 CBAM 模块"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class DnCNN(nn.Module):
    """
    DnCNN-17 残差学习降噪网络
    
    结构:
    - 第1层: Conv + ReLU
    - 第2-16层: Conv + BN + ReLU
    - 第17层: Conv
    - 可选: 在第9层后插入 CBAM
    
    输出: 噪声残差 (不是干净图像)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_layers: int = 17,
        use_attention: bool = False,
    ):
        super().__init__()
        
        self.use_attention = use_attention
        layers = []
        
        # 第1层: Conv + ReLU (无 BN)
        layers.append(nn.Conv2d(in_channels, num_features, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层: Conv + BN + ReLU
        for i in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
            
            # 在第8层后插入 CBAM (i=7, 因为从0开始)
            if use_attention and i == 7:
                layers.append(CBAM(num_features))
        
        # 最后一层: Conv (无 BN, 无 ReLU)
        layers.append(nn.Conv2d(num_features, out_channels, 3, padding=1, bias=False))
        
        self.dncnn = nn.Sequential(*layers)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 残差学习: 输出噪声，干净图 = 输入 - 噪声
        noise = self.dncnn(x)
        return x - noise


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch
    torch.manual_seed(42)
    
    # 测试基线模型
    model = DnCNN(use_attention=False)
    print(f"DnCNN (no attention): {count_parameters(model):,} params")
    
    # 测试带注意力的模型
    model_attn = DnCNN(use_attention=True)
    print(f"DnCNN + CBAM: {count_parameters(model_attn):,} params")
    
    # Forward test
    x = torch.randn(8, 3, 64, 64)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print("[PASS] Model works.")

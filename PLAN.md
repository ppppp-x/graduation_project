# 遥感影像降噪网络 (DnCNN + Attention) 开发计划

## 阶段一审计结论

### 1. 数据现状
- **训练集**: `data/train/` 包含 1411 张 DOTA 图像 (范围从 ~16KB 到 ~8MB)
- **噪声检查**: Sigma=25 的高斯噪声级别合适。但需注意：
  - ⚠️ **低对比度 Patch 问题**: 随机采样可能抽到纯黑/低方差区域，导致训练无效
  - ✅ **解决方案**: 在 Dataset 中增加方差阈值过滤

### 2. 硬件约束 (铁律)
| 资源 | 限制 |
|------|------|
| GPU | RTX 3050 (4GB VRAM) |
| 安全阈值 | 单 Batch 显存 < 3GB (留 1GB 余量) |

### 3. 编码红线 (来自 `.agent/rules/graduation_project.md`)
- ❌ 禁止: TensorFlow/Keras
- ❌ 禁止: 将 20000x20000 原图读入显存
- ✅ 必须: PyTorch + albumentations
- ✅ 必须: 脚本开头设置随机种子
- ✅ 必须: 权重保存到 `experiments/<exp_name>/weights/`

---

## 阶段二：4GB 显存优化策略

### 核心参数选择

```
Patch Size: 64x64  (不是 128，内存优先)
Batch Size: 8      (64x64x3xfp32 x 8 = ~100KB input, 可接受)
Workers:    0      (Windows 下 DataLoader 多进程坑太多，不值得)
```

**内存分析**:
- DnCNN-17 层 (无 Attention): ~550K 参数 → ~2MB 权重
- 加 CBAM Attention: 增加 ~100K 参数 → 共 ~2.5MB
- Forward 激活 (64x64, batch=8): ~500MB 峰值
- **总估算**: < 1GB，安全

### 文件结构规划

```
d:\毕设项目\
├── src\
│   ├── dataset.py      # 数据加载（含方差过滤）
│   ├── model.py        # DnCNN 基线 + 可选 Attention
│   ├── train.py        # 训练脚本
│   └── eval.py         # 评估脚本
├── experiments\        # 实验结果输出
├── data\train\         # 训练图像
└── PLAN.md             # 本文件
```

---

## 阶段三：实现清单

### 3.1 `src/dataset.py`

**设计要点**:
1. 使用 `albumentations.RandomCrop(64, 64)` 裁剪
2. 在 `__getitem__` 中动态添加高斯噪声
3. **关键**: 增加 `min_variance` 参数，跳过方差太低的 Patch
4. Normalize 到 [0, 1] 或 [-1, 1]

```python
# 伪代码逻辑
def __getitem__(idx):
    img = read_image(self.files[idx])
    patch = random_crop(img, 64, 64)
    if patch.var() < self.min_variance:
        return self.__getitem__(random.randint(0, len(self)-1))  # 重抽
    noisy = patch + gaussian_noise(sigma=25/255)
    return noisy, patch
```

### 3.2 `src/model.py`

**DnCNN 结构** (Residual Learning):
- 17 层 Conv-BN-ReLU
- 输入: 含噪图像
- 输出: **残差 (噪声估计)**，不是直接输出干净图像
- 可选: 在中间层插入 CBAM 模块

### 3.3 验证测试 (Forward Pass)

创建 `tests/test_forward.py`:
```bash
python tests/test_forward.py  # 必须通过且显存 < 3GB
```

---

## 当前阻塞项

无。规划完成，可进入 EXECUTION 阶段。

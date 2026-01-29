# 魔塔 ModelScope 云端训练指南

## 准备工作

需要上传的文件/文件夹：
```
毕设项目/
├── data/train/           # 训练数据 (1411 张图)
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── eval.py
├── requirements.txt
└── run_modelscope.py     # 一键训练脚本
```

## 步骤

### 1. 登录魔塔创空间
- 访问 https://www.modelscope.cn/
- 注册/登录账号
- 进入 **创空间 (Spaces)** → **创建空间**
- 选择 **Gradio** 或 **Notebook** 模板

### 2. 申请 GPU 算力
- 选择 **A10 GPU** (24GB 显存) 或 **V100** (16GB)
- 免费额度每月约 100 GPU 小时

### 3. 上传代码
方式一：直接上传
- 在创空间中点击 **上传文件**
- 上传整个项目文件夹

方式二：Git 克隆 (推荐)
- 把代码推到 GitHub/Gitee
- 在终端执行 `git clone <你的仓库地址>`

### 4. 上传训练数据
- 将 `data/train/` 文件夹打包成 zip
- 上传后解压：`unzip train.zip -d data/`

### 5. 一键训练
```bash
# 在创空间终端中执行
python run_modelscope.py
```

脚本会自动：
1. 安装依赖
2. 训练 baseline (50 epochs)
3. 训练 CBAM 版本 (50 epochs)
4. 生成对比图

### 6. 下载结果
训练完成后，下载 `experiments/` 文件夹，包含：
- `baseline_50ep/weights/best.pth`
- `cbam_50ep/weights/best.pth`
- 对比图

## 预估时间
| 平台 | 50 Epochs 耗时 |
|------|---------------|
| 本地 RTX 3050 | ~15 小时 |
| 魔塔 A10 | ~2-3 小时 |
| 魔塔 V100 | ~1.5-2 小时 |

## 注意事项
- 魔塔空间有时间限制 (通常 2-6 小时)，建议分批训练
- 使用 `screen` 或 `nohup` 防止断联中断

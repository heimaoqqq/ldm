# 潜在扩散模型 (LDM) 实现

本项目实现了一个两阶段的潜在扩散模型 (Latent Diffusion Model)，包括:
1. VAE 自编码器 (Vector Quantized VAE)
2. 潜在扩散模型 (Latent Diffusion Model)

## 项目结构

```
├── models/                  # 模型实现
│   ├── vae.py               # VAE 实现
│   └── ldm.py               # 潜在扩散模型实现
├── utils/                   # 工具函数
│   ├── dataset.py           # 数据集处理
│   └── helpers.py           # 辅助函数
├── train_vae.py             # VAE 训练脚本
├── train_ldm.py             # LDM 训练脚本
├── main.py                  # 主运行脚本
├── inference.py             # 图像生成推理脚本
├── visualize.py             # 训练结果可视化脚本
└── requirements.txt         # 项目依赖
```

## 环境要求

```
torch>=1.7.0
torchvision>=0.8.1
numpy>=1.19.2
matplotlib>=3.3.2
tqdm>=4.50.2
```

## 模型架构

### VAE 自编码器

- **编码器**: 4个下采样模块，每个模块包含2D卷积层(kernel=4, stride=2) + 批归一化 + LeakyReLU(0.2) + 2个残差块
- **量化模块**: 向量量化(VQ)层，编码本大小8192，潜在空间维度256
- **解码器**: 4个上采样模块，每个模块包含转置卷积(kernel=4, stride=2) + 2个残差块
- **判别器**: PatchGAN(70×70)
- **对抗训练**: 使用动态计算的权重平衡重建损失与对抗损失，避免训练不稳定或模式崩溃，权重根据当前批次的重建损失与对抗损失比例自适应调整

### 潜在扩散模型(LDM)

- **U-Net**: 编码路径初始通道数256，3个下采样层，注意力机制可配置（默认在16×16和8×8分辨率启用）
- **扩散过程**: 支持余弦（默认）或线性噪声调度，DDPM采样，1000步扩散和采样
- **梯度裁剪**: 最大范数为1.0

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 完整流程训练

```bash
python main.py --data_dir dataset --vae_epochs 3 --ldm_epochs 3 --attention_resolutions 16,8
```

### 3. 单独训练VAE

```bash
python train_vae.py --data_dir dataset --epochs 3
```

### 4. 单独训练LDM (需要预训练的VAE)

```bash
python train_ldm.py --data_dir dataset --epochs 3 --vae_checkpoint path/to/vae_checkpoint.pth --noise_schedule cosine --attention_resolutions 16,8
```

### 5. 使用训练好的模型生成图像

```bash
python inference.py --vae_checkpoint path/to/vae_checkpoint.pth --ldm_checkpoint path/to/ldm_checkpoint.pth --num_samples 16
```

### 6. 可视化训练结果

```bash
python visualize.py
```

## 参数说明

### 通用参数
- `--data_dir`: 数据集目录
- `--image_size`: 图像大小 (默认: 256)
- `--latent_dim`: 潜在空间维度 (默认: 256)
- `--num_embeddings`: 编码本大小 (默认: 8192)
- `--commitment_cost`: 承诺损失系数 (默认: 0.25)
- `--max_grad_norm`: 梯度裁剪最大范数 (默认: 1.0)

### VAE参数
- `--vae_batch_size`: VAE训练批次大小 (默认: 32)
- `--vae_epochs`: VAE训练轮数 (默认: 3)
- `--vae_lr`: VAE学习率 (默认: 1e-4)

### LDM参数
- `--ldm_batch_size`: LDM训练批次大小 (默认: 8)
- `--ldm_epochs`: LDM训练轮数 (默认: 3)
- `--ldm_lr`: LDM学习率 (默认: 1e-4)
- `--noise_steps`: 扩散步数 (默认: 1000)
- `--noise_schedule`: 噪声调度类型，可选: linear或cosine (默认: cosine)
- `--attention_resolutions`: 启用注意力的分辨率列表 (默认: "16,8")

## 输出结果

训练过程中会自动创建以下目录:
- `checkpoints/`: 保存模型检查点
- `samples/`: 保存生成的样本图像
- `logs/`: 保存训练日志
- `generated_samples/`: 使用inference.py生成的图像
- `visualization/`: 可视化训练过程和生成结果 
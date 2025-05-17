import os
import argparse
import torch
from train_vae import train_vae
from train_ldm import train_ldm

def get_args():
    parser = argparse.ArgumentParser(description='微多普勒时频图生成模型训练')
    
    # 数据集参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--image_size', type=int, default=256, help='图像大小 (默认: 256)')
    
    # 训练阶段选择
    parser.add_argument('--train_vae_only', action='store_true', help='仅训练VAE模型')
    parser.add_argument('--train_ldm_only', action='store_true', help='仅训练LDM模型')
    
    # VAE参数
    parser.add_argument('--vae_batch_size', type=int, default=32, help='VAE训练批次大小 (默认: 32)')
    parser.add_argument('--vae_epochs', type=int, default=300, help='VAE训练轮数 (默认: 300)')
    parser.add_argument('--vae_lr', type=float, default=1e-4, help='VAE学习率 (默认: 1e-4)')
    parser.add_argument('--disc_lr_factor', type=float, default=0.5, help='判别器学习率因子 (默认: 0.5)')
    parser.add_argument('--freq_augment', action='store_true', help='使用频域数据增强')
    parser.add_argument('--use_attention', action='store_true', help='在VAE中使用注意力机制')
    parser.add_argument('--use_freq', action='store_true', help='在VAE中使用频域增强')
    parser.add_argument('--use_ema', action='store_true', help='使用EMA更新模型参数')
    parser.add_argument('--use_perceptual', action='store_true', help='使用感知损失')
    parser.add_argument('--use_spectral_norm', action='store_true', help='使用谱归一化')
    
    # LDM参数
    parser.add_argument('--ldm_batch_size', type=int, default=8, help='LDM训练批次大小 (默认: 8)')
    parser.add_argument('--ldm_epochs', type=int, default=500, help='LDM训练轮数 (默认: 500)')
    parser.add_argument('--ldm_lr', type=float, default=1e-4, help='LDM学习率 (默认: 1e-4)')
    parser.add_argument('--latent_dim', type=int, default=256, help='潜在空间维度 (默认: 256)')
    parser.add_argument('--num_embeddings', type=int, default=8192, help='编码本大小 (默认: 8192)')
    parser.add_argument('--commitment_cost', type=float, default=0.15, help='承诺损失系数 (默认: 0.15)')
    parser.add_argument('--noise_steps', type=int, default=1000, help='扩散步数 (默认: 1000)')
    parser.add_argument('--noise_schedule', type=str, default='cosine', help='噪声调度类型 (默认: cosine)')
    parser.add_argument('--attention_resolutions', type=str, default='16,8,4', help='注意力层分辨率 (默认: 16,8,4)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪最大范数')
    
    # 检查点参数
    parser.add_argument('--vae_checkpoint', type=str, default=None, help='VAE检查点路径')
    parser.add_argument('--ldm_checkpoint', type=str, default=None, help='LDM检查点路径')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # 检查CUDA可用性
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
    else:
        print("警告: 未检测到可用GPU，将使用CPU训练 (速度会很慢)")
    
    # 创建目录结构
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 训练VAE
    if not args.train_ldm_only:
        print("\n" + "="*50)
        print("第一阶段: 训练VAE模型")
        print("="*50)
        
        # 构建VAE参数
        vae_args = argparse.Namespace(
            data_dir=args.data_dir,
            image_size=args.image_size,
            batch_size=args.vae_batch_size,
            epochs=args.vae_epochs,
            lr=args.vae_lr,
            disc_lr_factor=args.disc_lr_factor, 
            latent_dim=args.latent_dim,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            max_grad_norm=args.max_grad_norm,
            disc_filters=32,  # 使用较小的判别器
            use_attention=args.use_attention,
            use_freq=args.use_freq,
            use_ema=args.use_ema,
            use_perceptual=args.use_perceptual,
            use_spectral_norm=args.use_spectral_norm,
            freq_augment=args.freq_augment,
            resume=args.vae_checkpoint,
            save_interval=5  # 添加保存间隔参数
        )
        
        # 训练VAE
        vae_checkpoint = train_vae(vae_args)
    else:
        vae_checkpoint = args.vae_checkpoint
        if not vae_checkpoint:
            raise ValueError("如果跳过VAE训练，必须提供VAE检查点路径")
    
    # 训练LDM
    if not args.train_vae_only:
        print("\n" + "="*50)
        print("第二阶段: 训练LDM模型")
        print("="*50)
        
        # 构建LDM参数
        ldm_args = argparse.Namespace(
            data_dir=args.data_dir,
            image_size=args.image_size,
            batch_size=args.ldm_batch_size,
            epochs=args.ldm_epochs,
            lr=args.ldm_lr,
            latent_dim=args.latent_dim,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            noise_steps=args.noise_steps,
            noise_schedule=args.noise_schedule,
            attention_resolutions=args.attention_resolutions,
            max_grad_norm=args.max_grad_norm,
            use_ema=args.use_ema,
            sample_interval=100,
            save_interval=5,  # 添加保存间隔参数
            vae_checkpoint=vae_checkpoint,  # 直接使用前一阶段的结果
            resume=args.ldm_checkpoint
        )
        
        # 训练LDM
        ldm_checkpoint = train_ldm(ldm_args)
    
    print("\n" + "="*50)
    print("训练完成!")
    print("="*50)

if __name__ == "__main__":
    main() 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import argparse

from models.vae import VQVAE
from models.ldm import UNet, LatentDiffusionModel, DiffusionModel
from utils.dataset import get_data_loaders
from utils.helpers import save_checkpoint, load_checkpoint, setup_directories

def parse_attention_resolutions(resolution_str):
    """解析注意力分辨率字符串为整数列表"""
    if not resolution_str:
        return []
    return [int(res) for res in resolution_str.split(',')]

def train_ldm(args):
    """训练潜在扩散模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    print(f"使用设备: {device}")
    
    # 确认CUDA是否可用
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
    else:
        print("警告: 未检测到可用GPU，将使用CPU训练 (速度会很慢)")
    
    # 设置目录
    dirs = setup_directories()
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )
    
    # 使用经典的32×32×256潜在空间设计
    print(f"\n潜在空间设计: 32×32×{args.latent_dim}")
    latent_size = args.image_size // 8  # 对于经典设计，256×256图像会得到32×32的潜在空间
    
    # 加载预训练的VAE模型
    vae = VQVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost
    ).to(device)
    
    if args.vae_checkpoint:
        checkpoint = torch.load(args.vae_checkpoint, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载预训练VAE: {args.vae_checkpoint}")
    else:
        raise ValueError("必须提供预训练的VAE检查点")
    
    # 解析注意力分辨率
    attention_resolutions = parse_attention_resolutions(args.attention_resolutions)
    print(f"注意力机制将在以下分辨率启用: {attention_resolutions}")
    
    # 初始化U-Net扩散模型，指定注意力分辨率
    unet = UNet(
        in_channels=args.latent_dim,
        time_dim=args.latent_dim,
        attention_resolutions=attention_resolutions
    ).to(device)
    
    # 初始化潜在扩散模型
    ldm = LatentDiffusionModel(
        unet=unet,
        vae=vae,
        latent_dim=args.latent_dim,
        device=device,
        noise_schedule=args.noise_schedule
    ).to(device)
    
    # 初始化优化器
    optimizer = optim.Adam(ldm.unet.parameters(), lr=args.lr)
    
    # 初始化学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 定义扩散模型，并指定噪声调度类型
    diffusion = DiffusionModel(
        noise_steps=args.noise_steps,
        img_size=latent_size,  # 使用经典的32×32潜在空间
        device=device,
        schedule_type=args.noise_schedule
    )
    
    # 加载检查点（如果存在）
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            ldm.unet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"从轮次 {start_epoch} 恢复训练")
        else:
            print(f"找不到检查点 {args.resume}，从头开始训练")
    
    # 记录最佳验证损失，用于模型选择
    best_val_loss = float('inf')
    best_model_path = ""
    
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        # ==================== 训练阶段 ====================
        ldm.train()
        
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"训练轮次 [{epoch+1}/{args.epochs}]", leave=True)
        for i, batch in enumerate(progress_bar):
            images = batch.to(device)
            
            # 采样随机时间步
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            
            # 计算损失
            optimizer.zero_grad()
            
            noise, predicted_noise, _ = ldm(images, t)
            
            # 均方误差损失
            loss = nn.MSELoss()(noise, predicted_noise)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(ldm.unet.parameters(), args.max_grad_norm)
            
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_description(
                f"训练轮次 [{epoch+1}/{args.epochs}] 损失: {loss.item():.4f}"
            )
            
            # 生成样本（可视化）
            if i % args.sample_interval == 0:
                with torch.no_grad():
                    samples = ldm.sample(n=1)  # 只生成一张图像
                    samples = (samples + 1) / 2  # 反归一化到[0, 1]
                    sample_path = os.path.join(dirs['samples'], f"ldm_epoch_{epoch+1}_batch_{i}.png")
                    
                    # 保存生成的样本图像 - 只保存第一张
                    from torchvision.utils import save_image
                    save_image(samples[0], sample_path)
                    print(f"训练样本已保存: {sample_path}")
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        
        # ==================== 验证阶段 ====================
        ldm.eval()
        
        val_loss = 0
        
        progress_bar = tqdm(val_loader, desc=f"验证轮次 [{epoch+1}/{args.epochs}]", leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                images = batch.to(device)
                
                # 采样随机时间步
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                
                # 执行前向传播
                noise, predicted_noise, _ = ldm(images, t)
                
                # 计算损失
                loss = nn.MSELoss()(noise, predicted_noise)
                
                # 更新统计信息
                val_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_description(
                    f"验证轮次 [{epoch+1}/{args.epochs}] 损失: {loss.item():.4f}"
                )
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 打印训练和验证信息
        print(f"\n轮次 [{epoch+1}/{args.epochs}] 结果:")
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证损失: {avg_val_loss:.4f}")
        
        # 生成最终样本图像
        with torch.no_grad():
            samples = ldm.sample(n=1)  # 只生成一张图像
            samples = (samples + 1) / 2  # 反归一化到[0, 1]
            sample_path = os.path.join(dirs['samples'], f"ldm_epoch_{epoch+1}_final.png")
            
            # 保存生成的样本图像 - 只保存一张
            from torchvision.utils import save_image
            save_image(samples[0], sample_path)
            print(f"轮次结束样本已保存: {sample_path}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(dirs['checkpoints'], f"ldm_best.pth")
            save_checkpoint(
                ldm.unet,
                optimizer,
                epoch + 1,
                avg_val_loss,
                best_model_path
            )
            print(f"发现新的最佳模型！已保存至: {best_model_path}")
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                ldm.unet,
                optimizer,
                epoch + 1,
                avg_train_loss,
                os.path.join(dirs['checkpoints'], f"ldm_epoch_{epoch+1}.pth")
            )
    
    # 保存最终模型
    save_checkpoint(
        ldm.unet,
        optimizer,
        args.epochs,
        avg_train_loss,
        os.path.join(dirs['checkpoints'], "ldm_final.pth")
    )
    
    print(f"\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型路径: {best_model_path}")
    print(f"最终模型路径: {os.path.join(dirs['checkpoints'], 'ldm_final.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练潜在扩散模型")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--image_size", type=int, default=256, help="图像大小")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--latent_dim", type=int, default=256, help="潜在空间维度")
    parser.add_argument("--num_embeddings", type=int, default=8192, help="VAE编码本大小")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="VAE承诺损失系数")
    parser.add_argument("--noise_steps", type=int, default=1000, help="扩散步数")
    parser.add_argument("--noise_schedule", type=str, default="cosine", help="噪声调度类型，可选: linear或cosine")
    parser.add_argument("--attention_resolutions", type=str, default="16,8", help="在指定分辨率启用注意力，如'16,8'")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--sample_interval", type=int, default=100, help="生成样本的间隔（批次）")
    parser.add_argument("--save_interval", type=int, default=1, help="保存检查点的间隔（轮次）")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="预训练VAE检查点文件")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的LDM检查点文件")
    
    args = parser.parse_args()
    
    train_ldm(args) 
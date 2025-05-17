import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import argparse

from models.vae import VQVAE, PatchGANDiscriminator
from utils.dataset import get_data_loaders
from utils.helpers import save_checkpoint, load_checkpoint, save_samples, plot_reconstruction, setup_directories

def train_vae(args):
    """训练VAE模型"""
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
    
    # 获取数据加载器（现在包括验证集）
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        use_freq_augment=args.freq_augment
    )
    
    # 使用经典的32×32×256潜在空间设计
    print(f"\n潜在空间设计: 32×32×{args.latent_dim}")
    # 调整下采样层数，以获得32×32的特征图大小
    downsample_factor = args.image_size // 32
    
    # 初始化模型
    vae = VQVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        use_attention=args.use_attention,
        use_freq=args.use_freq,
        use_ema=args.use_ema,
        use_perceptual=args.use_perceptual
    ).to(device)
    
    discriminator = PatchGANDiscriminator(
        in_channels=3,
        ndf=args.disc_filters,
        use_spectral_norm=args.use_spectral_norm
    ).to(device)
    
    # 初始化优化器
    vae_optimizer = optim.Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr * args.disc_lr_factor, betas=(0.5, 0.999))
    
    # 使用学习率优化器
    print(f"使用OneCycleLR学习率调度，初始学习率: {args.lr}, 最大轮次: {args.epochs}")
    vae_scheduler = optim.lr_scheduler.OneCycleLR(
        vae_optimizer,
        max_lr=args.lr,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.2,  # 前20%用于预热
        div_factor=25.0,  # 初始学习率为最大学习率的 1/25
        final_div_factor=5000.0  # 修改为更温和的衰减速率
    )
    
    disc_scheduler = optim.lr_scheduler.OneCycleLR(
        disc_optimizer,
        max_lr=args.lr * args.disc_lr_factor,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.2,
        div_factor=25.0,
        final_div_factor=5000.0  # 修改为更温和的衰减速率
    )
    
    # 梯度裁剪配置
    max_grad_norm = 1.0
    print(f"使用梯度裁剪，最大范数: {max_grad_norm}")
    
    # 定义损失函数
    recon_criterion = nn.MSELoss(reduction='mean')
    adv_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    # 加载检查点（如果存在）
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            vae.load_state_dict(checkpoint['model_state_dict'])
            vae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
        vae.train()
        discriminator.train()
        
        train_recon_loss = 0
        train_gan_loss = 0
        train_commit_loss = 0
        train_total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"训练轮次 [{epoch+1}/{args.epochs}]", leave=True)
        for batch in progress_bar:
            real_images = batch.to(device)
            batch_size = real_images.size(0)
            
            # ============= 训练判别器 =============
            disc_optimizer.zero_grad()
            
            # 使用VAE生成重建图像
            with torch.no_grad():
                reconstructed, _, _ = vae(real_images)
            
            # 真实图像的判别器输出
            real_logits = discriminator(real_images)
            real_labels = torch.ones_like(real_logits, device=device)
            
            # 重建图像的判别器输出
            fake_logits = discriminator(reconstructed)
            fake_labels = torch.zeros_like(fake_logits, device=device)
            
            # 判别器损失 - 确保是标量
            d_real_loss = adv_criterion(real_logits, real_labels)
            d_fake_loss = adv_criterion(fake_logits, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) * 0.5  # 取平均
            
            # 反向传播判别器损失
            d_loss.backward()
            
            # 对判别器应用梯度裁剪
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
            
            disc_optimizer.step()
            
            # ============= 训练VAE =============
            vae_optimizer.zero_grad()
            
            # 通过VAE前向传播
            reconstructed, commit_loss, _ = vae(real_images)
            
            # 重建图像的判别器输出
            fake_logits = discriminator(reconstructed)
            
            # 生成器损失 - 欺骗判别器将假图像识别为真
            adv_loss = adv_criterion(fake_logits, real_labels)
            
            # 重建损失 - 使重建图像接近原始图像
            recon_loss = recon_criterion(reconstructed, real_images)
            
            # 使用VQVAE的损失计算方法，实现动态平衡
            # 原始公式: L_Stage1 = L_rec(x, D(E(x))) - lambda_adv * L_adv(Phi, x, D(E(x)))
            total_loss, recon_loss = vae.compute_loss(
                real_images, 
                reconstructed, 
                commit_loss, 
                adv_loss, 
                recon_criterion
            )
            
            # 反向传播总损失
            total_loss.backward()
            
            # 对VAE应用梯度裁剪
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_grad_norm)
            
            vae_optimizer.step()
            
            # 更新学习率(每个batch)
            vae_scheduler.step()
            disc_scheduler.step()
            
            # 更新统计信息
            train_recon_loss += recon_loss.item()
            train_gan_loss += adv_loss.item()
            train_commit_loss += commit_loss.item()
            train_total_loss += total_loss.item()
            
            # 更新进度条
            progress_bar.set_description(
                f"训练轮次 [{epoch+1}/{args.epochs}] "
                f"重建: {recon_loss.item():.4f}, "
                f"GAN: {adv_loss.item():.4f}, "
                f"承诺: {commit_loss.item():.4f}"
            )
        
        # 计算平均训练损失
        avg_train_recon_loss = train_recon_loss / len(train_loader)
        avg_train_gan_loss = train_gan_loss / len(train_loader)
        avg_train_commit_loss = train_commit_loss / len(train_loader)
        avg_train_total_loss = train_total_loss / len(train_loader)
        
        # ==================== 验证阶段 ====================
        vae.eval()
        discriminator.eval()
        
        val_recon_loss = 0
        val_gan_loss = 0
        val_commit_loss = 0
        val_total_loss = 0
        
        progress_bar = tqdm(val_loader, desc=f"验证轮次 [{epoch+1}/{args.epochs}]", leave=True)
        with torch.no_grad():
            for batch in progress_bar:
                real_images = batch.to(device)
                
                # 通过VAE前向传播
                reconstructed, commit_loss, _ = vae(real_images)
                
                # 重建图像的判别器输出
                fake_logits = discriminator(reconstructed)
                real_labels = torch.ones_like(fake_logits, device=device)
                
                # 计算损失
                adv_loss = adv_criterion(fake_logits, real_labels)
                recon_loss = recon_criterion(reconstructed, real_images)
                
                # 使用相同的损失计算函数
                total_loss, recon_loss = vae.compute_loss(
                    real_images, 
                    reconstructed, 
                    commit_loss, 
                    adv_loss, 
                    recon_criterion
                )
                
                # 更新统计信息
                val_recon_loss += recon_loss.item()
                val_gan_loss += adv_loss.item()
                val_commit_loss += commit_loss.item()
                val_total_loss += total_loss.item()
                
                # 更新进度条
                progress_bar.set_description(
                    f"验证轮次 [{epoch+1}/{args.epochs}] "
                    f"重建: {recon_loss.item():.4f}, "
                    f"GAN: {adv_loss.item():.4f}, "
                    f"承诺: {commit_loss.item():.4f}"
                )
        
        # 计算平均验证损失
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_gan_loss = val_gan_loss / len(val_loader)
        avg_val_commit_loss = val_commit_loss / len(val_loader)
        avg_val_total_loss = val_total_loss / len(val_loader)
        
        # 打印训练和验证信息
        print(f"\n轮次 [{epoch+1}/{args.epochs}] 结果:")
        print(f"训练 - 重建: {avg_train_recon_loss:.4f}, GAN: {avg_train_gan_loss:.4f}, 承诺: {avg_train_commit_loss:.4f}, 总损失: {avg_train_total_loss:.4f}")
        print(f"验证 - 重建: {avg_val_recon_loss:.4f}, GAN: {avg_val_gan_loss:.4f}, 承诺: {avg_val_commit_loss:.4f}, 总损失: {avg_val_total_loss:.4f}")
        print(f"当前学习率: {vae_scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            best_model_path = os.path.join(dirs['checkpoints'], f"vae_best.pth")
            save_checkpoint(
                vae,
                vae_optimizer,
                epoch + 1,
                avg_val_total_loss,
                best_model_path
            )
            print(f"发现新的最佳模型！已保存至: {best_model_path}")
        
        # 每10轮或最后一轮生成和保存图像
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                vae.eval()
                
                # 选择测试批次
                test_batch = next(iter(test_loader)).to(device)
                
                # 生成重建图像
                reconstructed, _, _ = vae(test_batch)
                
                # 绘制重建图像
                plot_reconstruction(
                    test_batch,
                    reconstructed,
                    dirs['samples'],
                    f"vae_reconstruction_epoch_{epoch+1}.png"
                )
                
                # 使用解码器生成样本
                save_samples(
                    vae.decoder,
                    device,
                    1,  # 只生成1个样本
                    args.image_size,
                    args.latent_dim,
                    dirs['samples'],
                    f"vae_samples_epoch_{epoch+1}.png"
                )
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                vae,
                vae_optimizer,
                epoch + 1,
                avg_train_total_loss,
                os.path.join(dirs['checkpoints'], f"vae_epoch_{epoch+1}.pth")
            )
    
    # 保存最终模型
    save_checkpoint(
        vae,
        vae_optimizer,
        args.epochs,
        avg_train_total_loss,
        os.path.join(dirs['checkpoints'], "vae_final.pth")
    )
    
    print(f"\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型路径: {best_model_path}")
    print(f"最终模型路径: {os.path.join(dirs['checkpoints'], 'vae_final.pth')}")
    
    return best_model_path  # 返回最佳模型路径

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练VAE模型")
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--image_size", type=int, default=256, help="图像大小")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--latent_dim", type=int, default=256, help="潜在空间维度")
    parser.add_argument("--num_embeddings", type=int, default=8192, help="编码本大小")
    parser.add_argument("--commitment_cost", type=float, default=0.15, help="承诺损失系数")
    parser.add_argument("--disc_filters", type=int, default=32, help="判别器基础过滤器数量")
    parser.add_argument("--save_interval", type=int, default=5, help="保存检查点的间隔（轮次）")
    parser.add_argument("--resume", type=str, default="", help="恢复训练的检查点文件")
    parser.add_argument("--use_attention", action="store_true", help="是否使用注意力机制")
    parser.add_argument("--use_freq", action="store_true", help="是否使用频率特征")
    parser.add_argument("--use_ema", action="store_true", help="是否使用指数移动平均")
    parser.add_argument("--use_perceptual", action="store_true", help="是否使用感知损失")
    parser.add_argument("--use_spectral_norm", action="store_true", help="是否使用谱归一化")
    parser.add_argument("--disc_lr_factor", type=float, default=0.5, help="判别器学习率因子")
    parser.add_argument("--freq_augment", action="store_true", help="是否使用频率增强")
    
    # 为微多普勒时频图设置推荐参数
    parser.set_defaults(
        use_attention=True, 
        use_freq=True, 
        use_ema=True, 
        use_spectral_norm=True,
        use_perceptual=False,  # 感知损失非常占显存
        freq_augment=True
    )
    
    args = parser.parse_args()
    
    train_vae(args) 

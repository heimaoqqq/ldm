import torch
import argparse
import os
from torchvision.utils import save_image
from models.vae import VQVAE
from models.ldm import UNet, LatentDiffusionModel
from utils.helpers import setup_directories

def generate_samples(args):
    """使用训练好的模型生成样本"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置目录
    sample_dir = "generated_samples"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # 加载VAE模型
    vae = VQVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost
    ).to(device)
    
    # 加载VAE检查点
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    print(f"VAE模型已加载: {args.vae_checkpoint}")
    
    # 加载U-Net扩散模型
    unet = UNet(
        in_channels=args.latent_dim,
        time_dim=args.latent_dim
    ).to(device)
    
    # 加载LDM检查点
    unet_checkpoint = torch.load(args.ldm_checkpoint, map_location=device)
    unet.load_state_dict(unet_checkpoint['model_state_dict'])
    print(f"LDM模型已加载: {args.ldm_checkpoint}")
    
    # 创建潜在扩散模型
    ldm = LatentDiffusionModel(
        unet=unet,
        vae=vae,
        latent_dim=args.latent_dim,
        device=device
    ).to(device)
    
    # 生成样本
    print(f"生成 {args.num_samples} 个样本...")
    ldm.eval()
    with torch.no_grad():
        samples = ldm.sample(n=args.num_samples)
        
        # 反归一化到[0, 1]
        samples = (samples + 1) / 2
        
        # 保存每个样本
        for i in range(args.num_samples):
            save_image(samples[i], os.path.join(sample_dir, f"sample_{i+1}.png"))
        
        # 保存网格图像
        save_image(samples, os.path.join(sample_dir, "samples_grid.png"), nrow=int(args.num_samples**0.5))
    
    print(f"样本已保存到 {sample_dir} 目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的模型生成图像")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="VAE模型检查点路径")
    parser.add_argument("--ldm_checkpoint", type=str, required=True, help="LDM模型检查点路径")
    parser.add_argument("--num_samples", type=int, default=16, help="要生成的样本数量")
    parser.add_argument("--latent_dim", type=int, default=256, help="潜在空间维度")
    parser.add_argument("--num_embeddings", type=int, default=8192, help="编码本大小")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="承诺损失系数")
    
    args = parser.parse_args()
    
    generate_samples(args) 
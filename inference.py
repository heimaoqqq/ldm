import torch
import argparse
import os
from torchvision.utils import save_image
from models.vae import VQVAE
from models.ldm import UNet, LatentDiffusionModel
import matplotlib.pyplot as plt
from utils.helpers import setup_directories

def parse_attention_resolutions(resolution_str):
    """解析注意力分辨率字符串为整数列表"""
    if not resolution_str:
        return []
    return [int(res) for res in resolution_str.split(',')]

def generate_samples(args):
    """使用训练好的模型生成样本"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建目录
    sample_dir = "generated_samples"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # 加载VAE模型
    vae = VQVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        use_attention=False,  # 关闭注意力机制以匹配训练的模型
        use_freq=False,       # 关闭频域增强以匹配训练的模型
        use_ema=True,
        use_perceptual=False
    ).to(device)
    
    # 加载VAE检查点
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    print(f"VAE模型已加载: {args.vae_checkpoint}")
    
    # 解析注意力分辨率
    attention_resolutions = parse_attention_resolutions(args.attention_resolutions)
    print(f"注意力机制在以下分辨率启用: {attention_resolutions}")
    
    # 加载U-Net扩散模型
    unet = UNet(
        in_channels=args.latent_dim,
        time_dim=args.latent_dim,
        attention_resolutions=attention_resolutions
    ).to(device)
    
    # 加载LDM检查点
    unet_checkpoint = torch.load(args.ldm_checkpoint, map_location=device)
    # 忽略额外的键值，解决动态创建的层加载问题
    unet.load_state_dict(unet_checkpoint['model_state_dict'], strict=False)
    print(f"LDM模型已加载: {args.ldm_checkpoint}")
    
    # 创建潜在扩散模型
    ldm = LatentDiffusionModel(
        unet=unet,
        vae=vae,
        latent_dim=args.latent_dim,
        device=device,
        noise_schedule=args.noise_schedule,
        use_ema=True
    ).to(device)
    
    # 生成样本
    print(f"生成 {args.num_samples} 个样本...")
    ldm.eval()
    with torch.no_grad():
        # 如果需要展示生成过程
        if args.show_process:
            # 创建一个单独的样本，保存中间步骤
            steps = [0, 100, 200, 400, 600, 800, 999]  # 选择要显示的时间步
            samples_process = []
            
            print("生成diffusion过程可视化...")
            for t in steps:
                sample_t = ldm.sample(n=1, start_t=t, return_intermediates=False)
                sample_t = (sample_t + 1) / 2  # 反归一化到[0, 1]
                samples_process.append(sample_t[0])
            
            # 保存过程图
            plt.figure(figsize=(15, 3))
            for i, sample in enumerate(samples_process):
                plt.subplot(1, len(steps), i+1)
                plt.imshow(sample.permute(1, 2, 0).cpu().numpy())
                plt.title(f"Step {steps[i]}")
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, "generation_process.png"))
            plt.close()
        
        # 生成完整样本
        samples = ldm.sample(n=args.num_samples)
        
        # 反归一化到[0, 1]
        samples = (samples + 1) / 2
        
        # 保存每个样本
        for i in range(args.num_samples):
            save_image(samples[i], os.path.join(sample_dir, f"sample_{i+1}.png"))
        
        # 保存网格图像
        save_image(samples, os.path.join(sample_dir, "samples_grid.png"), nrow=int(args.num_samples**0.5))
    
    print(f"样本已保存到 {sample_dir} 目录")
    
    # 如果指定了参考图像，生成重建比较
    if args.reference_dir:
        print("\n生成原始-重建图像对比...")
        from utils.dataset import ImageDataset
        from torch.utils.data import DataLoader
        from torchvision import transforms
        
        # 创建测试数据集
        test_transforms = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        test_dataset = ImageDataset(args.reference_dir, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=args.num_samples, shuffle=True)
        
        # 获取一批图像
        real_images = next(iter(test_loader))
        
        # 最多处理指定数量的图像
        real_images = real_images[:args.num_samples].to(device)
        
        # 使用VAE进行重建
        with torch.no_grad():
            reconstructed, _, _ = vae(real_images)
            
            # 保存原始图像和重建图像的对比
            comparison = torch.cat([
                real_images[:8], 
                reconstructed[:8]
            ])
            comparison = (comparison + 1) / 2  # 反归一化
            save_image(comparison, os.path.join(sample_dir, "reconstruction_comparison.png"), nrow=8)
            
            print(f"重建比较已保存到 {os.path.join(sample_dir, 'reconstruction_comparison.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的模型生成图像")
    parser.add_argument("--vae_checkpoint", type=str, required=True, help="VAE模型检查点路径")
    parser.add_argument("--ldm_checkpoint", type=str, required=True, help="LDM模型检查点路径")
    parser.add_argument("--num_samples", type=int, default=16, help="要生成的样本数量")
    parser.add_argument("--latent_dim", type=int, default=256, help="潜在空间维度")
    parser.add_argument("--image_size", type=int, default=256, help="图像大小")
    parser.add_argument("--num_embeddings", type=int, default=8192, help="编码本大小")
    parser.add_argument("--commitment_cost", type=float, default=0.15, help="承诺损失系数 (默认: 0.15)")
    parser.add_argument("--attention_resolutions", type=str, default="16,8,4", help="注意力分辨率 (默认: 16,8,4)")
    parser.add_argument("--noise_schedule", type=str, default="cosine", help="噪声调度类型 (默认: cosine)")
    parser.add_argument("--show_process", action="store_true", help="可视化生成过程")
    parser.add_argument("--reference_dir", type=str, default="", help="参考图像目录，用于生成重建比较")
    
    args = parser.parse_args()
    
    generate_samples(args) 

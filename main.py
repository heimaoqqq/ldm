import argparse
import os
import torch
from train_vae import train_vae
from train_ldm import train_ldm

def main(args):
    """主函数，训练VAE和LDM模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    print("=" * 50)
    print(f"使用设备: {device}")
    
    # 确认CUDA是否可用
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
    else:
        print("警告: 未检测到可用GPU，将使用CPU训练 (速度会很慢)")
    print("=" * 50)
    
    # 第一阶段：训练VAE
    best_vae_path = None
    if args.vae_epochs > 0:
        print("\n\n" + "=" * 20 + " 第一阶段：训练VAE " + "=" * 20 + "\n")
        
        # 构建VAE训练参数
        vae_args = argparse.Namespace(
            data_dir=args.data_dir,
            image_size=args.image_size,
            batch_size=args.vae_batch_size,
            epochs=args.vae_epochs,
            lr=args.vae_lr,
            latent_dim=args.latent_dim,
            num_embeddings=args.num_embeddings,
            commitment_cost=args.commitment_cost,
            disc_filters=args.disc_filters,
            save_interval=args.save_interval,
            resume=args.vae_resume
        )
        
        # 训练VAE，获取最佳模型路径
        best_vae_path = train_vae(vae_args)
    
    # 第二阶段：训练LDM
    if args.ldm_epochs > 0:
        print("\n\n" + "=" * 20 + " 第二阶段：训练LDM " + "=" * 20 + "\n")
        
        # 如果VAE检查点没有指定，使用最新训练的VAE
        if not args.vae_checkpoint:
            if best_vae_path:
                args.vae_checkpoint = best_vae_path
                print(f"使用最佳VAE模型: {args.vae_checkpoint}")
            elif args.vae_epochs > 0:
                # 获取最新的检查点时间戳目录
                latest_checkpoint_dir = None
                latest_time = 0
                
                for dir_name in os.listdir('checkpoints'):
                    dir_path = os.path.join('checkpoints', dir_name)
                    if os.path.isdir(dir_path):
                        try:
                            # 尝试解析目录名中的时间戳
                            time_value = int(dir_name)
                            if time_value > latest_time:
                                latest_time = time_value
                                latest_checkpoint_dir = dir_path
                        except ValueError:
                            # 如果目录名不是时间戳格式，跳过
                            continue
                
                if latest_checkpoint_dir:
                    # 优先使用最佳模型，如果存在
                    best_model_path = os.path.join(latest_checkpoint_dir, "vae_best.pth")
                    if os.path.exists(best_model_path):
                        args.vae_checkpoint = best_model_path
                    else:
                        args.vae_checkpoint = os.path.join(latest_checkpoint_dir, "vae_final.pth")
                    print(f"使用VAE检查点: {args.vae_checkpoint}")
                else:
                    print("未找到有效的VAE检查点，请指定--vae_checkpoint参数")
                    return
            else:
                print("必须指定VAE检查点或先训练VAE模型")
                return
        
        # 构建LDM训练参数
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
            sample_interval=args.sample_interval,
            save_interval=args.save_interval,
            vae_checkpoint=args.vae_checkpoint,
            resume=args.ldm_resume
        )
        
        # 训练LDM
        train_ldm(ldm_args)
    
    print("\n训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两阶段潜在扩散模型训练")
    
    # 通用参数
    parser.add_argument("--data_dir", type=str, default="dataset", help="数据集目录")
    parser.add_argument("--image_size", type=int, default=256, help="图像大小")
    parser.add_argument("--latent_dim", type=int, default=256, help="潜在空间维度")
    parser.add_argument("--num_embeddings", type=int, default=8192, help="编码本大小")
    parser.add_argument("--commitment_cost", type=float, default=0.25, help="承诺损失系数")
    parser.add_argument("--save_interval", type=int, default=1, help="保存检查点的间隔（轮次）")
    parser.add_argument("--sample_interval", type=int, default=100, help="生成样本的间隔（批次）")
    
    # VAE参数
    parser.add_argument("--vae_epochs", type=int, default=0, help="VAE训练轮数")
    parser.add_argument("--vae_batch_size", type=int, default=16, help="VAE批次大小")
    parser.add_argument("--vae_lr", type=float, default=1e-4, help="VAE学习率")
    parser.add_argument("--disc_filters", type=int, default=64, help="判别器基础过滤器数量")
    parser.add_argument("--vae_resume", type=str, default="", help="恢复VAE训练的检查点文件")
    
    # LDM参数
    parser.add_argument("--ldm_epochs", type=int, default=0, help="LDM训练轮数")
    parser.add_argument("--ldm_batch_size", type=int, default=8, help="LDM批次大小")
    parser.add_argument("--ldm_lr", type=float, default=1e-4, help="LDM学习率")
    parser.add_argument("--noise_steps", type=int, default=1000, help="扩散步数")
    parser.add_argument("--noise_schedule", type=str, default="cosine", help="噪声调度类型，可选: linear或cosine")
    parser.add_argument("--attention_resolutions", type=str, default="16,8", help="在指定分辨率启用注意力，如'16,8'")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪最大范数")
    parser.add_argument("--vae_checkpoint", type=str, default="", help="预训练VAE检查点文件，用于LDM训练")
    parser.add_argument("--ldm_resume", type=str, default="", help="恢复LDM训练的检查点文件")
    
    args = parser.parse_args()
    
    main(args) 
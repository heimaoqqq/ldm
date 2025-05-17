import argparse
import matplotlib.pyplot as plt
import os
import torch
from torchvision.utils import make_grid
import numpy as np
from PIL import Image
import glob

def visualize_training_results(args):
    """可视化训练过程和生成结果"""
    # 查找训练样本目录
    samples_dirs = glob.glob(os.path.join("samples", "*"))
    if not samples_dirs:
        print("找不到训练样本目录")
        return
    
    # 使用最新的样本目录
    latest_samples_dir = max(samples_dirs, key=os.path.getctime)
    print(f"使用样本目录: {latest_samples_dir}")
    
    # 创建保存可视化结果的目录
    vis_dir = "visualization"
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 1. 可视化VAE重建过程
    recon_images = sorted(glob.glob(os.path.join(latest_samples_dir, "*recon*.png")))
    if recon_images:
        plt.figure(figsize=(15, 10))
        plt.suptitle("VAE重建过程", fontsize=16)
        
        for i, img_path in enumerate(recon_images[:min(9, len(recon_images))]):
            img = Image.open(img_path)
            plt.subplot(3, 3, i+1)
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "vae_reconstruction.png"))
        plt.close()
        print(f"已保存VAE重建可视化到 {os.path.join(vis_dir, 'vae_reconstruction.png')}")
    
    # 2. 可视化VAE采样结果
    sample_images = sorted(glob.glob(os.path.join(latest_samples_dir, "*samples*.png")))
    if sample_images:
        plt.figure(figsize=(15, 10))
        plt.suptitle("VAE采样结果", fontsize=16)
        
        for i, img_path in enumerate(sample_images[:min(9, len(sample_images))]):
            img = Image.open(img_path)
            plt.subplot(3, 3, i+1)
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "vae_samples.png"))
        plt.close()
        print(f"已保存VAE采样可视化到 {os.path.join(vis_dir, 'vae_samples.png')}")
    
    # 3. 可视化LDM生成结果
    ldm_images = sorted(glob.glob(os.path.join(latest_samples_dir, "ldm_*.png")))
    if ldm_images:
        plt.figure(figsize=(15, 10))
        plt.suptitle("LDM生成结果", fontsize=16)
        
        for i, img_path in enumerate(ldm_images[:min(9, len(ldm_images))]):
            img = Image.open(img_path)
            plt.subplot(3, 3, i+1)
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "ldm_samples.png"))
        plt.close()
        print(f"已保存LDM生成可视化到 {os.path.join(vis_dir, 'ldm_samples.png')}")
    
    # 4. 可视化最终生成的样本
    generated_samples = glob.glob(os.path.join("generated_samples", "sample_*.png"))
    if generated_samples:
        plt.figure(figsize=(15, 10))
        plt.suptitle("最终生成的样本", fontsize=16)
        
        for i, img_path in enumerate(generated_samples[:min(16, len(generated_samples))]):
            img = Image.open(img_path)
            plt.subplot(4, 4, i+1)
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "final_samples.png"))
        plt.close()
        print(f"已保存最终生成样本可视化到 {os.path.join(vis_dir, 'final_samples.png')}")
    
    print("可视化完成，结果保存在 visualization 目录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化训练过程和生成结果")
    args = parser.parse_args()
    
    visualize_training_results(args) 
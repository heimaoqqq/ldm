import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import os
from datetime import datetime

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """保存模型检查点
    
    参数:
        model: 要保存的模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失值
        filename: 保存的文件名
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"模型检查点已保存至 {filename}")

def load_checkpoint(model, optimizer, filename):
    """加载模型检查点
    
    参数:
        model: 要加载检查点的模型
        optimizer: 优化器
        filename: 检查点文件名
    
    返回:
        epoch: 加载的轮次
        loss: 加载的损失值
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"模型检查点已从 {filename} 加载")
    return epoch, loss

def denormalize(tensor):
    """反归一化张量，从[-1,1]转换到[0,1]
    
    参数:
        tensor: 输入的归一化张量
    
    返回:
        反归一化的张量，范围[0,1]
    """
    return (tensor + 1) / 2

def save_samples(model, device, batch_size, image_size, latent_dim, sample_dir, filename):
    """保存模型生成的样本
    
    参数:
        model: 生成模型（通常是解码器）
        device: 计算设备
        batch_size: 批次大小
        image_size: 图像大小
        latent_dim: 潜在空间维度
        sample_dir: 保存目录
        filename: 保存文件名
    """
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # 生成随机潜在向量
    z = torch.randn(1, latent_dim, image_size // 16, image_size // 16).to(device)
    
    # 使用模型生成样本
    with torch.no_grad():
        # 检查是否为VQVAE类型的模型，如有decode方法则使用，否则直接forward
        if hasattr(model, 'decode'):
            samples = model.decode(z)
        else:
            samples = model(z)
    
    # 反归一化并保存图像
    samples = denormalize(samples)
    save_path = os.path.join(sample_dir, filename)
    save_image(samples[0], save_path)
    print(f"样本已保存到 {save_path}")

def plot_reconstruction(original, reconstructed, sample_dir, filename):
    """绘制原始图像和重建图像的对比
    
    参数:
        original: 原始图像
        reconstructed: 重建图像
        sample_dir: 保存目录
        filename: 保存文件名
    """
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # 确保张量在CPU上并转换为numpy数组
    original = denormalize(original.cpu())
    reconstructed = denormalize(reconstructed.cpu())
    
    # 只使用第一张图像进行对比
    plt.figure(figsize=(10, 5))
    
    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original[0].permute(1, 2, 0).numpy())
    plt.title("原始图像")
    plt.axis('off')
    
    # 重建图像
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed[0].permute(1, 2, 0).numpy())
    plt.title("重建图像")
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(sample_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"重建图像对比已保存到 {save_path}")

def setup_directories():
    """设置必要的目录结构"""
    dirs = {
        'checkpoints': 'checkpoints',
        'samples': 'samples',
        'logs': 'logs'
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for key, path in dirs.items():
        dirs[key] = os.path.join(path, timestamp)
        if not os.path.exists(dirs[key]):
            os.makedirs(dirs[key])
    
    return dirs 
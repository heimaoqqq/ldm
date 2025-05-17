import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import random

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        参数:
            root_dir (str): 数据集根目录路径
            transform (callable, optional): 图像转换函数
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 获取所有图像文件路径
        self.image_paths = []
        for dir_path, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(dir_path, filename))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def get_data_loaders(data_dir, image_size=256, batch_size=32, train_val_test_ratio=(0.8, 0.1, 0.1)):
    """创建训练集、验证集和测试集的数据加载器
    
    参数:
        data_dir (str): 数据集根目录
        image_size (int): 图像大小
        batch_size (int): 批次大小
        train_val_test_ratio (tuple): 训练集、验证集、测试集的比例
    
    返回:
        train_loader, val_loader, test_loader: 训练集、验证集和测试集的数据加载器
    """
    # 确保比例之和为1
    assert sum(train_val_test_ratio) == 1.0, "比例之和必须为1.0"
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建完整数据集
    full_dataset = ImageDataset(
        root_dir=data_dir,
        transform=transform
    )
    
    # 计算数据集总大小
    total_size = len(full_dataset)
    
    # 计算每个子集的大小
    train_size = int(total_size * train_val_test_ratio[0])
    val_size = int(total_size * train_val_test_ratio[1])
    test_size = total_size - train_size - val_size  # 确保总数一致
    
    # 输出数据集信息
    print(f"\n数据集信息:")
    print(f"总图片数量: {total_size}")
    print(f"训练集: {train_size}张图片 ({train_val_test_ratio[0]*100:.1f}%)")
    print(f"验证集: {val_size}张图片 ({train_val_test_ratio[1]*100:.1f}%)")
    print(f"测试集: {test_size}张图片 ({train_val_test_ratio[2]*100:.1f}%)")
    print(f"图像大小: {image_size}x{image_size}，批次大小: {batch_size}\n")
    
    # 设置随机种子，确保可复现性
    torch.manual_seed(42)
    
    # 创建数据集的随机排列
    indices = torch.randperm(total_size).tolist()
    
    # 划分数据集
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

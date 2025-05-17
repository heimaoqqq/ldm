import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不一致，添加1x1卷积进行调整
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # 跳跃连接
        out = F.relu(out)
        
        return out

class DownsampleBlock(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.res1 = ResidualBlock(out_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class UpsampleBlock(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.res1 = ResidualBlock(out_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class VectorQuantizer(nn.Module):
    """向量量化层"""
    def __init__(self, num_embeddings=8192, embedding_dim=256, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 创建嵌入向量
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    
    def forward(self, inputs):
        # 输入形状: [B, C, H, W]
        input_shape = inputs.shape
        
        # 调整输入为[B*H*W, C]以便进行量化
        flat_input = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = flat_input.view(-1, self.embedding_dim)
        
        # 计算与每个嵌入向量的距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                   + torch.sum(self.embedding.weight**2, dim=1)
                   - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # 找到最近的嵌入向量索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # 转换为one-hot编码
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化操作，将输入替换为最近的嵌入向量
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        # 计算损失
        # 计算编码器损失: 使输入靠近量化向量
        q_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # 计算承诺损失: 使量化向量靠近输入
        e_latent_loss = F.mse_loss(quantized, inputs.detach())
        # 组合损失
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 直通估计器
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices

class Encoder(nn.Module):
    """编码器"""
    def __init__(self, in_channels=3, latent_dim=256):
        super(Encoder, self).__init__()
        
        # 下采样层
        self.down1 = DownsampleBlock(in_channels, 64)         # 256x256 -> 128x128
        self.down2 = DownsampleBlock(64, 128)                # 128x128 -> 64x64
        self.down3 = DownsampleBlock(128, 256)               # 64x64 -> 32x32
        self.down4 = DownsampleBlock(256, latent_dim)        # 32x32 -> 16x16
        
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x

class Decoder(nn.Module):
    """解码器"""
    def __init__(self, out_channels=3, latent_dim=256):
        super(Decoder, self).__init__()
        
        # 上采样层
        self.up1 = UpsampleBlock(latent_dim, 256)           # 16x16 -> 32x32
        self.up2 = UpsampleBlock(256, 128)                 # 32x32 -> 64x64
        self.up3 = UpsampleBlock(128, 64)                  # 64x64 -> 128x128
        self.up4 = UpsampleBlock(64, 64)                   # 128x128 -> 256x256
        
        # 输出层
        self.output = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.output(x)
        x = torch.tanh(x)  # 输出范围[-1, 1]
        return x

class PatchGANDiscriminator(nn.Module):
    """PatchGAN判别器"""
    def __init__(self, in_channels=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # 70x70 PatchGAN
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x  # 不使用sigmoid，因为使用BCEWithLogitsLoss

class VQVAE(nn.Module):
    """Vector Quantized VAE"""
    def __init__(self, in_channels=3, latent_dim=256, num_embeddings=8192, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels, latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(in_channels, latent_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        
        return x_recon, vq_loss, quantized
    
    def encode(self, x):
        """编码函数，返回量化后的向量表示"""
        z = self.encoder(x)
        quantized, _, indices = self.vq_layer(z)
        return quantized, indices
    
    def decode(self, quantized):
        """解码函数，从量化的表示重建图像"""
        return self.decoder(quantized) 
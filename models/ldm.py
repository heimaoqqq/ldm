import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ResidualBlock(nn.Module):
    """U-Net中的残差块"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 改进的残差连接，确保维度匹配
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        # 添加额外的激活层和归一化层，增强特征传播
        self.dropout = nn.Dropout(0.1)  # 添加dropout增强泛化能力
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)  # 添加dropout
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 强化残差连接
        out = out + residual
        out = F.relu(out)
        
        return out

class AttentionBlock(nn.Module):
    """自注意力机制块"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        
        self.mha = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        self.ln1 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        batch_size, c, h, w = x.shape
        
        # 将特征图重塑为序列
        x_seq = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # 自注意力
        attn_output, _ = self.mha(x_seq, x_seq, x_seq)
        x_seq = x_seq + attn_output
        x_seq = self.ln1(x_seq)
        
        # 前馈网络
        ff_output = self.ff(x_seq)
        x_seq = x_seq + ff_output
        
        # 重塑回特征图
        x = x_seq.permute(0, 2, 1).reshape(batch_size, c, h, w)
        
        return x

class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels, add_attention=False):
        super(DownBlock, self).__init__()
        self.res1 = ResidualBlock(in_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.add_attention = add_attention
        if add_attention:
            self.attention = AttentionBlock(out_channels)
    
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        if self.add_attention:
            x = self.attention(x)
        x_skip = x
        x = self.downsample(x)
        return x, x_skip

class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels, add_attention=False):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        # 当与skip连接时，通道数需要调整
        self.res1 = ResidualBlock(out_channels * 2, out_channels)  # *2 是因为有跳跃连接
        self.res2 = ResidualBlock(out_channels, out_channels)
        self.add_attention = add_attention
        if add_attention:
            self.attention = AttentionBlock(out_channels)
    
    def forward(self, x, skip):
        # 上采样
        x = self.upsample(x)
        
        # 跳跃连接，确保尺寸匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        # 拼接特征
        x = torch.cat([x, skip], dim=1)
        
        # 通过残差块处理
        x = self.res1(x)
        x = self.res2(x)
        
        # 如果启用了注意力，则应用注意力机制
        if self.add_attention:
            x = self.attention(x)
            
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    """时间嵌入"""
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim
    
    def forward(self, time):
        """
        time: [B]，每个批次样本的时间步
        """
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return embeddings

class UNet(nn.Module):
    """扩散模型的U-Net架构"""
    def __init__(self, in_channels=256, time_dim=256, attention_resolutions=None):
        super(UNet, self).__init__()
        
        # 注意力分辨率，默认在16x16和8x8分辨率启用
        if attention_resolutions is None:
            attention_resolutions = [16, 8]
        self.attention_resolutions = attention_resolutions
        
        # 确保输入通道等于嵌入维度
        self.in_channels = in_channels
        
        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        
        # 编码器下采样路径，根据注意力分辨率决定是否添加注意力
        self.down1 = DownBlock(256, 512, add_attention=16 in attention_resolutions)    # 16x16 -> 8x8
        self.down2 = DownBlock(512, 512, add_attention=8 in attention_resolutions)     # 8x8 -> 4x4
        self.down3 = DownBlock(512, 512, add_attention=4 in attention_resolutions)     # 4x4 -> 2x2
        
        # 中间层 - 增加残差连接
        self.mid_res1 = ResidualBlock(512, 512)
        self.mid_attn = AttentionBlock(512)
        self.mid_res2 = ResidualBlock(512, 512)
        
        # 解码器上采样路径，根据注意力分辨率决定是否添加注意力
        self.up1 = UpBlock(512, 512, add_attention=4 in attention_resolutions)         # 2x2 -> 4x4
        self.up2 = UpBlock(512, 512, add_attention=8 in attention_resolutions)         # 4x4 -> 8x8
        
        # 修改up3为更简单的结构，使用UpBlock类，但确保通道数匹配
        self.up3 = UpBlock(512, 256, add_attention=16 in attention_resolutions)      # 8x8 -> 16x16
        
        # 时间嵌入注入
        self.time_embed1 = nn.Linear(time_dim, 256)
        self.time_embed2 = nn.Linear(time_dim, 512)
        self.time_embed3 = nn.Linear(time_dim, 512)
        self.time_embed4 = nn.Linear(time_dim, 512)
        
        # 输出层
        self.final_res = ResidualBlock(256, 256)
        self.final_conv = nn.Conv2d(256, in_channels, kernel_size=1)
        
        print(f"UNet初始化完成，注意力将在分辨率{self.attention_resolutions}启用")
    
    def forward(self, x, t):
        """
        x: [B, C, H, W] 输入噪声潜在表示
        t: [B] 时间步
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 初始卷积
        h = self.init_conv(x)
        h = h + self.time_embed1(t_emb)[:, :, None, None]
        skip1 = h  # 保存第一个跳跃连接
        
        # 编码器路径
        h, skip2 = self.down1(h)
        h = h + self.time_embed2(t_emb)[:, :, None, None]
        
        h, skip3 = self.down2(h)
        h = h + self.time_embed3(t_emb)[:, :, None, None]
        
        h, skip4 = self.down3(h)
        h = h + self.time_embed4(t_emb)[:, :, None, None]
        
        # 中间层
        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)
        
        # 解码器路径
        h = self.up1(h, skip4)
        h = self.up2(h, skip3)
        h = self.up3(h, skip1)  # 使用简化的UpBlock结构
        
        # 输出层
        h = self.final_res(h)
        h = self.final_conv(h)
        
        return h

class DiffusionModel:
    """扩散模型类，处理前向扩散过程和逆向采样"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=16, device="cuda", schedule_type="cosine"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule_type = schedule_type
        
        # 根据选择的调度类型定义噪声调度
        if schedule_type == "linear":
            self.betas = self._linear_beta_schedule()
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule()
        else:
            raise ValueError(f"不支持的噪声调度类型: {schedule_type}，可选: linear或cosine")
        
        print(f"使用{schedule_type}噪声调度")
        
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def _linear_beta_schedule(self):
        """线性噪声调度"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps, device=self.device)
    
    def _cosine_beta_schedule(self):
        """余弦噪声调度"""
        steps = self.noise_steps + 1
        s = 0.008
        x = torch.linspace(0, self.noise_steps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def add_noise(self, x, t):
        """给x添加t时刻的噪声"""
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        # 噪声公式: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def _extract(self, a, t, x_shape):
        """从a中提取对应于t时刻的元素，并reshape到与x相同的维度"""
        batch_size = t.shape[0]
        # 确保t与a在同一设备上
        t = t.to(a.device)
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def sample_timesteps(self, n):
        """随机采样n个时间步"""
        return torch.randint(1, self.noise_steps, (n,), device=self.device).long()
    
    @torch.no_grad()
    def sample(self, model, n, latent_dim=256, channels=256):
        """使用DDPM采样过程从噪声生成图像"""
        model.eval()
        # 从随机噪声开始
        x = torch.randn(n, channels, self.img_size, self.img_size, device=self.device)
        
        for i in reversed(range(1, self.noise_steps)):
            t = torch.ones(n, device=self.device).long() * i
            predicted_noise = model(x, t)
            
            # DDPM采样公式
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
        model.train()
        x = torch.clamp(x, -1, 1)
        return x

class LatentDiffusionModel(nn.Module):
    """潜在扩散模型，结合预训练的VQVAE和扩散U-Net"""
    def __init__(self, unet, vae, latent_dim=256, device="cuda", noise_schedule="cosine"):
        super(LatentDiffusionModel, self).__init__()
        self.unet = unet
        self.vae = vae
        self.latent_dim = latent_dim
        self.device = device
        self.noise_schedule = noise_schedule
        self.diffusion = DiffusionModel(device=device, schedule_type=noise_schedule)
        
        # 冻结VAE参数
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def forward(self, x, t):
        """
        x: 原始图像 [B, 3, H, W]
        t: 时间步 [B]
        """
        # 使用VAE编码到潜在空间
        with torch.no_grad():
            z, _ = self.vae.encode(x)
        
        # 添加扩散过程的噪声
        noised_z, noise = self.diffusion.add_noise(z, t)
        
        # 预测噪声
        predicted_noise = self.unet(noised_z, t)
        
        return noise, predicted_noise, z
    
    @torch.no_grad()
    def sample(self, n=1):
        """从随机噪声采样生成图像"""
        # 从扩散模型采样潜在表示
        z = self.diffusion.sample(self.unet, n, latent_dim=self.latent_dim)
        
        # 通过VAE解码器生成图像
        samples = self.vae.decode(z)
        
        return samples 
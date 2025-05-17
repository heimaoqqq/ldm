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

class FrequencyEnhancedBlock(nn.Module):
    """频域增强模块，专门为微多普勒信号设计"""
    def __init__(self, channels):
        super(FrequencyEnhancedBlock, self).__init__()
        self.conv_freq = nn.Conv2d(channels, channels, kernel_size=(5, 3), padding=(2, 1), groups=channels//2)
        self.conv_time = nn.Conv2d(channels, channels, kernel_size=(3, 5), padding=(1, 2), groups=channels//2)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        residual = x
        
        # 分别处理频率和时间域特征
        freq_feat = self.conv_freq(x)
        time_feat = self.conv_time(x)
        
        # 组合特征
        combined = freq_feat + time_feat
        combined = self.bn(combined)
        combined = self.act(combined)
        combined = self.proj(combined)
        
        return combined + residual

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
        # 添加频域增强模块
        self.use_freq_enhance = out_channels <= 512  # 在较小的特征图上使用频域增强
        if self.use_freq_enhance:
            self.freq_enhance = FrequencyEnhancedBlock(out_channels)
    
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        if self.add_attention:
            x = self.attention(x)
        if self.use_freq_enhance:
            x = self.freq_enhance(x)
        x_skip = x
        x = self.downsample(x)
        return x, x_skip

class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels, add_attention=False):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        # 记录通道数
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 因为拼接可能导致不同的输入通道数，这里需要灵活处理
        self.res1 = ResidualBlock(out_channels * 2, out_channels)  # 默认情况下通道数翻倍
        self.res2 = ResidualBlock(out_channels, out_channels)
        
        # 添加通道调整层，处理可能的通道数不匹配情况
        self.channel_adjust = None
        
        self.add_attention = add_attention
        if add_attention:
            self.attention = AttentionBlock(out_channels)
            
        # 添加频域增强模块
        self.use_freq_enhance = out_channels <= 512  # 在较小的特征图上使用频域增强
        if self.use_freq_enhance:
            self.freq_enhance = FrequencyEnhancedBlock(out_channels)
    
    def forward(self, x, skip):
        # 上采样
        x = self.upsample(x)
        
        # 跳跃连接，确保尺寸匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
        # 拼接特征
        combined = torch.cat([x, skip], dim=1)
        
        # 检查通道数是否匹配预期，如果不匹配，动态创建调整层
        expected_channels = self.out_channels * 2
        if combined.shape[1] != expected_channels and self.channel_adjust is None:
            print(f"创建通道调整层: {combined.shape[1]} -> {expected_channels}")
            self.channel_adjust = nn.Conv2d(combined.shape[1], expected_channels, kernel_size=1).to(x.device)
        
        # 应用通道调整（如果需要）
        if self.channel_adjust is not None:
            combined = self.channel_adjust(combined)
            
        # 通过残差块处理
        x = self.res1(combined)
        x = self.res2(x)
        
        # 如果启用了注意力，则应用注意力机制
        if self.add_attention:
            x = self.attention(x)
            
        # 应用频域增强
        if self.use_freq_enhance:
            x = self.freq_enhance(x)
            
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
        
        # 注意力分辨率，默认在16x16、8x8和4x4分辨率启用
        if attention_resolutions is None:
            attention_resolutions = [16, 8, 4, 2]  # 增加更细粒度的注意力层
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
        
        # 编码器下采样路径，增加一层以捕获更多频域特征
        self.down1 = DownBlock(256, 512, add_attention=16 in self.attention_resolutions)    # 16x16
        self.down2 = DownBlock(512, 512, add_attention=8 in self.attention_resolutions)     # 8x8
        self.down3 = DownBlock(512, 512, add_attention=4 in self.attention_resolutions)     # 4x4
        self.down4 = DownBlock(512, 768, add_attention=2 in self.attention_resolutions)     # 2x2，增加新的下采样层
        
        # 中间层 - 增加中间层残差块数量和增强注意力机制
        self.mid_res1 = ResidualBlock(768, 768)
        self.mid_attn = AttentionBlock(768)
        self.mid_res2 = ResidualBlock(768, 768)
        self.mid_res3 = ResidualBlock(768, 768)  # 增加额外残差块
        
        # 解码器上采样路径，增加一层以匹配编码器
        self.up1 = UpBlock(768, 512, add_attention=2 in self.attention_resolutions)       # 2x2，新的上采样层
        self.up2 = UpBlock(512, 512, add_attention=4 in self.attention_resolutions)       # 4x4
        self.up3 = UpBlock(512, 512, add_attention=8 in self.attention_resolutions)       # 8x8
        self.up4 = UpBlock(512, 256, add_attention=16 in self.attention_resolutions)      # 16x16
        
        # 时间嵌入注入 - 确保维度匹配
        self.time_embed1 = nn.Linear(time_dim, 256)  # 匹配初始层 256通道
        self.time_embed2 = nn.Linear(time_dim, 512)  # 匹配down1层 512通道
        self.time_embed3 = nn.Linear(time_dim, 512)  # 匹配down2层 512通道
        self.time_embed4 = nn.Linear(time_dim, 512)  # 匹配down3层 512通道
        self.time_embed5 = nn.Linear(time_dim, 768)  # 匹配down4层 768通道
        
        # 增强输出层
        self.final_res1 = ResidualBlock(256, 256)
        self.final_res2 = ResidualBlock(256, 256)  # 增加额外残差块
        self.final_conv = nn.Conv2d(256, in_channels, kernel_size=1)
        
        # 频域特征增强模块：专为微多普勒时频图设计
        self.freq_enhance = FrequencyEnhancedBlock(256)
        
        # 添加专门处理频率特征的模块
        self.freq_attn = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(5, 1), padding=(2, 0)),  # 垂直方向（频率）卷积
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=1)
        )
        
        print(f"增强版UNet初始化完成，注意力将在分辨率{self.attention_resolutions}启用")
    
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
        
        # 应用频域注意力增强
        h = h + self.freq_attn(h)
        h = self.freq_enhance(h)
        
        # 编码器路径
        h, skip1 = self.down1(h)  # 获取第一个skip连接 (16x16)
        h = h + self.time_embed2(t_emb)[:, :, None, None]
        
        h, skip2 = self.down2(h)  # 获取第二个skip连接 (8x8)
        h = h + self.time_embed3(t_emb)[:, :, None, None]
        
        h, skip3 = self.down3(h)  # 获取第三个skip连接 (4x4)
        h = h + self.time_embed4(t_emb)[:, :, None, None]
        
        h, skip4 = self.down4(h)  # 获取第四个skip连接 (2x2)
        h = h + self.time_embed5(t_emb)[:, :, None, None]
        
        # 中间层
        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)
        h = self.mid_res3(h)  # 增加的中间层
        
        # 解码器路径 - 注意正确的skip连接顺序
        h = self.up1(h, skip4)  # 使用2x2分辨率的skip4
        h = self.up2(h, skip3)  # 使用4x4分辨率的skip3
        h = self.up3(h, skip2)  # 使用8x8分辨率的skip2
        h = self.up4(h, skip1)  # 使用16x16分辨率的skip1
        
        # 增强输出
        h = self.final_res1(h)
        h = self.final_res2(h)  # 增加的输出层残差块
        
        # 输出层
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

class EMA:
    """指数移动平均模型，用于提高扩散模型的采样质量"""
    def __init__(self, beta=0.9999):
        self.beta = beta
        self.shadow = {}
        self.backup = {}
    
    def register(self, model):
        """注册模型参数"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """更新EMA参数"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    new_average = (1.0 - self.beta) * param.data + self.beta * self.shadow[name]
                    self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        """应用EMA参数进行推理"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        """恢复原始模型参数"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class LatentDiffusionModel(nn.Module):
    """潜在扩散模型，结合预训练的VQVAE和扩散U-Net"""
    def __init__(self, unet, vae, latent_dim=256, device="cuda", noise_schedule="cosine", use_ema=True):
        super(LatentDiffusionModel, self).__init__()
        self.unet = unet
        self.vae = vae
        self.latent_dim = latent_dim
        self.device = device
        self.noise_schedule = noise_schedule
        self.diffusion = DiffusionModel(device=device, schedule_type=noise_schedule)
        
        # 添加EMA模型支持
        self.use_ema = use_ema
        if self.use_ema:
            self.ema = EMA(beta=0.9999)
            self.ema.register(self.unet)
        
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
    
    def update_ema(self):
        """更新EMA模型"""
        if self.use_ema:
            self.ema.update(self.unet)
    
    @torch.no_grad()
    def sample(self, n=1):
        """从随机噪声采样生成图像"""
        # 使用EMA模型采样
        if self.use_ema:
            self.ema.apply_shadow(self.unet)
        
        # 从扩散模型采样潜在表示
        z = self.diffusion.sample(self.unet, n, latent_dim=self.latent_dim)
        
        # 通过VAE解码器生成图像
        samples = self.vae.decode(z)
        
        # 恢复原始模型
        if self.use_ema:
            self.ema.restore(self.unet)
            
        return samples 

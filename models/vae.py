import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """通道注意力模块 (Squeeze-and-Excitation Block)"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class FrequencyEnhancedBlock(nn.Module):
    """频域增强模块，特别适用于微多普勒时频图"""
    def __init__(self, channels):
        super(FrequencyEnhancedBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels//2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels//4)
        self.bn = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        
    def forward(self, x):
        residual = x
        
        # 垂直方向卷积处理频率特征
        freq_feat = self.conv1(x)
        # 水平方向卷积处理时间特征
        time_feat = self.conv2(x)
        
        # 合并特征
        out = freq_feat + time_feat
        out = self.bn(out)
        out = F.leaky_relu(out, 0.2)
        
        # 使用SE注意力增强重要通道
        out = self.se(out)
        
        return out + residual

class SelfAttention(nn.Module):
    """自注意力模块"""
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 生成查询、键、值矩阵
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, height * width)
        
        # 计算注意力映射
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        # 注意力加权的值
        proj_value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        return out

class ResidualBlock(nn.Module):
    """增强版残差块"""
    def __init__(self, in_channels, out_channels, use_se=False):
        super(ResidualBlock, self).__init__()
        self.use_se = use_se
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 添加SE注意力块
        if use_se:
            self.se = SEBlock(out_channels)
        
        # 如果输入输出通道数不一致，添加1x1卷积进行调整
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        
        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用SE注意力
        if self.use_se:
            out = self.se(out)
        
        out += residual  # 跳跃连接
        out = F.leaky_relu(out, 0.2)
        
        return out

class DownsampleBlock(nn.Module):
    """增强版下采样模块"""
    def __init__(self, in_channels, out_channels, use_attention=False, use_freq=False):
        super(DownsampleBlock, self).__init__()
        self.use_attention = use_attention
        self.use_freq = use_freq
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.res1 = ResidualBlock(out_channels, out_channels, use_se=True)
        self.res2 = ResidualBlock(out_channels, out_channels, use_se=True)
        
        # 添加自注意力机制
        if use_attention:
            self.attention = SelfAttention(out_channels)
            
        # 添加频域特征增强
        if use_freq:
            self.freq_enhance = FrequencyEnhancedBlock(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.res1(x)
        x = self.res2(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        if self.use_freq:
            x = self.freq_enhance(x)
            
        return x

class UpsampleBlock(nn.Module):
    """增强版上采样模块"""
    def __init__(self, in_channels, out_channels, use_attention=False, use_freq=False):
        super(UpsampleBlock, self).__init__()
        self.use_attention = use_attention
        self.use_freq = use_freq
        
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.res1 = ResidualBlock(out_channels, out_channels, use_se=True)
        self.res2 = ResidualBlock(out_channels, out_channels, use_se=True)
        
        # 添加自注意力机制
        if use_attention:
            self.attention = SelfAttention(out_channels)
            
        # 添加频域特征增强
        if use_freq:
            self.freq_enhance = FrequencyEnhancedBlock(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.res1(x)
        x = self.res2(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        if self.use_freq:
            x = self.freq_enhance(x)
            
        return x

class VectorQuantizer(nn.Module):
    """增强的向量量化层，使用EMA更新编码本"""
    def __init__(self, num_embeddings=8192, embedding_dim=256, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 创建嵌入向量
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA相关变量
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.weight.data.clone())
    
    def forward(self, inputs, use_ema=True):
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
        
        # 训练时使用EMA更新编码本
        if self.training and use_ema:
            # 更新嵌入向量cluster size
            encodings_sum = encodings.sum(0)
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * encodings_sum
            
            # 强制处理未使用的嵌入向量
            n = self.ema_cluster_size.sum()
            cluster_size = self.ema_cluster_size + self.epsilon
            self.ema_cluster_size = cluster_size
            
            # 更新嵌入向量
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
            
            # 正规化嵌入权重
            self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)
        
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
    """增强版编码器"""
    def __init__(self, in_channels=3, latent_dim=256, use_attention=True, use_freq=True):
        super(Encoder, self).__init__()
        
        self.use_attention = use_attention
        self.use_freq = use_freq
        
        # 初始处理层
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.init_bn = nn.BatchNorm2d(64)
        self.init_act = nn.LeakyReLU(0.2)
        
        # 频域处理模块
        if use_freq:
            self.freq_init = FrequencyEnhancedBlock(64)
        
        # 下采样层
        self.down1 = DownsampleBlock(64, 128, use_attention=False, use_freq=use_freq)          # 256x256 -> 128x128
        self.down2 = DownsampleBlock(128, 256, use_attention=use_attention, use_freq=use_freq) # 128x128 -> 64x64
        self.down3 = DownsampleBlock(256, 512, use_attention=use_attention, use_freq=use_freq) # 64x64 -> 32x32
        self.down4 = DownsampleBlock(512, latent_dim, use_attention=use_attention, use_freq=use_freq) # 32x32 -> 16x16
        
        # 额外的处理层
        self.final_res = ResidualBlock(latent_dim, latent_dim, use_se=True)
        
    def forward(self, x):
        # 初始处理
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_act(x)
        
        if self.use_freq:
            x = self.freq_init(x)
        
        # 下采样路径
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        # 最终处理
        x = self.final_res(x)
        
        return x

class Decoder(nn.Module):
    """增强版解码器"""
    def __init__(self, out_channels=3, latent_dim=256, use_attention=True, use_freq=True):
        super(Decoder, self).__init__()
        
        self.use_attention = use_attention
        self.use_freq = use_freq
        
        # 初始处理层
        self.init_res = ResidualBlock(latent_dim, latent_dim, use_se=True)
        
        # 上采样层
        self.up1 = UpsampleBlock(latent_dim, 512, use_attention=use_attention, use_freq=use_freq)    # 16x16 -> 32x32
        self.up2 = UpsampleBlock(512, 256, use_attention=use_attention, use_freq=use_freq)          # 32x32 -> 64x64
        self.up3 = UpsampleBlock(256, 128, use_attention=use_attention, use_freq=use_freq)          # 64x64 -> 128x128
        self.up4 = UpsampleBlock(128, 64, use_attention=False, use_freq=use_freq)                  # 128x128 -> 256x256
        
        # 输出层处理
        self.final_res = ResidualBlock(64, 64, use_se=True)
        
        # 频域增强
        if use_freq:
            self.freq_final = FrequencyEnhancedBlock(64)
            
        # 输出层
        self.output = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 初始处理
        x = self.init_res(x)
        
        # 上采样路径
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        
        # 最终处理
        x = self.final_res(x)
        
        if self.use_freq:
            x = self.freq_final(x)
        
        # 输出层
        x = self.output(x)
        x = torch.tanh(x)  # 输出范围[-1, 1]
        
        return x
    
    def decode(self, x):
        """添加decode方法与forward保持一致，解决调用兼容性问题"""
        return self.forward(x)

class SpectralNorm:
    """谱归一化装饰器"""
    def __init__(self, name='weight'):
        self.name = name
        self.power_iterations = 1

    def compute_weight(self, module):
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        w = getattr(module, self.name)

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = F.normalize(torch.mv(torch.t(w.view(height, -1).data), u.data), dim=0)
            u.data = F.normalize(torch.mv(w.view(height, -1).data, v.data), dim=0)

        sigma = u.dot(torch.mv(w.view(height, -1), v))
        return w / sigma.expand_as(w)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        height = weight.data.shape[0]
        width = weight.view(height, -1).data.shape[1]

        u = nn.Parameter(torch.FloatTensor(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(torch.FloatTensor(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0)
        v.data = F.normalize(v.data, dim=0)

        setattr(module, name + '_u', u)
        setattr(module, name + '_v', v)

        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, nn.Parameter(weight))

def spectral_norm(module, name='weight'):
    """应用谱归一化"""
    SpectralNorm.apply(module, name)
    return module

class PatchGANDiscriminator(nn.Module):
    """增强的PatchGAN判别器，支持谱归一化"""
    def __init__(self, in_channels=3, ndf=64, use_spectral_norm=True):
        super(PatchGANDiscriminator, self).__init__()
        
        # 使用谱归一化层处理判别器
        norm_layer = lambda x: spectral_norm(x) if use_spectral_norm else x
        
        # 70x70 PatchGAN
        self.layer1 = nn.Sequential(
            norm_layer(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            norm_layer(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 自注意力层
        self.attention = SelfAttention(ndf * 8)
        
        # 输出层
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 应用自注意力
        x = self.attention(x)
        
        x = self.layer5(x)
        return x  # 不使用sigmoid，因为使用BCEWithLogitsLoss

class PerceptualLoss(nn.Module):
    """感知损失模块，基于预训练网络提取特征"""
    def __init__(self, weight=1.0):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        
        # 使用一个轻量级网络，这里假设使用部分预训练网络
        # 在实际使用时，可以替换为加载预训练的VGG或ResNet模型
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # 冻结参数
        for param in self.layers.parameters():
            param.requires_grad = False
    
    def forward(self, x, target):
        # 提取特征
        x_features = self.layers(x)
        target_features = self.layers(target)
        
        # 计算L1损失
        loss = F.l1_loss(x_features, target_features) * self.weight
        return loss

class VQVAE(nn.Module):
    """增强版Vector Quantized VAE"""
    def __init__(self, in_channels=3, latent_dim=256, num_embeddings=8192, commitment_cost=0.25, 
                 use_attention=True, use_freq=True, use_ema=True, use_perceptual=False):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels, latent_dim, use_attention, use_freq)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(in_channels, latent_dim, use_attention, use_freq)
        
        # 保存配置
        self.commitment_cost = commitment_cost  # 保存承诺损失权重
        self.use_ema = use_ema                  # 是否使用EMA更新编码本
        self.use_perceptual = use_perceptual    # 是否使用感知损失
        
        # 初始化感知损失
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(weight=0.1)
        
    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, _ = self.vq_layer(z, use_ema=self.use_ema)
        x_recon = self.decoder(quantized)
        
        return x_recon, vq_loss, quantized
    
    def encode(self, x):
        """编码函数，返回量化后的向量表示"""
        z = self.encoder(x)
        quantized, _, indices = self.vq_layer(z, use_ema=self.use_ema)
        return quantized, indices
    
    def decode(self, quantized):
        """解码函数，从量化的表示重建图像"""
        return self.decoder(quantized)
    
    def compute_loss(self, x, recon_x, vq_loss, adv_loss, recon_criterion):
        """计算总损失，包括重建损失、VQ损失、感知损失和对抗损失的动态平衡"""
        # 计算重建损失
        recon_loss = recon_criterion(recon_x, x)
        
        # 添加感知损失(如果启用)
        perceptual_loss = 0
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(recon_x, x)
        
        # 动态平衡各种损失
        # 对抗损失权重系数在训练过程中可以进一步调整
        total_loss = recon_loss + perceptual_loss - self.commitment_cost * adv_loss + vq_loss
        
        return total_loss, recon_loss 

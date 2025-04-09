import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

################################################################################
# SpatialInteractionAttention
################################################################################
class SpatialInteractionAttention(nn.Module):
    def __init__(self, in_channels, reduction=8, pooled_len=256):
        super(SpatialInteractionAttention, self).__init__()
        self.reduction = reduction
        self.pooled_len = pooled_len

        # 将输入特征 x 先映射到 (C//reduction)
        self.embed_psi = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )
        self.embed_phi = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )
        # 拼接后特征维度 = in_channels + pooled_len
        self.fc1 = nn.Linear(in_channels + pooled_len, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W  # 空间位置数

        # 1. 计算空间亲和矩阵，先将通道降维
        psi_feat = self.embed_psi(x)       # (B, C//reduction, H, W)
        phi_feat = self.embed_phi(x)       # (B, C//reduction, H, W)
        psi_reshaped = psi_feat.view(B, -1, N)    # (B, C//reduction, N)
        phi_reshaped = phi_feat.view(B, -1, N)      # (B, C//reduction, N)
        affinity = torch.bmm(psi_reshaped.transpose(1, 2), phi_reshaped)  # (B, N, N)

        # 2. 拼接亲和矩阵及其转置 -> (B, N, 2N)
        r_i = torch.cat([affinity, affinity.transpose(1, 2)], dim=2)

        # 3. 对 concat 后的结果做池化，得到 (B, N, pooled_len)
        r_i_pooled = F.adaptive_avg_pool1d(r_i, self.pooled_len)

        # 4. 对原始 x 在通道维度全局池化 -> (B, C)，再扩展到 (B, N, C)
        x_reshaped = x.view(B, C, N)
        x_pooled = F.adaptive_avg_pool1d(x_reshaped, 1).squeeze(-1)  # (B, C)
        x_pooled = x_pooled.unsqueeze(1).repeat(1, N, 1)

        # 5. 拼接后过全连接，生成每个空间位置的注意力
        att_input = torch.cat([x_pooled, r_i_pooled], dim=2)  # (B, N, C + pooled_len)
        s = F.relu(self.fc1(att_input))
        a = torch.sigmoid(self.fc2(s)).view(B, 1, H, W)
        return x * a

################################################################################
# ChannelInteractionAttention
################################################################################
class ChannelInteractionAttention(nn.Module):
    def __init__(self, in_channels, reduction=8, pooled_len=256):
        super(ChannelInteractionAttention, self).__init__()
        self.reduction = reduction
        self.pooled_len = pooled_len
        # fc1 输入维度 = in_channels + pooled_len
        self.fc1 = nn.Linear(in_channels + pooled_len, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # 1. 对空间维度全局池化 -> (B, C, 1)
        x_flat = x.view(B, C, -1)
        x_pool = x_flat.mean(dim=2, keepdim=True)  # (B, C, 1)

        # 2. 构造通道间亲和矩阵：用外积计算 (B, C, C)
        x_c = x_pool.squeeze(-1)  # (B, C)
        x_c_a = x_c.unsqueeze(2)  # (B, C, 1)
        x_c_b = x_c.unsqueeze(1)  # (B, 1, C)
        affinity = x_c_a * x_c_b   # (B, C, C)

        # 拼接转置 -> (B, C, 2C)
        r_i = torch.cat([affinity, affinity.transpose(1, 2)], dim=2)

        # 3. 池化到 pooled_len -> (B, C, pooled_len)
        r_i_pooled = F.adaptive_avg_pool1d(r_i, self.pooled_len)

        # 4. x 全局池化得到 (B, C)，扩展至 (B, C, C)
        x_pooled = x_c_b.repeat(1, C, 1)  # (B, C, C)

        # 5. 拼接 -> (B, C, C + pooled_len)，全连接求注意力
        att_input = torch.cat([x_pooled, r_i_pooled], dim=2)
        s = F.relu(self.fc1(att_input))
        a = torch.sigmoid(self.fc2(s)).view(B, C, 1, 1)
        return x * a

################################################################################
# DecoderBlock
################################################################################
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, pooled_len=256):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )
        self.spatial_attn = SpatialInteractionAttention(out_channels, pooled_len=pooled_len)
        self.channel_attn = ChannelInteractionAttention(out_channels, pooled_len=pooled_len)

    def forward(self, x):
        x_conv = self.conv(x)
        x_spatial = self.spatial_attn(x_conv)
        x_channel = self.channel_attn(x_spatial)
        return x_channel

################################################################################
# IGAFEncoder
################################################################################
class IGAFEncoder(nn.Module):
    def __init__(self):
        super(IGAFEncoder, self).__init__()
        wrn = models.wide_resnet50_2(pretrained=True)
        self.block1 = nn.Sequential(
            wrn.conv1,
            wrn.bn1,
            wrn.relu,
            wrn.maxpool,
            wrn.layer1  # 输出约 (B, 256, H/4, W/4)
        )
        self.block2 = wrn.layer2      # 输出约 (B, 512, H/8, W/8)
        self.block3 = wrn.layer3      # 输出约 (B, 1024, H/16, W/16)
        self.block4 = wrn.layer4      # 输出约 (B, 2048, H/32, W/32)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x1, x2, x3, x4

################################################################################
# IGAFDecoder
################################################################################
class IGAFDecoder(nn.Module):
    def __init__(self, out_channel=1):
        super(IGAFDecoder, self).__init__()
        # 对于 decoder4，由于输入分辨率较低 (8x8)，设 pooled_len 为 128
        self.decoder4 = DecoderBlock(in_channels=2048, mid_channels=1024, out_channels=1024, pooled_len=128)
        # 其他阶段设为默认 256
        self.decoder3 = DecoderBlock(in_channels=1024, mid_channels=512, out_channels=512, pooled_len=256)
        self.decoder2 = DecoderBlock(in_channels=512, mid_channels=256, out_channels=256, pooled_len=256)
        self.decoder1 = DecoderBlock(in_channels=256, mid_channels=128, out_channels=64, pooled_len=256)
        self.final_conv = nn.Conv2d(64, out_channel, kernel_size=1)

        # 跳跃连接采用 1x1 卷积 + ReLU
        self.skip3 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x4, x3, x2, x1, input_size):
        # Stage 4：最深层特征解码
        d4 = self.decoder4(x4)
        d4_up = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)

        # Stage 3：上采样并与 x3 跳跃连接
        skip3_feat = self.skip3(x3)
        d3_input = d4_up + skip3_feat
        d3 = self.decoder3(d3_input)
        d3_up = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)

        # Stage 2：上采样并与 x2 跳跃连接
        skip2_feat = self.skip2(x2)
        d2_input = d3_up + skip2_feat
        d2 = self.decoder2(d2_input)
        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)

        # Stage 1：上采样并与 x1 跳跃连接
        skip1_feat = self.skip1(x1)
        d1_input = d2_up + skip1_feat
        d1 = self.decoder1(d1_input)
        d1_up = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=False)
        out = self.final_conv(d1_up)
        return out

################################################################################
# IGAFNet
################################################################################
class IGAFNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(IGAFNet, self).__init__()
        # 如果输入通道不为 3，则先调整为 3 通道以兼容预训练的 WideResNet50
        if in_channel != 3:
            self.input_adjust = nn.Conv2d(in_channel, 3, kernel_size=1)
        else:
            self.input_adjust = None
        self.encoder = IGAFEncoder()
        self.decoder = IGAFDecoder(out_channel)

    def forward(self, x):
        input_size = x.shape[2:]  # (H, W)
        if self.input_adjust is not None:
            x = self.input_adjust(x)
        x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(x4, x3, x2, x1, input_size)
        return out

# if __name__ == "__main__":
#     # 测试用：输入 3×256×256
#     model = IGAFNet(in_channel=3, out_channel=1)
#     x_test = torch.randn(1, 3, 256, 256)
#     y = model(x_test)
#     print("Output shape:", y.shape)  # 期望为 (1, 1, 256, 256)
#     print("Model OK!")

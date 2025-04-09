import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialInteractionAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SpatialInteractionAttention, self).__init__()
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
        self.fc1 = nn.Linear(in_channels + 2 * 256, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_reshaped = x.view(B, C, N).permute(0, 2, 1)
        psi = self.embed_psi(x).view(B, -1, N).permute(0, 2, 1)
        phi = self.embed_phi(x).view(B, -1, N)
        affinity = torch.bmm(psi, phi)
        r_i = torch.cat([affinity, affinity.permute(0, 2, 1)], dim=-1)
        r_i_pooled = F.adaptive_avg_pool1d(r_i, 256)
        x_pooled = F.adaptive_avg_pool1d(x_reshaped.permute(0, 2, 1), 1).squeeze(-1)
        x_pooled = x_pooled.unsqueeze(1).repeat(1, N, 1)
        att_input = torch.cat([x_pooled, r_i_pooled], dim=-1)
        s = F.relu(self.fc1(att_input))
        a = torch.sigmoid(self.fc2(s)).view(B, 1, H, W)
        return x * a

class ChannelInteractionAttention(nn.Module):
    def __init__(self, in_channels, spatial_size=256, reduction=8):
        super(ChannelInteractionAttention, self).__init__()
        self.embed_psi = nn.Sequential(
            nn.Conv1d(spatial_size, spatial_size // reduction, kernel_size=1),
            nn.BatchNorm1d(spatial_size // reduction),
            nn.ReLU(inplace=True)
        )
        self.embed_phi = nn.Sequential(
            nn.Conv1d(spatial_size, spatial_size // reduction, kernel_size=1),
            nn.BatchNorm1d(spatial_size // reduction),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(in_channels + 2 * 256, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        psi = self.embed_psi(x_flat).permute(0, 2, 1)
        phi = self.embed_phi(x_flat)
        affinity = torch.bmm(psi, phi.permute(0, 2, 1))
        r_i = torch.cat([affinity, affinity.permute(0, 2, 1)], dim=-1)
        r_i_pooled = F.adaptive_avg_pool1d(r_i, 256)
        x_pooled = F.adaptive_avg_pool1d(x_flat, 1).squeeze(-1).unsqueeze(1).repeat(1, C, 1)
        att_input = torch.cat([x_pooled, r_i_pooled], dim=-1)
        s = F.relu(self.fc1(att_input))
        a = torch.sigmoid(self.fc2(s)).view(B, C, 1, 1)
        return x * a

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )
        self.spatial_attn = SpatialInteractionAttention(out_channels)
        self.channel_attn = ChannelInteractionAttention(out_channels, spatial_size=256)

    def forward(self, x):
        x_conv = self.conv(x)
        x_spatial = self.spatial_attn(x_conv)
        x_channel = self.channel_attn(x_spatial)
        return x_channel

class IGAFEncoder(nn.Module):
    def __init__(self):
        super(IGAFEncoder, self).__init__()
        wrn = models.wide_resnet50_2(pretrained=True)
        self.block1 = nn.Sequential(
            wrn.conv1,
            wrn.bn1,
            wrn.relu,
            wrn.maxpool,
            wrn.layer1
        )
        self.block2 = wrn.layer2
        self.block3 = wrn.layer3
        self.block4 = wrn.layer4

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x1, x2, x3, x4

class IGAFDecoder(nn.Module):
    def __init__(self, out_channel=1):
        super(IGAFDecoder, self).__init__()
        self.decoder4 = DecoderBlock(in_channels=2048, mid_channels=1024, out_channels=1024)
        self.decoder3 = DecoderBlock(in_channels=1024, mid_channels=512, out_channels=512)
        self.decoder2 = DecoderBlock(in_channels=512, mid_channels=256, out_channels=256)
        self.decoder1 = DecoderBlock(in_channels=256, mid_channels=128, out_channels=64)
        self.final_conv = nn.Conv2d(64, out_channel, kernel_size=1)

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
        d4 = self.decoder4(x4)
        d4_up = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        skip3_feat = self.skip3(x3)
        d3_input = d4_up + skip3_feat
        d3 = self.decoder3(d3_input)
        d3_up = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        skip2_feat = self.skip2(x2)
        d2_input = d3_up + skip2_feat
        d2 = self.decoder2(d2_input)
        d2_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        skip1_feat = self.skip1(x1)
        d1_input = d2_up + skip1_feat
        d1 = self.decoder1(d1_input)
        d1_up = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=False)
        out = self.final_conv(d1_up)
        return out

class IGAFNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(IGAFNet, self).__init__()
        if in_channel != 3:
            self.input_adjust = nn.Conv2d(in_channel, 3, kernel_size=1)
        else:
            self.input_adjust = None
        self.encoder = IGAFEncoder()
        self.decoder = IGAFDecoder(out_channel=out_channel)

    def forward(self, x):
        input_size = x.shape[2:]
        if self.input_adjust is not None:
            x = self.input_adjust(x)
        x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(x4, x3, x2, x1, input_size)
        return out

if __name__ == "__main__":
    model = IGAFNet(in_channel=3, out_channel=1)
    print(model)


# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器部分（下采样）
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        # 解码器部分（上采样）
        self.dec4 = self.deconv_block(1024, 512)
        self.dec3 = self.deconv_block(512, 256)
        self.dec2 = self.deconv_block(256, 128)
        self.dec1 = self.deconv_block(128, 64)

        # 最后一个卷积层，将通道数转换为类别数
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # 为了恢复至256x256，再加一次上采样
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_channels, out_channels):
        """卷积块，包含卷积层、ReLU激活和最大池化"""
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return block

    def deconv_block(self, in_channels, out_channels):
        """解卷积块，用于上采样"""
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # 编码器（下采样）
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # 解码器（上采样）
        dec4 = self.dec4(enc5)
        dec3 = self.dec3(dec4 + enc4)  # 跳跃连接
        dec2 = self.dec2(dec3 + enc3)  # 跳跃连接
        dec1 = self.dec1(dec2 + enc2)  # 跳跃连接

        # 最后一层卷积
        out = self.final_conv(dec1 + enc1)  # 跳跃连接

        # 使用上采样调整到256x256
        out = self.final_upsample(out)  # 使用双线性插值上采样

        return out

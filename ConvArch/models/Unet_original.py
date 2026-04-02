import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(无填充卷积 => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 原版精髓 1: padding=0 (PyTorch默认就是0，这里显式写出)
            # 每次卷积，特征图尺寸宽高各减 2
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样：最大池化 + 双卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样：转置卷积 + 中心裁剪拼接 + 双卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 是下方传上来的特征图 (较小)
        x1 = self.up(x1)

        # x2 是左侧跳跃连接过来的特征图 (较大)
        # 原版精髓 2: 计算尺寸差异，进行中心裁剪 (Center Cropping)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 裁剪掉 x2 上下左右多余的部分
        x2_cropped = x2[:, :,
        diffY // 2: diffY // 2 + x1.size()[2],
        diffX // 2: diffX // 2 + x1.size()[3]]

        # 拼接并进行卷积
        x = torch.cat([x2_cropped, x1], dim=1)
        return self.conv(x)


class UNetOriginal(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetOriginal, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 通道数完全按照论文 Figure 1 设定
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


# --- 测试原版架构的尺寸变化 ---
if __name__ == '__main__':
    # 按照论文，输入必须是 572x572
    x = torch.randn((1, 3, 572, 572))
    # 实例化模型
    model = UNetOriginal(n_channels=3, n_classes=21)

    out = model(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(卷积 => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # padding=1 保证了卷积后尺寸不缩小
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), # 现代改良：加入BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
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
    """上采样：转置卷积拼接特征 + 双卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用转置卷积进行上采样（也就是论文里的上卷积）
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 是上一层传上来的图，x2 是左边跳跃连接过来的高清图
        x1 = self.up(x1)

        # 将左边的 x2 和下方的 x1 在通道维度（dim=1）进行拼接 (Concatenate)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """1x1 卷积降维到类别数"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # 输入通道，通常RGB图为3
        self.n_classes = n_classes  # 输出类别，VOC2012为21

        # 1. 最开始的进入网络的双卷积
        self.inc = DoubleConv(n_channels, 64)

        # 2. 左侧的下采样路径 (Contracting Path)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # 3. 右侧的上采样路径 (Expansive Path)
        # 注意通道数的变化，比如 up1 接收下方传来的 1024 和左边传来的 512
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # 4. 最后的 1x1 卷积输出
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 左侧一路向下，记住每一层的输出，留着做跳跃连接！
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # x5 是最底层的特征

        # 右侧一路向上，每次都要带上左边对应的特征
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 得到最终分类概率图
        logits = self.outc(x)
        return logits


# 测试一下模型能否跑通
if __name__ == '__main__':
    # 模拟一张 VOC2012 的输入图片: Batch=1, Channel=3, 尺寸 512x512
    x = torch.randn((1, 3, 512, 512))
    # 实例化模型 (输入3通道，VOC2012一共21个类别)
    model = UNet(n_channels=3, n_classes=21)
    # 前向传播
    out = model(x)
    print("输入尺寸:", x.shape)
    print("输出尺寸:", out.shape)
    # 期望输出尺寸应该是 [1, 21, 512, 512]
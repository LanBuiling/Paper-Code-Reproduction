import torch
import torch.nn as nn
from torchvision import models


class FCN32s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN32s, self).__init__()

        # 1. 搬来 VGG16 的特征提取底座 (预训练权重)
        vgg16 = models.vgg16(weights='DEFAULT')
        features = list(vgg16.features.children())

        # 将 VGG16 的 features 部分全部拿过来，这部分包含了 5 个池化层
        # 经过这部分后，图像尺寸被缩小了 2^5 = 32 倍
        self.features = nn.Sequential(*features)

        # 2. 全连接层转卷积层 (Convolutionalized) -> 对应论文 3.1 节
        # 原版 VGG16 的全连接层(fc6)输入是 7x7 的特征图，所以这里卷积核设为 7
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()  # 论文 4.3 节提到使用了 Dropout

        # 原版 VGG16 的 fc7，转为 1x1 卷积
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # 3. 降维打分 -> 对应论文 4.1 节
        # 追加一个 1x1 卷积，把 4096 个通道压缩成 21 个通道（20个类别+1个背景）
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # 4. 简单粗暴的反卷积上采样 -> 对应论文 3.3 节
        # 步长 stride=32，意味着长宽各放大 32 倍。
        # (kernel_size=64, padding=16 是标准的双线性插值反卷积参数设定，保证尺寸严格放大32倍)
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes,
                                          kernel_size=64, stride=32, padding=16, bias=False)

    def forward(self, x):
        # x 的输入形状通常是 [batch_size, 3, H, W]

        # 提取特征，输出尺寸变为原图的 1/32
        x = self.features(x)

        # 经过卷积化的全连接层
        x = self.drop6(self.relu6(self.fc6(x)))
        x = self.drop7(self.relu7(self.fc7(x)))

        # 对深层特征图的每个像素点进行 21 类别打分
        x = self.score_fr(x)

        # 放大 32 倍，恢复到输入图片的尺寸 [batch_size, 21, H, W]
        x = self.upscore(x)

        return x
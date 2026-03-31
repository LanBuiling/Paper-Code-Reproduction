import torch
import torch.nn as nn
from torchvision import models


class FCN16s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN16s, self).__init__()

        vgg16 = models.vgg16(weights='DEFAULT')
        features = list(vgg16.features.children())

        # 截取 VGG16 的不同阶段
        # pool4 之前的部分 (经过4次池化，尺寸缩小 16倍)
        self.pool4 = nn.Sequential(*features[:24])
        # pool4 到 pool5 的部分 (再经过1次池化，总共缩小 32倍)
        self.pool5 = nn.Sequential(*features[24:])

        # 全卷积化
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # 分别给深层(fc7)和浅层(pool4)打分
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)

        # 上采样层
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1,
                                           bias=False)  # 放大2倍
        self.upscore16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, padding=8,
                                            bias=False)  # 放大16倍

    def forward(self, x):
        # 1. 提取浅层 pool4 特征 (1/16 尺寸)
        p4 = self.pool4(x)
        # 2. 提取深层特征 (1/32 尺寸)
        p5 = self.pool5(p4)

        # 3. 经过全卷积层
        fc7 = self.drop7(self.relu7(self.fc7(self.drop6(self.relu6(self.fc6(p5))))))

        # 4. 深层打分并放大2倍 (使其尺寸和 pool4 一样大)
        score_fr = self.score_fr(fc7)
        upscore2 = self.upscore2(score_fr)

        # 5. 浅层打分
        score_pool4 = self.score_pool4(p4)

        # 【核心跳跃融合】：相加！
        # 注意：实际运算中因为 padding 原因可能差 1-2 个像素，严格复现需要 crop 对齐
        # 为了代码能直接跑，这里假设输入尺寸刚好能被 32 整除
        fuse = upscore2 + score_pool4

        # 6. 融合后一次性放大 16 倍，恢复原图大小
        out = self.upscore16(fuse)
        return out